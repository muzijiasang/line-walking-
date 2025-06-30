#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniCar 角度 PID 巡线（无 Web 界面 + “无检测保持转弯” + 彩色输出速度）
--------------------------------------------------------------
• 每帧对图像做车道线检测，只保留连通域面积在 [VALID_AREA, MAX_AREA] 之间的部分；
• 将每个切片的连通域质心与图像底部中点连线，计算该连线相对于竖直向上的角度（单位：度）；
• 取各切片角度平均绝对值并保留方向，作为 PID 输入，计算角速度 vz（单位：度/s）；
• “无检测时保持转弯”：如果当前帧在所有切片中都没有找到合法连通域（count == 0），
  则 vx = VX_CONST，vz = last_sign * MAX_VZ；
• 否则 vx 固定为 VX_CONST，vz 为 PID 计算结果并限幅在 [-MAX_VZ, +MAX_VZ]（单位：度/s）；
• 不使用 Web 界面，仅在后台运行；任何退出方式（包括 KeyboardInterrupt、SIGTERM），
  都会先发送零速度帧，再关闭串口与摄像头；
• 同时在终端以彩色打印实时的线速度和角速度。
"""

import os
import time
import cv2
import numpy as np
import serial
import struct
import argparse
import signal
import sys

# ───────── ANSI 彩色输出定义 ─────────
COLOR_RESET = "\033[0m"
COLOR_RED   = "\033[31m"   # 用于线速度 vx
COLOR_BLUE  = "\033[34m"   # 用于角速度 vz

# ───────── 摄像头 & 控制参数 ─────────
DEVICE        = int(os.getenv("CAM_DEVICE", 0))
CAM_WIDTH     = int(os.getenv("CAM_WIDTH", 1280))
CAM_HEIGHT    = int(os.getenv("CAM_HEIGHT", 720))
CAM_FPS       = int(os.getenv("CAM_FPS", 30))
CONTROL_FPS   = int(os.getenv("CONTROL_FPS", 15))
FRAME_INTERVAL = 1.0 / CONTROL_FPS

# ───────── 车道线算法参数 ─────────
ROI_RATIO  = float(os.getenv("LANE_ROI_RATIO", 0.35))
NUM_SLICES = int(os.getenv("LANE_SLICES", 5))
VALID_AREA = int(os.getenv("LANE_MIN_AREA", 3000))
MAX_AREA   = int(os.getenv("LANE_MAX_AREA", 20000))

HSV_LOWER = np.array([0, 0, 0],   np.uint8)
HSV_UPPER = np.array([180, 255, 60], np.uint8)
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ───────── 串口协议常量 ─────────
FRAME_HEADER = 0x7B
FRAME_TAIL   = 0x7D

# ───────── PID 控制器 （角度单位：度） ─────────
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float) -> float:
        p_term = self.kp * error
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        self.prev_error = error
        return p_term + i_term + d_term

# ───────── 构造并发送 串口 帧 ─────────
def build_frame(vx: float, vy: float, vz: float) -> bytes:
    frame = bytearray()
    frame.append(FRAME_HEADER)
    frame += struct.pack('<f', vx)
    frame += struct.pack('<f', vy)
    frame += struct.pack('<f', vz)
    frame.append(FRAME_TAIL)
    return bytes(frame)

# ───────── 全局 资源 ─────────
cap = None
ser = None
pid = None

# 固定线速度、最大角速度
VX_CONST = 0.5   # m/s
MAX_VZ   = 3.5   # deg/s

# ───────── 初始化 摄像头 ─────────
def init_camera(device: int, width: int, height: int, fps: int):
    global cap
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ───────── 车道线检测 + 角度误差 计算 ─────────
def compute_angle_error(frame: np.ndarray) -> (float, int):
    h, w = frame.shape[:2]
    roi_y0 = int(h * (1 - ROI_RATIO))
    slice_h = (h - roi_y0) // NUM_SLICES

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, 1)
    mask = cv2.dilate(mask, KERNEL, 1)

    sum_abs = 0.0
    sum_raw = 0.0
    count = 0

    for i in range(NUM_SLICES):
        y1 = roi_y0 + i * slice_h
        y2 = h if i == NUM_SLICES - 1 else (roi_y0 + (i + 1) * slice_h)
        slice_mask = mask[y1:y2]

        cnts, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        largest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < VALID_AREA or area > MAX_AREA:
            continue

        x, y, w_box, h_box = cv2.boundingRect(largest)
        cx = x + w_box // 2
        cy = y + h_box // 2 + y1

        dx = (w / 2.0) - float(cx)
        dy = float(h) - float(cy)
        angle_i = np.degrees(np.arctan2(dx, dy))

        sum_abs += abs(angle_i)
        sum_raw += angle_i
        count += 1

    if count == 0:
        return 0.0, 0

    avg_abs = sum_abs / count
    avg_raw = sum_raw / count
    direction = 1.0 if avg_raw > 1e-6 else (-1.0 if avg_raw < -1e-6 else 0.0)
    return avg_abs * direction, count

# ───────── 退出时 发送零速度 并 清理 ─────────
def send_zero_and_cleanup():
    global ser, cap
    try:
        if ser is not None and ser.is_open:
            zero_frame = build_frame(0.0, 0.0, 0.0)
            ser.write(zero_frame)
            time.sleep(0.05)
    except:
        pass
    try:
        if cap is not None:
            cap.release()
    except:
        pass
    try:
        if ser is not None and ser.is_open:
            ser.close()
    except:
        pass

def signal_handler(sig, frame):
    send_zero_and_cleanup()
    sys.exit(0)

# ───────── 主循环 ─────────
def main():
    global cap, ser, pid, VX_CONST, VALID_AREA, MAX_AREA

    parser = argparse.ArgumentParser(
        description='MiniCar 角度 PID 巡线（无 Web、无检测保持转弯、彩色输出）')
    parser.add_argument('--cam_device', type=int, default=DEVICE)
    parser.add_argument('--cam_width', type=int, default=CAM_WIDTH)
    parser.add_argument('--cam_height', type=int, default=CAM_HEIGHT)
    parser.add_argument('--cam_fps', type=int, default=CAM_FPS)
    parser.add_argument('--port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--vx', type=float, default=VX_CONST)
    parser.add_argument('--kp', type=float, default=0.005)
    parser.add_argument('--ki', type=float, default=0.000015)
    parser.add_argument('--kd', type=float, default=0.00001)
    parser.add_argument('--valid_area', type=int, default=VALID_AREA)
    parser.add_argument('--max_area', type=int, default=MAX_AREA)
    args = parser.parse_args()

    VALID_AREA = args.valid_area
    MAX_AREA   = args.max_area
    VX_CONST   = args.vx

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_camera(args.cam_device, args.cam_width, args.cam_height, args.cam_fps)
    print(f"[INFO] Camera 初始化: {args.cam_width}×{args.cam_height} @ {args.cam_fps} FPS")

    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=1.0)
        print(f"[INFO] 已打开串口 {args.port}, 波特率 {args.baudrate}")
    except Exception as e:
        ser = None
        print(f"[WARNING] 无法打开串口 {args.port}: {e}, 串口发送功能失效")

    dt = 1.0 / CONTROL_FPS
    pid = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, dt=dt)
    pid.reset()
    print(f"[INFO] PID 参数: KP={args.kp}, KI={args.ki}, KD={args.kd}, 线速度 vx={VX_CONST} m/s")
    print(f"[INFO] 连通域面积范围: [{VALID_AREA}, {MAX_AREA}] 像素")

    # 上一帧的转向方向，默认左转
    last_sign = 1.0

    try:
        last_time = time.time()
        while True:
            now = time.time()
            elapsed = now - last_time
            if elapsed < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - elapsed)
            last_time = time.time()

            ok, frame = cap.read()
            if not ok:
                continue

            error, count = compute_angle_error(frame)

            if count > 0:
                # 有检测：正常 PID 计算
                raw_vz = pid.compute(error)
                if raw_vz > MAX_VZ:
                    send_vz = MAX_VZ
                elif raw_vz < -MAX_VZ:
                    send_vz = -MAX_VZ
                else:
                    send_vz = raw_vz
                send_vx = VX_CONST
                # 更新转向方向
                last_sign = 1.0 if error > 1e-6 else (-1.0 if error < -1e-6 else last_sign)
            else:
                # 无检测：保持线速度并以最大角速度转弯
                send_vx = VX_CONST
                send_vz = last_sign * MAX_VZ

            vy = 0.0
            if ser is not None and ser.is_open:
                try:
                    ser.write(build_frame(send_vx, vy, send_vz))
                except Exception as e:
                    print(f"[ERROR] 串口发送异常: {e}")

            # 彩色终端输出
            sys.stdout.write(
                f"\r线速度 {COLOR_RED}{send_vx:+.2f} m/s{COLOR_RESET}   "
                f"角速度 {COLOR_BLUE}{send_vz:+.2f} °/s{COLOR_RESET}"
            )
            sys.stdout.flush()

    except Exception as ex:
        print(f"\n[ERROR] 运行异常: {ex}")
        send_zero_and_cleanup()
        sys.exit(1)
    finally:
        send_zero_and_cleanup()
        print()  # 换行
        print("[INFO] 程序退出，已发送零速度帧。")

if __name__ == '__main__':
    main()
