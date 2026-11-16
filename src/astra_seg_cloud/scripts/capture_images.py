#!/usr/bin/env python3
"""从相机抓帧存 PNG，用于做训练集。

用法：
    python3 scripts/capture_images.py --out ~/grasp_ds/raw --n 300 --interval 0.4
"""
import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Grab(Node):
    def __init__(self):
        super().__init__('capture_images')
        self.frame = None
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._cb, qos_profile_sensor_data)

    def _cb(self, msg):
        b = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
        self.frame = b[:, :, ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/grasp_ds/raw'))
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--interval', type=float, default=0.4, help='两帧间隔秒数')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init()
    node = Grab()
    saved = 0
    while saved < args.n:
        node.frame = None
        t0 = time.time()
        while node.frame is None and time.time() - t0 < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.frame is None:
            print('没收到画面，检查相机')
            break
        path = os.path.join(args.out, f'img_{saved:04d}.png')
        cv2.imwrite(path, node.frame)
        saved += 1
        print(f'{saved}/{args.n} -> {path}', flush=True)
        t_end = time.time() + args.interval
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(f'完成：{saved} 张 -> {args.out}')


if __name__ == '__main__':
    main()
