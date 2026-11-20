#!/usr/bin/env python3
"""标定"marker → 物体"的固定偏移，写回 config/gt_markers.yaml。

同一帧里同时拿到 ArUco 位姿和管线给出的物体 OBB，把物体中心/朝向换算到 marker 系，
平均 N 帧后写入 t_marker_obj / rpy_marker_obj。

注意：这样标出来的 GT 与管线"自洽"，适合做 **A/B 对比与抖动评估**；
绝对精度仍需手工量一次偏移（两者可以互相核对，差得多说明标定或几何有问题）。

用法:
    python3 scripts/calibrate_gt_offset.py --marker-id 0 --label book --frames 120
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_gt_aruco import (make_detector, detect, solve_planar, rpy_to_R,  # noqa: E402
                           yaw_up, DEFAULT_K)


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / max(n, 1e-12)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/gt_markers.yaml')
    ap.add_argument('--marker-id', type=int, default=0)
    ap.add_argument('--label', default='book')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--marker-size', type=float, default=None)
    ap.add_argument('--dims', default=None,
                    help='期望尺寸 l,w,h（米，逗号分隔）；给了就只匹配尺寸接近的实例')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    marker_size = float(args.marker_size or cfg.get('marker_size_m', 0.096))
    want_dims = [float(x) for x in args.dims.split(',')] if args.dims else None
    dict_name = cfg.get('dict', 'DICT_4X4_50')
    normal = np.asarray(cfg.get('normal', [0.0, -1.0, 0.0]), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    u_ax = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u_ax) < 1e-6:
        u_ax = np.cross(normal, [0.0, 0.0, 1.0])
    u_ax /= np.linalg.norm(u_ax)
    R_base = np.stack([u_ax, np.cross(normal, u_ax), normal], axis=1)
    obj_pts = np.array([[-marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, -marker_size / 2, 0],
                        [-marker_size / 2, -marker_size / 2, 0]])

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from astra_seg_interfaces.msg import ObjectClouds

    det = make_detector(dict_name)
    samples = []

    class Cal(Node):
        def __init__(self):
            super().__init__('calib_gt')
            self.marker = None
            self.book = None
            self.create_subscription(Image, '/camera/color/image_raw', self.on_img,
                                     qos_profile_sensor_data)
            self.create_subscription(ObjectClouds, '/astra_seg/objects', self.on_obj, 10)
            self.create_timer(0.05, self.try_collect)

        def on_img(self, msg):
            gray = cv2.cvtColor(np.frombuffer(bytes(msg.data), np.uint8)
                                .reshape(msg.height, msg.width, -1)[:, :, :3],
                                cv2.COLOR_BGR2GRAY)
            for mid, pts in detect(det, gray):
                if mid != args.marker_id:
                    continue
                p4 = solve_planar(pts, obj_pts, R_base)
                R = R_base @ rpy_to_R(0, 0, p4[3])
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p4[:3]
                self.marker = T

        def on_obj(self, msg):
            for i, lab in enumerate(msg.labels):
                if lab not in [x.strip() for x in args.label.split(',')]:
                    continue
                if want_dims is not None:
                    d = msg.obb_dims[i]
                    got = sorted([d.x, d.y, d.z], reverse=True)
                    exp = sorted(want_dims, reverse=True)
                    if any(abs(a - b) > 0.03 for a, b in zip(got, exp)):
                        continue
                p = msg.obb_poses[i].position
                q = msg.obb_poses[i].orientation
                self.book = (np.array([p.x, p.y, p.z]), quat_to_R(q.x, q.y, q.z, q.w),
                             msg.obb_dims[i])

        def try_collect(self):
            if self.marker is None or self.book is None:
                return
            c, R_b, _ = self.book
            Tm = self.marker
            t = Tm[:3, :3].T @ (c - Tm[:3, 3])          # 物体中心在 marker 系
            # 物理口径：两个 x 轴在相机水平面内的夹角（不经过 marker 系变换，
            # 否则基不一致会差十几度——踩过这个坑）
            yaw = math.radians(yaw_up(R_b) - yaw_up(Tm[:3, :3]))
            samples.append((t, yaw))
            if len(samples) % 10 == 0:
                print(f'  采集 {len(samples)}/{args.frames}', flush=True)
            if len(samples) >= args.frames:
                raise KeyboardInterrupt

    rclpy.init()
    node = Cal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not samples:
        print('\n没采到样本：检查 marker id / 物体标签 / 两者是否同时在画面里')
        return 1
    arr = np.array([s[0] for s in samples])
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0) * 1.4826 + 1e-6
    inl = np.all(np.abs(arr - med) < 4.0 * mad, axis=1)     # 去掉偶发翻转/错配帧
    if int(inl.sum()) >= max(10, int(0.5 * len(arr))):
        samples = [sm for sm, ok in zip(samples, inl) if ok]
        print(f'（剔除 {int((~inl).sum())} 个离群样本）')
    T = np.median(np.array([s[0] for s in samples]), axis=0)
    yaw = math.atan2(np.mean(np.sin([s[1] for s in samples])),
                     np.mean(np.cos([s[1] for s in samples])))
    tstd = np.std(np.array([s[0] for s in samples]), axis=0) * 1000
    print(f'\n样本 {len(samples)} 帧')
    print(f't_marker_obj = [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}] m  (std {tstd.round(1)} mm)')
    print(f'rpy_marker_obj = [0.0, 0.0, {yaw:.4f}] rad  ({math.degrees(yaw):.1f}°)')

    cfg.setdefault('markers', {})
    entry = cfg['markers'].get(args.marker_id) or cfg['markers'].get(str(args.marker_id)) or {}
    entry['label'] = args.label
    entry['t_marker_obj'] = [round(float(v), 4) for v in T]
    entry['rpy_marker_obj'] = [0.0, 0.0, round(float(yaw), 4)]
    cfg['markers'][args.marker_id] = entry
    with open(args.config, 'w') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f'已写入 {args.config}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
