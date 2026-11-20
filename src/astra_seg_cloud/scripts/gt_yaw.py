#!/usr/bin/env python3
"""只做 yaw 的 GT：GT_yaw = ArUco 的 yaw + 设计常数（默认 +90°）。

为什么不标定：
  你已保证"书的短边 ∥ marker 的某条边"，所以书的长轴必然 ⟂ marker 的 x 轴，
  偏移就是 90°，是**设计常数**，不需要任何拟合/标定，也就不存在循环论证。
  （尺寸/位置 GT 不在本脚本范围内。）

用法:
    python3 scripts/gt_yaw.py --config config/gt_markers.yaml --minutes 3
输出:
    CSV: t, gt_yaw_deg, pred_yaw_deg, dyaw_deg, label, id
    统计: p50/p95/max |Δyaw|，以及 30 帧滑动均值/标准差
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_gt_aruco import make_detector, detect, solve_planar, rpy_to_R, yaw_up  # noqa: E402

DEFAULT_K = (285.1711, 285.1711, 159.5, 119.5)
UP = np.array([0.0, -1.0, 0.0])


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def wrap90(a):
    """折叠到 (-90, 90]：长方体的长轴没有方向，±180° 是同一个姿态。"""
    v = wrap180(a)
    if v > 90.0:
        v -= 180.0
    elif v <= -90.0:
        v += 180.0
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/gt_markers.yaml')
    ap.add_argument('--minutes', type=float, default=3.0)
    ap.add_argument('--csv', default=os.path.expanduser('~/gt_aruco/gt_yaw.csv'))
    ap.add_argument('--max-jump-cm', type=float, default=5.0)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) or {}
    dict_name = cfg.get('dict', 'DICT_4X4_50')
    size = float(cfg.get('marker_size_m', 0.096))
    yaw_off = float(cfg.get('yaw_offset_deg', 90.0))
    markers = cfg.get('markers', {}) or {}
    normal = np.asarray(cfg.get('normal', [0.0, -1.0, 0.0]), float)
    normal /= np.linalg.norm(normal)
    u_ax = np.cross(normal, [1.0, 0.0, 0.0]); u_ax /= np.linalg.norm(u_ax)
    R_base = np.stack([u_ax, np.cross(normal, u_ax), normal], axis=1)
    obj = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                    [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]])

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from astra_seg_interfaces.msg import ObjectClouds

    det = make_detector(dict_name)
    f = open(args.csv, 'w')
    f.write('t,label,id,gt_yaw_deg,pred_yaw_deg,dyaw_deg\n')
    t0 = None
    prev_c = None
    ys = []
    objs = []

    def quat_to_R(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        n = x * x + y * y + z * z + w * w
        s = 2.0 / max(n, 1e-12)
        return np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])

    class N(Node):
        def __init__(self):
            super().__init__('gt_yaw')
            self.create_subscription(Image, '/camera/color/image_raw', self.img,
                                     qos_profile_sensor_data)
            self.create_subscription(ObjectClouds, '/astra_seg/objects', self.obj, 10)

        def obj(self, m):
            objs.clear()
            for i, lab in enumerate(m.labels):
                objs.append((int(m.ids[i]), str(lab),
                             quat_to_R(m.obb_poses[i].orientation),
                             np.array([m.obb_poses[i].position.x,
                                       m.obb_poses[i].position.y,
                                       m.obb_poses[i].position.z])))

        def img(self, msg):
            nonlocal t0, prev_c
            g = cv2.cvtColor(np.frombuffer(bytes(msg.data), np.uint8)
                             .reshape(msg.height, msg.width, -1)[:, :, :3],
                             cv2.COLOR_BGR2GRAY)
            found = detect(det, g)
            if not found:
                return
            mid, pts = found[0]
            info = markers.get(mid) or markers.get(str(mid))
            if not info:
                return
            p4 = solve_planar(pts, obj, R_base)
            Rm = R_base @ rpy_to_R(0, 0, p4[3])
            c = p4[:3]
            if prev_c is not None and np.linalg.norm(c - prev_c) > args.max_jump_cm / 100.0:
                return
            prev_c = c
            gt_yaw = yaw_up(Rm) + yaw_off
            want = [x.strip() for x in str(info.get('label', '')).split(',') if x.strip()]
            cands = [o for o in objs if o[1] in want] or objs
            if not cands:
                return
            tid, lab, Rp, cp = min(cands, key=lambda o: np.linalg.norm(o[3] - c))
            pred_yaw = yaw_up(Rp)
            dyaw = wrap90(pred_yaw - gt_yaw)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t0 is None:
                t0 = t
            f.write(f'{t - t0:.3f},{lab},{tid},{gt_yaw:.2f},{pred_yaw:.2f},{dyaw:.2f}\n')
            f.flush()
            ys.append(dyaw)

    rclpy.init()
    node = N()
    end = time.time() + args.minutes * 60
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f'采集 {len(ys)} 帧，CSV -> {args.csv}')
    finally:
        f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not ys:
        print('没有样本：检查 marker 是否在画面内、物体是否被检测到')
        return 1
    y = np.array(ys)
    print(f'Δyaw: mean={y.mean():+.2f}°  std={y.std():.2f}°  '
          f'p50|·|={np.percentile(np.abs(y),50):.2f}°  '
          f'p95|·|={np.percentile(np.abs(y),95):.2f}°  max|·|={np.abs(y).max():.1f}°')
    if len(y) > 60:
        k = np.ones(30) / 30
        m = np.convolve(y, k, 'valid')
        print(f'30 帧滑动均值: 范围 [{m.min():+.2f}, {m.max():+.2f}]°  '
              f'峰峰 {m.max()-m.min():.2f}°；滑动 std 中位 {np.median(np.abs(y-m.mean())):.2f}°')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
