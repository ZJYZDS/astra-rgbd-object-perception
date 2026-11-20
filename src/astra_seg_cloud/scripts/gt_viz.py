#!/usr/bin/env python3
"""GT 可视化模式：在 RGB 图上同时画 "预测轴" 与 "ArUco GT 轴"。

输出：
    /astra_seg/gt_compare   (sensor_msgs/Image, rgb8)  ← RViz 里加一个 Image 显示即可
可选：
    --save-dir DIR  每 --save-every 秒存一张 png
    --video FILE    录一段 mp4（便于发演示）

配色约定：
    预测 (PRED)：实线 + 亮色   x=红 y=绿 z=蓝
    GT        ：虚线 + 同色偏暗，原点画空心圈
    两者用一条细白线连起来，旁边标 Δp(mm) / Δyaw(°)
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_gt_aruco import (make_detector, detect, solve_planar, rpy_to_R,  # noqa: E402
                           yaw_up, T_from)

DEFAULT_K = (285.1711, 285.1711, 159.5, 119.5)
UP = np.array([0.0, -1.0, 0.0])


def project(K, p):
    fx, fy, cx, cy = K
    if p[2] <= 1e-3:
        return None
    return (int(round(fx * p[0] / p[2] + cx)), int(round(fy * p[1] / p[2] + cy)))


def draw_axis(img, K, T, length, colors, dashed=False, thick=1, tip=False, tag=''):
    o = T[:3, 3]
    o2 = project(K, o)
    if o2 is None:
        return
    for i, c in enumerate(colors):
        p = o + T[:3, i] * length
        p2 = project(K, p)
        if p2 is None:
            continue
        if dashed:
            _dashed(img, o2, p2, c, thick)
        else:
            cv2.line(img, o2, p2, c, thick, cv2.LINE_AA)
        if tip:
            cv2.putText(img, 'XYZ'[i], (p2[0] + 1, p2[1] - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, c, 1, cv2.LINE_AA)
    cv2.circle(img, o2, 1, (255, 255, 255), -1, cv2.LINE_AA)
    if tag:
        cv2.putText(img, tag, (o2[0] + 3, o2[1] + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, colors[0], 1, cv2.LINE_AA)


BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))


def box_corners(T, dims):
    l, w, h = float(dims[0]), float(dims[1]), float(dims[2])
    sgn = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                    [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]], float)
    local = sgn * np.array([l, w, h]) / 2.0
    return (T[:3, :3] @ local.T).T + T[:3, 3]


def draw_mask_poly(img, K, T, dims, color, alpha=0.22):
    pts = [project(K, p) for p in box_corners(T, dims)]
    pts = [p for p in pts if p is not None]
    if len(pts) < 3:
        return
    hull = cv2.convexHull(np.array(pts, np.int32))
    ov = img.copy()
    cv2.fillConvexPoly(ov, hull, color, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)
    cv2.polylines(img, [hull], True, color, 1, cv2.LINE_AA)


def draw_box(img, K, T, dims, color, thick=1, dashed=False):
    pts = [project(K, p) for p in box_corners(T, dims)]
    for a, b in BOX_EDGES:
        if pts[a] is None or pts[b] is None:
            continue
        if dashed:
            _dashed(img, pts[a], pts[b], color, thick, n=4)
        else:
            cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)


def _dashed(img, a, b, color, thick, n=6):
    a = np.array(a, float); b = np.array(b, float)
    for i in range(n):
        if i % 2:
            continue
        p = a + (b - a) * (i / n)
        q = a + (b - a) * ((i + 1) / n)
        cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/gt_markers.yaml')
    ap.add_argument('--save-dir', default=os.path.expanduser('~/gt_aruco/viz'))
    ap.add_argument('--save-every', type=float, default=1.0)
    ap.add_argument('--no-pub', action='store_true')
    ap.add_argument('--axis-len', type=float, default=0.06)
    ap.add_argument('--boxes', action='store_true', help='同时画 3D 框的 12 条棱')
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    dict_name = cfg.get('dict', 'DICT_4X4_50')
    marker_size = float(cfg.get('marker_size_m', 0.096))
    markers = cfg.get('markers', {}) or {}
    normal = np.asarray(cfg.get('normal', [0.0, -1.0, 0.0]), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    u_ax = np.cross(normal, [1.0, 0.0, 0.0])
    u_ax /= np.linalg.norm(u_ax)
    R_base = np.stack([u_ax, np.cross(normal, u_ax), normal], axis=1)
    obj_pts = np.array([[-marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, -marker_size / 2, 0],
                        [-marker_size / 2, -marker_size / 2, 0]])
    K = DEFAULT_K
    det = make_detector(dict_name)
    os.makedirs(args.save_dir, exist_ok=True)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from astra_seg_interfaces.msg import ObjectClouds

    PRED_COL = ((60, 60, 255), (60, 220, 60), (255, 160, 40))     # BGR: x红 y绿 z蓝
    GT_COL = ((120, 120, 255), (120, 255, 120), (255, 200, 120))
    state = {'img': None, 'stamp': None, 'objs': [], 'err': None, 'n': 0, 'last_save': 0.0}

    class Viz(Node):
        def __init__(self):
            super().__init__('gt_viz')
            self.pub = None if args.no_pub else self.create_publisher(
                Image, '/astra_seg/gt_compare', 5)
            self.bridge = None
            self.create_subscription(Image, '/camera/color/image_raw', self.on_img,
                                     qos_profile_sensor_data)
            self.create_subscription(ObjectClouds, '/astra_seg/objects', self.on_obj, 10)

        def on_obj(self, msg):
            rows = []
            for i, lab in enumerate(msg.labels):
                p = msg.obb_poses[i].position
                q = msg.obb_poses[i].orientation
                x, y, z, w = q.x, q.y, q.z, q.w
                nn = x * x + y * y + z * z + w * w
                s = 2.0 / max(nn, 1e-12)
                R = np.array([
                    [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                    [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                    [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [p.x, p.y, p.z]
                rows.append((lab, float(msg.scores[i]), T, msg.obb_dims[i]))
            state['objs'] = rows

        def on_img(self, msg):
            img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
            img = np.ascontiguousarray(img[:, :, :3].copy())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            state['n'] += 1
            n_mk = 0
            for mid, pts in detect(det, gray):
                info = markers.get(mid) or markers.get(str(mid))
                if not info:
                    continue
                p4 = solve_planar(pts, obj_pts, R_base)
                T_m = np.eye(4); T_m[:3, :3] = R_base @ rpy_to_R(0, 0, p4[3]); T_m[:3, 3] = p4[:3]
                T_mo = T_from(info.get('t_marker_obj', [0, 0, 0]),
                              info.get('rpy_marker_obj', [0, 0, 0]))
                T_gt = T_m @ T_mo
                n_mk += 1
                # 选最近的同标签实例
                want = [x.strip() for x in str(info.get('label', '')).split(',') if x.strip()]
                cands = [r for r in state['objs'] if r[0] in want] or state['objs']
                if not cands:
                    continue
                lab, score, T_pd, d = min(cands, key=lambda r: np.linalg.norm(r[2][:3, 3] - T_gt[:3, 3]))
                if np.linalg.norm(T_pd[:3, 3] - T_gt[:3, 3]) > 0.25:
                    continue                      # 对不上号就不画，避免误导
                dp = (T_gt[:3, 3] - T_pd[:3, 3]) * 1000.0
                dyaw = (yaw_up(T_pd[:3, :3]) - yaw_up(T_gt[:3, :3]) + 180.0) % 360.0 - 180.0
                state['err'] = (dp, dyaw)
                # 画：GT 虚线 + 预测实线
                # 预测：浅色 3D 面片 + 细轴（RGB 上不画线框——“框对不对”在点云里判断）
                draw_mask_poly(img, K, T_pd, [d.x, d.y, d.z], PRED_COL[1], alpha=0.22)
                draw_axis(img, K, T_pd, args.axis_len, PRED_COL, thick=1, tip=False, tag='')
                # GT：细轴（虚线，方便和预测轴区分）
                draw_axis(img, K, T_gt, args.axis_len, GT_COL, dashed=True, thick=1, tip=False)
                # ArUco 本体：一条细轴，标出码的位姿
                draw_axis(img, K, T_m, args.axis_len * 0.8,
                          ((180, 180, 255), (180, 255, 180), (255, 220, 160)),
                          thick=1, tip=False, tag='ArUco')
                o_gt = project(K, T_gt[:3, 3]); o_pd = project(K, T_pd[:3, 3])
                if o_gt and o_pd:
                    cv2.line(img, o_gt, o_pd, (255, 255, 255), 1, cv2.LINE_AA)
                txt = f'GT  dp={np.linalg.norm(dp):.1f}mm dyaw={dyaw:+.1f}deg  {lab}'
                y0 = 12 + 12 * (n_mk - 1)
                cv2.putText(img, txt, (6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, txt, (6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (30, 30, 30), 1, cv2.LINE_AA)
            # 图例
            legend = 'PRED solid | GT dashed'
            cv2.putText(img, legend, (6, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, legend, (6, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (30, 30, 30), 1, cv2.LINE_AA)

            if self.pub is not None:
                if self.bridge is None:
                    from cv_bridge import CvBridge
                    self.bridge = CvBridge()
                m = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                m.header = msg.header
                self.pub.publish(m)

            now = time.time()
            if now - state['last_save'] >= args.save_every:
                state['last_save'] = now
                if not hasattr(self, '_save_i'):
                    self._save_i = 0
                self._save_i += 1
                cv2.imwrite(os.path.join(args.save_dir, f'gtviz_{self._save_i:05d}.png'), img)
            if state['n'] % 60 == 0 and state['err']:
                print(f'[gtviz] frames={state["n"]} dp={np.linalg.norm(state["err"][0]):.1f}mm '
                      f'dyaw={state["err"][1]:+.1f}°', flush=True)

    rclpy.init()
    node = Viz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as exc:
        if exc.__class__.__name__ not in ('KeyboardInterrupt', 'ExternalShutdownException'):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
