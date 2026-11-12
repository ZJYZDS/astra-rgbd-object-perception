"""把每个实例的 3D OBB + 自身三根正交轴（x/y/z）画到 RGB 上。

轴定义（与该实例 obb_pose 一致）：
    x = OBB 长边（l 方向）  → 红
    y = OBB 短边（w 方向）  → 绿
    z = OBB 法线（h 方向，朝相机一侧）→ 蓝
所有量都在相机光学坐标系（x 右、y 下、z 前，单位米）。

用法：
    PYTHONNOUSERSITE=1 python3 scripts/obb_axes_view.py ~/bags/rec16 --out /tmp/axes
"""
import argparse
import json
import os

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image

from astra_seg_interfaces.msg import ObjectClouds

FX = FY = 285.1711
CX, CY = 159.5, 119.5

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
COLORS = [(0, 255, 255), (0, 165, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255)]

# BGR
AXIS_COLORS = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}


def quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def pose_rt(pose):
    R = quat_to_mat(pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w)
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    return R, t


def obb_corners(pose, dims):
    R, c = pose_rt(pose)
    l, w, h = dims.x / 2, dims.y / 2, dims.z / 2
    out = []
    for sz in (-1, 1):
        for sy in (-1, 1):
            for sx in (-1, 1):
                out.append(c + R @ np.array([sx * l, sy * w, sz * h]))
    return np.asarray(out)


def project(pts):
    pts = np.asarray(pts, dtype=np.float64)
    z = pts[:, 2]
    ok = z > 1e-3
    u = np.full(len(z), -1.0)
    v = np.full(len(z), -1.0)
    u[ok] = FX * pts[ok, 0] / z[ok] + CX
    v[ok] = FY * pts[ok, 1] / z[ok] + CY
    return np.stack([u, v], 1), ok


def load(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    colors, objs = {}, {}
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic.endswith('/color'):
            m = deserialize_message(data, Image)
            buf = np.frombuffer(bytes(m.data), np.uint8)
            arr = buf.reshape(m.height, m.width, -1)
            colors[ts] = arr[:, :, ::-1].copy() if m.encoding == 'rgb8' else arr.copy()
        elif topic.endswith('/objects'):
            objs[ts] = deserialize_message(data, ObjectClouds)
    keys = sorted(set(colors) & set(objs))
    return [(k, colors[k], objs[k]) for k in keys]


def draw(img, m, axis_len, scale_note=True):
    """在 img（原图）上画所有实例：OBB 棱 + 三轴。"""
    for i, lab in enumerate(m.labels):
        R, c = pose_rt(m.obb_poses[i])
        dims = m.obb_dims[i]
        col = COLORS[i % len(COLORS)]

        # --- OBB 12 条棱
        corners = obb_corners(m.obb_poses[i], dims)
        uv, ok = project(corners)
        for a, b in EDGES:
            if ok[a] and ok[b]:
                cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])),
                         col, 1, cv2.LINE_AA)

        # --- 三根正交轴（从 OBB 中心出发）
        for k, axis in enumerate('xyz'):
            vec = R[:, k] * axis_len
            tip = c + vec
            uv0, ok0 = project(np.array([c]))
            uv1, ok1 = project(np.array([tip]))
            if not (ok0[0] and ok1[0]):
                continue
            p0 = tuple(np.int32(uv0[0]))
            p1 = tuple(np.int32(uv1[0]))
            cv2.arrowedLine(img, p0, p1, AXIS_COLORS[axis], 1, cv2.LINE_AA,
                            tipLength=0.25)
            cv2.putText(img, axis, (p1[0] + 2, p1[1] + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, AXIS_COLORS[axis], 1, cv2.LINE_AA)

        # --- 标签
        uv_min = uv[ok].min(axis=0)
        txt = f'{lab} {dims.x*100:.1f}x{dims.y*100:.1f}x{dims.z*100:.1f}cm'
        cv2.putText(img, txt, (int(np.clip(uv_min[0], 1, 170)),
                               int(np.clip(uv_min[1] - 3, 9, 235))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
    return img


def legend(img_bgr):
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h + 24, w, 3), np.uint8)
    canvas[:h] = img_bgr
    cv2.putText(canvas, 'axis frame = OBB pose (camera optical):', (4, h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    x = 210
    for axis in 'xyz':
        cv2.arrowedLine(canvas, (x, h + 12), (x + 14, h + 12), AXIS_COLORS[axis], 1,
                        cv2.LINE_AA, tipLength=0.35)
        cv2.putText(canvas, axis, (x + 16, h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    AXIS_COLORS[axis], 1, cv2.LINE_AA)
        x += 34
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default='/tmp/axes')
    ap.add_argument('--axis-len', type=float, default=0.04, help='轴长（米）')
    ap.add_argument('--frames', type=int, nargs='*', default=None)
    ap.add_argument('--montage', type=int, default=4, help='拼图帧数')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = load(args.bag)
    print(f'帧数 {len(frames)}')

    # 默认挑"物体最多、且含 book"的帧
    scored = sorted(range(len(frames)),
                    key=lambda i: (-len(frames[i][2].labels),
                                   -('book' in frames[i][2].labels), i))
    picks = args.frames if args.frames else [scored[0]] + sorted(scored[1:])[:0]
    if args.frames is None:
        # 均匀再取几张做拼图
        step = max(1, len(frames) // args.montage)
        picks = sorted(set([scored[0]] + list(range(0, len(frames), step))[:args.montage]))

    report = []
    tiles = []
    for idx in picks:
        ts, bgr, m = frames[idx]
        img = draw(bgr.copy(), m, args.axis_len)
        big = cv2.resize(legend(img), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        path = f'{args.out}/axes_frame{idx:03d}.png'
        cv2.imwrite(path, big)
        tiles.append((idx, big))
        rep = {'frame': idx, 'stamp': int(ts), 'objects': []}
        for i, lab in enumerate(m.labels):
            R, c = pose_rt(m.obb_poses[i])
            rep['objects'].append({
                'id': int(m.ids[i]), 'label': lab, 'center_m': [round(float(v), 4) for v in c],
                'dims_cm': [round(float(m.obb_dims[i].x) * 100, 1),
                            round(float(m.obb_dims[i].y) * 100, 1),
                            round(float(m.obb_dims[i].z) * 100, 1)],
                # 三轴在相机系下的方向（单位向量）
                'axis_x': [round(float(v), 3) for v in R[:, 0]],
                'axis_y': [round(float(v), 3) for v in R[:, 1]],
                'axis_z': [round(float(v), 3) for v in R[:, 2]],
                'yaw_from_z_deg': round(float(np.degrees(np.arccos(
                    np.clip(abs(R[2, 2]), -1, 1)))), 1),
            })
        report.append(rep)
        print(f'  帧 {idx}: ' + ' | '.join(
            f"{o['label']} {o['dims_cm']} z轴={o['axis_z']}" for o in rep['objects']))

    json.dump(report, open(f'{args.out}/axes_report.json', 'w'), ensure_ascii=False, indent=1)

    # 2x2 拼图
    if len(tiles) >= 2:
        th, tw = tiles[0][1].shape[:2]
        rows = []
        for r in range(0, len(tiles), 2):
            pair = tiles[r:r + 2]
            if len(pair) == 1:
                pair.append((None, np.zeros_like(tiles[0][1])))
            row = np.hstack([p[1] for p in pair])
            for p in pair:
                if p[0] is not None:
                    cv2.putText(row, f'frame {p[0]}', (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)
        cv2.imwrite(f'{args.out}/axes_montage.png', np.vstack(rows))
        print(f'拼图: {args.out}/axes_montage.png')
    print(f'输出目录: {args.out}')


if __name__ == '__main__':
    main()
