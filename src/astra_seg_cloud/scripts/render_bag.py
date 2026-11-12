"""把本节点产出的 rosbag 渲染成可视化结果。

    python3 scripts/render_bag.py ~/bags/special_scene --out ~/bags

产出：
    <out>/preview.jpg      6 帧分割叠加（按 id 着色 + 框 + 标签）
    <out>/cloud3d.jpg      4 个视角的实例点云
    <out>/overlay.mp4      整段叠加视频
"""

import argparse
import os

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image

from astra_seg_interfaces.msg import ObjectClouds

FX = FY = 285.1711
CX, CY = 159.5, 119.5

PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
]


def load(uri, with_color=True):
    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    frames = {}
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic.endswith('/objects'):
            frames.setdefault(ts, {})['obj'] = deserialize_message(data, ObjectClouds)
        elif with_color and topic.endswith('/color') and with_color:
            img = deserialize_message(data, Image)
            buf = np.frombuffer(bytes(img.data), np.uint8)
            if img.encoding == 'rgb8':
                bgr = buf.reshape(img.height, img.width, 3)[:, :, ::-1].copy()
            else:
                bgr = buf.reshape(img.height, img.width, 3).copy()
            frames.setdefault(ts, {})['bgr'] = bgr
    return [(k, v.get('bgr'), v['obj']) for k, v in sorted(frames.items()) if 'obj' in v]


def xyz_of(cloud):
    return np.frombuffer(bytes(cloud.data), np.float32).reshape(-1, 4)[:, :3]


def project(xyz):
    z = xyz[:, 2]
    ok = z > 0.05
    u = np.full(len(z), -1)
    v = np.full(len(z), -1)
    u[ok] = np.round(xyz[ok, 0] * FX / z[ok] + CX).astype(int)
    v[ok] = np.round(xyz[ok, 1] * FY / z[ok] + CY).astype(int)
    return u, v, ok


def render_overlay(bgr, msg):
    h, w = bgr.shape[:2]
    layer = bgr.copy()
    for tid, label, score, cloud in zip(msg.ids, msg.labels, msg.scores, msg.clouds):
        pts = xyz_of(cloud)
        if len(pts) == 0:
            continue
        u, v, ok = project(pts)
        keep = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        color = PALETTE[int(tid) % len(PALETTE)]
        layer[v[keep], u[keep]] = color

        if keep.any():
            x1, x2 = u[keep].min(), u[keep].max()
            y1, y2 = v[keep].min(), v[keep].max()
            cv2.rectangle(layer, (x1, y1), (x2, y2), color, 1)
            txt = f'{int(tid)} {label} {score:.2f}'
            ty = min(y2 - 3, h - 3)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            cv2.rectangle(layer, (x1, ty - th - 3), (x1 + tw + 3, ty + 2), (0, 0, 0), -1)
            cv2.putText(layer, txt, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.75, bgr, 0.25, 0)


def rotate(points, yaw_deg, pitch_deg):
    """相机光学系 (x右, y下, z前) -> 观察系，绕 Y 偏航、绕 X 俯仰。"""
    x, y, z = points[:, 0], -points[:, 1], points[:, 2]   # y 轴翻上
    ya, pa = np.radians(yaw_deg), np.radians(pitch_deg)
    x1 = x * np.cos(ya) + z * np.sin(ya)
    z1 = -x * np.sin(ya) + z * np.cos(ya)
    y2 = y * np.cos(pa) - z1 * np.sin(pa)
    z2 = y * np.sin(pa) + z1 * np.cos(pa)
    return np.stack([x1, y2, z2], axis=1)


def render_cloud(points, colors, size=(560, 420), yaw=0, pitch=-18):
    """正交投影 + 自动取景：近距离特写也不会被透视甩出画面。"""
    W, H = size
    if len(points) == 0:
        return np.full((H, W, 3), 18, np.uint8)

    # 平移到质心，按最大跨度归一化
    center = np.median(points, axis=0)
    p0 = points - center
    # 用 98 分位而不是最大值，避免个别飞点把整个画面压扁
    extent = float(np.percentile(np.abs(p0), 98))
    if extent <= 0:
        return np.full((H, W, 3), 18, np.uint8)
    p0 = p0 / extent
    keep = np.all(np.abs(p0) < 1.6, axis=1)
    p0, colors = p0[keep], colors[keep]
    if len(p0) == 0:
        return np.full((H, W, 3), 18, np.uint8)

    p = rotate(p0, yaw, pitch)
    s = 0.42 * min(W, H)
    u = np.round(W / 2 + s * p[:, 0]).astype(int)
    v = np.round(H / 2 - s * p[:, 1]).astype(int)
    z = p[:, 2]

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return np.full((H, W, 3), 18, np.uint8)
    u, v, z, col = u[inside], v[inside], z[inside], colors[inside]

    order = np.argsort(-z)          # 远的先画
    u, v, z, col = u[order], v[order], z[order], col[order]

    img = np.full((H, W, 3), 18, np.uint8)
    for du in range(-1, 2):
        for dv in range(-1, 2):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = col[m]

    near = z < np.percentile(z, 25)
    img[v[near], u[near]] = col[near]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default=os.path.expanduser('~/bags'))
    ap.add_argument('--video', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = load(args.bag)
    print(f'帧数: {len(frames)}')

    # ---- 选 6 帧：目标最多的那些，按时间排序 ----
    # 均匀取样 6 帧（要求至少有 2 个目标），覆盖整段
    cand = [i for i, f in enumerate(frames) if len(f[2].clouds) >= 2]
    cand = cand or list(range(len(frames)))
    picks = [cand[int(k * (len(cand) - 1) / 5)] for k in range(6)]
    tiles = []
    for i in picks:
        _, bgr, msg = frames[i]
        ov = render_overlay(bgr, msg)
        ov = cv2.resize(ov, (480, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(ov, f'frame {i}', (6, 352), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(ov)
    while len(tiles) < 6:
        tiles.append(np.zeros((360, 480, 3), np.uint8))
    rows = [np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]
    cv2.imwrite(os.path.join(args.out, 'preview.jpg'), np.vstack(rows))

    # ---- 3D 点云：取点数最多的一帧 ----
    best = max(range(len(frames)), key=lambda i: sum(c.width for c in frames[i][2].clouds))
    _, _, msg = frames[best]
    pts, cols = [], []
    for tid, cloud in zip(msg.ids, msg.clouds):
        p = xyz_of(cloud)
        if len(p) == 0:
            continue
        pts.append(p)
        cols.append(np.tile(PALETTE[int(tid) % len(PALETTE)], (len(p), 1)))
    pts = np.concatenate(pts)
    cols = np.concatenate(cols).astype(np.uint8)
    print(f'3D 帧 {best}: {len(pts)} 点, 物体 {len(msg.clouds)}')

    views = [render_cloud(pts, cols, yaw=a) for a in (0, 35, 70, 105)]
    for k, v in enumerate(views):
        cv2.putText(v, f'yaw {0 if k==0 else [0,35,70,105][k]} deg', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cloud_img = np.vstack([np.hstack(views[0:2]), np.hstack(views[2:4])])
    cv2.imwrite(os.path.join(args.out, 'cloud3d.jpg'), cloud_img)

    # ---- 视频 ----
    if args.video:
        vw = None
        path = os.path.join(args.out, 'overlay.mp4')
        for i, (_, bgr, m) in enumerate(frames):
            ov = render_overlay(bgr, m)
            if vw is None:
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 15.0,
                                     (ov.shape[1], ov.shape[0]))
            vw.write(ov)
        if vw:
            vw.release()
        print('视频:', path)

    print('预览:', os.path.join(args.out, 'preview.jpg'))
    print('点云:', os.path.join(args.out, 'cloud3d.jpg'))


if __name__ == '__main__':
    main()
