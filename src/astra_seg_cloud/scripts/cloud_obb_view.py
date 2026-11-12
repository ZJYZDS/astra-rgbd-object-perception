"""点云 + 3D OBB 线框的可视化（离线，读 run5 的导出数据）。

    PYTHONNOUSERSITE=1 python3 scripts/cloud_obb_view.py --pkl /tmp/run5_dump.pkl --out ~/bags
"""

import argparse
import os
import pickle

from astra_seg_cloud.entry import _prepend_venv, _strip_user_site

_strip_user_site()
_prepend_venv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from astra_seg_cloud.obb import filter_to_obb, fit_obb  # noqa: E402

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (2, 3, 7, 6),
    (0, 3, 7, 4), (1, 2, 6, 5),
]

PALETTE = [(0, 200, 255), (0, 128, 255), (255, 128, 0), (0, 255, 0)]

FX = FY = 285.1711
CX, CY = 159.5, 119.5


def rot(p, yaw, pitch):
    x, y, z = p[:, 0], -p[:, 1], p[:, 2]
    ya, pa = np.radians(yaw), np.radians(pitch)
    x1 = x * np.cos(ya) + z * np.sin(ya)
    z1 = -x * np.sin(ya) + z * np.cos(ya)
    y2 = y * np.cos(pa) - z1 * np.sin(pa)
    z2 = y * np.sin(pa) + z1 * np.cos(pa)
    return np.stack([x1, y2, z2], axis=1)


def render(points, colors, boxes, size=(620, 470), yaw=0, pitch=-18):
    W, H = size
    p = rot(points, yaw, pitch)
    s = 0.40 * min(W, H)

    # 1) 半透明盒面打底
    img = np.full((H, W, 3), 16, np.uint8)
    overlay = img.copy()
    projected = []
    for corners, color in boxes:
        q = rot(corners, yaw, pitch)
        qu = np.round(W / 2 + s * q[:, 0]).astype(int)
        qv = np.round(H / 2 - s * q[:, 1]).astype(int)
        projected.append((qu, qv, color, np.mean(q[:, 2])))
        for f in sorted(FACES, key=lambda f: -np.mean(q[list(f), 2])):
            poly = np.stack([[qu[i], qv[i]] for i in f], axis=0)
            cv2.fillConvexPoly(overlay, poly, color)
    img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)

    # 2) 点云
    u = np.round(W / 2 + s * p[:, 0]).astype(int)
    v = np.round(H / 2 - s * p[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    order = np.argsort(-p[ok, 2])
    uu, vv, cc = u[ok][order], v[ok][order], colors[ok][order]
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            x, y = uu + du, vv + dv
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[m], x[m]] = cc[m]

    # 3) 线框
    for qu, qv, color, _z in projected:
        for a, b in EDGES:
            cv2.line(img, (qu[a], qv[a]), (qu[b], qv[b]), color, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='/tmp/run5_dump.pkl')
    ap.add_argument('--out', default=os.path.expanduser('~/bags'))
    args = ap.parse_args()

    frames = pickle.load(open(args.pkl, 'rb'))

    best, best_n = None, -1
    for idx, (ts, bgr, items) in enumerate(frames):
        n = sum(len(it[3]) for it in items)
        if n > best_n:
            best, best_n = (idx, items), n
    idx, items = best
    print(f'选帧 {idx}，点数 {best_n}，物体 {len(items)}')

    pts_all, cols_all, boxes, legend = [], [], [], []
    txt_colors = []
    bgr = frames[idx][1]
    h_img, w_img = bgr.shape[:2]

    for k, item in enumerate(items):
        tid, label, score, pts = item[0], item[1], item[2], item[3]
        obb = fit_obb(pts, label)
        if obb is None:
            continue

        # 用 3D 点投影回原图取真实颜色
        z = pts[:, 2]
        u = np.round(FX * pts[:, 0] / np.maximum(z, 1e-6) + CX).astype(int)
        v = np.round(FY * pts[:, 1] / np.maximum(z, 1e-6) + CY).astype(int)
        ok = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img) & (z > 0)
        rgb = np.full((len(pts), 3), 40, np.uint8)
        rgb[ok] = bgr[v[ok], u[ok]]

        pts, keep = filter_to_obb(pts, obb)
        pts_all.append(pts)
        cols_all.append(rgb[keep])
        boxes.append((obb.corners(), (0, 255, 0)))
        l, w, h = obb.dims * 100
        legend.append(f'{label}  {l:.1f}x{w:.1f}x{h:.1f} cm')
        txt_colors.append(PALETTE[k % len(PALETTE)])

    points = np.concatenate(pts_all)
    colors = np.concatenate(cols_all).astype(np.uint8)

    # 全局归一化，保证多个物体的相对位置正确
    center = np.median(points, axis=0)
    extent = float(np.percentile(np.abs(points - center), 98))
    points = (points - center) / extent
    boxes = [((c - center) / extent, col) for c, col in boxes]

    views = [(0, -18, 'front'), (45, -18, 'yaw 45'),
             (90, -18, 'yaw 90'), (0, -78, 'top')]
    tiles = []
    for yaw, pitch, name in views:
        img = render(points, colors, boxes, yaw=yaw, pitch=pitch)
        cv2.putText(img, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 150, 150), 1, cv2.LINE_AA)
        tiles.append(img)

    for i, text in enumerate(legend):
        cv2.putText(tiles[0], text, (8, 44 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, txt_colors[i], 1, cv2.LINE_AA)

    out = os.path.join(args.out, 'cloud_obb_3d.jpg')
    cv2.imwrite(out, np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])]))
    print('saved', out)


if __name__ == '__main__':
    main()
