"""点云 + 3D bbox + 三轴 可视化（每实例一格，两个视角）。

左列：沿 OBB 法线看（俯视，x-y 平面）
右列：斜视（把局部 y-z 一起投影，能看到厚度）
点云用真实 RGB 上色；框线为 OBB；红/绿/蓝 = 该实例的 x/y/z 轴。
"""
import argparse
import os

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image

from astra_seg_interfaces.msg import ObjectClouds

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
AX = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}
TILE = (420, 420)


def quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def corners_local(dims):
    l, w, h = dims / 2.0
    return np.array([[sx * l, sy * w, sz * h]
                     for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])


def render(ax_u, ax_v, pts, cols, box, scale, ox, oy, tile, thick=2):
    h, w = tile
    img = np.zeros((h, w, 3), np.uint8)
    u = ox + (pts[:, ax_u] * scale).astype(int)
    v = oy - (pts[:, ax_v] * scale).astype(int)
    ok = (u >= thick) & (u < w - thick) & (v >= thick) & (v < h - thick)
    uu, vv = u[ok], v[ok]
    for du in range(thick):
        for dv in range(thick):
            img[vv + dv, uu + du] = cols[ok]
    # 框
    bu = ox + (box[:, ax_u] * scale).astype(int)
    bv = oy - (box[:, ax_v] * scale).astype(int)
    for a, b in EDGES:
        cv2.line(img, (bu[a], bv[a]), (bu[b], bv[b]), (0, 255, 255), 1, cv2.LINE_AA)
    # 三轴
    o = np.array([0., 0., 0.])
    for k, name in enumerate('xyz'):
        tip = o.copy()
        tip[k] = 0.55 * max(abs(box[:, k]).max(), 1e-3) * 2
        p0 = (int(ox + o[ax_u] * scale), int(oy - o[ax_v] * scale))
        p1 = (int(ox + tip[ax_u] * scale), int(oy - tip[ax_v] * scale))
        cv2.arrowedLine(img, p0, p1, AX[name], 1, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(img, name, (p1[0] + 2, p1[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    AX[name], 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default='/tmp/cloud_obb')
    ap.add_argument('--max-points', type=int, default=6000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    r = SequentialReader()
    r.open(StorageOptions(uri=args.bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    objs, cols = {}, {}
    while r.has_next():
        t, d, ts = r.read_next()
        if t.endswith('/objects'):
            objs[ts] = deserialize_message(d, ObjectClouds)
        elif t.endswith('/color'):
            m = deserialize_message(d, Image)
            a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
            cols[ts] = a[:, :, ::-1].copy()
    keys = sorted(objs)
    best = max(keys, key=lambda k: len(objs[k].labels))
    m = objs[best]
    print(f'frame {best}: {len(m.labels)} 实例')

    tiles = []
    for i, lab in enumerate(m.labels):
        q = m.obb_poses[i]
        R = quat_to_mat(q.orientation.x, q.orientation.y, q.orientation.z, q.orientation.w)
        c = np.array([q.position.x, q.position.y, q.position.z])
        dims = np.array([m.obb_dims[i].x, m.obb_dims[i].y, m.obb_dims[i].z])
        P = np.frombuffer(bytes(m.clouds[i].data), np.float32).reshape(-1, 4)
        xyz, rgb = P[:, :3], P[:, 3].astype(np.uint32)
        if len(xyz) > args.max_points:
            sel = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]
        col = np.stack([(rgb & 0xFF), ((rgb >> 8) & 0xFF), ((rgb >> 16) & 0xFF)], 1)
        col = np.clip(col.astype(np.float32) * 1.7 + 30, 0, 255).astype(np.uint8)

        local = (xyz - c) @ R
        box = corners_local(dims)
        extent = float(np.abs(box).max()) * 2.2
        scale = 0.62 * min(TILE) / extent

        views = [('top  (x-y)', 0, 1), ('side (x-z)', 0, 2), ('front(y-z)', 1, 2)]
        tiles_v = [render(au, av, local, col, box, scale, TILE[1] // 2, TILE[0] // 2, TILE)
                   for _, au, av in views]
        inside = float(np.mean(np.all(np.abs(box) <= dims / 2 + 0.002, axis=1))) if False else None
        row = np.hstack(tiles_v)
        ov = np.abs(local) - dims / 2.0
        frac_in = float(np.mean(np.all(ov <= 0.002, axis=1)))
        cv2.putText(row, f'{lab}  id={int(m.ids[i])}  '
                         f'{dims[0]*100:.1f}x{dims[1]*100:.1f}x{dims[2]*100:.1f} cm  '
                         f'({len(xyz)} pts, 框内 {frac_in*100:.1f}%)',
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        for vi, (vname, _, _) in enumerate(views):
            cv2.putText(row, vname, (vi * TILE[1] + 6, TILE[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        tiles.append(row)

    out = np.vstack(tiles)
    path = f'{args.out}/cloud_obb_tiles.png'
    cv2.imwrite(path, out)
    cv2.imwrite(f'{args.out}/cloud_obb_frame{best}.png', out)
    print('wrote', path, out.shape)


if __name__ == '__main__':
    main()
