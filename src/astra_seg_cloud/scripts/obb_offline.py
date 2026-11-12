"""离线验证 OBB + 抓取位姿：读 bag，逐帧算，画标注图并统计。

    PYTHONNOUSERSITE=1 python3 scripts/obb_offline.py ~/bags/run5 --out ~/bags
"""

import argparse
import collections
import os
import sys
import time

from astra_seg_cloud.entry import _prepend_venv, _strip_user_site

_strip_user_site()
_prepend_venv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from astra_seg_interfaces.msg import ObjectClouds  # noqa: E402

from astra_seg_cloud.annotate import draw_grasp, draw_label, draw_obb  # noqa: E402
from astra_seg_cloud.grasp import GRASP_NAMES, GripperConfig, evaluate  # noqa: E402
from astra_seg_cloud.obb import fit_obb  # noqa: E402

FX = FY = 285.1711
CX, CY = 159.5, 119.5

COLORS = {1: (0, 200, 255), 2: (0, 128, 255)}  # 吸盘 / 夹爪


def load(uri):
    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    colors, objs = {}, {}
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic.endswith('/color'):
            m = deserialize_message(data, Image)
            buf = np.frombuffer(bytes(m.data), np.uint8)
            bgr = buf.reshape(m.height, m.width, 3)[:, :, ::-1].copy()
            colors[ts] = bgr
        elif topic.endswith('/objects'):
            objs[ts] = deserialize_message(data, ObjectClouds)
    frames = []
    for ts in sorted(objs):
        if ts not in colors:
            continue
        items = []
        for tid, lab, sc, c in objs[ts]:
            pts = np.frombuffer(bytes(c.data), np.float32).reshape(-1, 4)[:, :3]
            items.append((int(tid), lab, float(sc), pts))
        frames.append((ts, colors[ts], items))
    return frames


def load_pkl(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def xyz_of(cloud):
    return np.frombuffer(bytes(cloud.data), np.float32).reshape(-1, 4)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag', nargs='?', default='')
    ap.add_argument('--pkl', default='')
    ap.add_argument('--out', default=os.path.expanduser('~/bags'))
    ap.add_argument('--suction-radius', type=float, default=0.03)
    ap.add_argument('--jaw-max-open', type=float, default=0.08)
    ap.add_argument('--jaw-width', type=float, default=0.04)
    args = ap.parse_args()

    cfg = GripperConfig(suction_radius=args.suction_radius,
                        jaw_max_open=args.jaw_max_open,
                        jaw_width=args.jaw_width)

    frames = load_pkl(args.pkl) if args.pkl else load(args.bag)
    print(f'帧数 {len(frames)}')

    kinds = collections.Counter()
    modes = collections.Counter()
    dims = collections.defaultdict(list)
    scores = []
    timings = []
    rendered = []

    for ts, bgr, items in frames:
        img = bgr.copy()
        t0 = time.perf_counter()
        label_y = 16
        for tid, label, score, pts in items:
            if len(pts) < 80:
                continue
            obb = fit_obb(pts, label)
            if obb is None:
                continue
            grasp = evaluate(obb, cfg)
            kinds[grasp.name] += 1
            modes[grasp.mode] += 1
            dims[label].append(obb.dims)
            scores.append(grasp.score)
            draw_obb(img, obb, FX, FY, CX, CY, COLORS.get(grasp.kind, (200, 200, 200)))
            draw_grasp(img, grasp, FX, FY, CX, CY, COLORS.get(grasp.kind, (200, 200, 200)))
            text = (f'{int(tid)} {label} {obb.dims[0]*100:.1f}x{obb.dims[1]*100:.1f}'
                    f'x{obb.dims[2]*100:.1f}cm {grasp.mode} w={grasp.width*100:.0f}cm '
                    f's={grasp.score:.2f}')
            draw_label(img, text, (8, label_y), (255, 255, 255))
            label_y += 14
        timings.append((time.perf_counter() - t0) * 1000)
        rendered.append((len(items), ts, img))

    print(f'对象 {sum(kinds.values())} 个')
    print('抓取方式:', dict(kinds))
    print('抓取模式:', dict(modes))
    print(f'平均耗时 {np.mean(timings):.1f} ms/帧（含绘图）')

    for label, ds in dims.items():
        d = np.array(ds)
        print(f'  {label:9s} n={len(d):3d}  l/w/h 中位数 = '
              f'{np.median(d[:,0])*100:.1f} / {np.median(d[:,1])*100:.1f} / '
              f'{np.median(d[:,2])*100:.1f} cm')

    # 取目标最多的 6 帧做预览
    rendered.sort(key=lambda r: -r[0])
    picks = rendered[:6]
    tiles = []
    for n, ts, img in picks:
        t = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
        tiles.append(t)
    while len(tiles) < 6:
        tiles.append(np.zeros((360, 480, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    out_img = os.path.join(args.out, 'obb_preview.jpg')
    cv2.imwrite(out_img, grid)
    print('预览:', out_img)


if __name__ == '__main__':
    main()
