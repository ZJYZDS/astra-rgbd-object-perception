"""把 bag 里的 /astra_seg/annotated 导出成 mp4。

    python3 scripts/bag_to_video.py ~/bags/recXX --out ~/bags/recXX.mp4 --scale 2
"""
import argparse
import os

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default=None)
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--fps', type=float, default=15.0)
    args = ap.parse_args()

    out = args.out or (args.bag.rstrip('/') + '.mp4')
    r = SequentialReader()
    r.open(StorageOptions(uri=args.bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    frames = []
    while r.has_next():
        topic, data, _ = r.read_next()
        if not topic.endswith('/annotated'):
            continue
        m = deserialize_message(data, Image)
        arr = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
        frames.append(arr[:, :, ::-1].copy())      # rgb/bgr -> bgr
    if not frames:
        raise SystemExit('bag 里没有 /astra_seg/annotated')

    h, w = frames[0].shape[:2]
    vw, vh = int(w * args.scale), int(h * args.scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_obj = cv2.VideoWriter(out, fourcc, args.fps, (vw, vh))
    for f in frames:
        vw_obj.write(cv2.resize(f, (vw, vh), interpolation=cv2.INTER_NEAREST))
    vw_obj.release()
    print(f'写出 {out}：{len(frames)} 帧, {vw}x{vh} @ {args.fps:.0f} fps')


if __name__ == '__main__':
    main()
