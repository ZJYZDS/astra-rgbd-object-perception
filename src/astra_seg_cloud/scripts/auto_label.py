#!/usr/bin/env python3
"""半自动标注：用 ONNX 分割模型给图片打 YOLO-seg 标签（polygon），人工再挑错。

用法：
    python3 scripts/auto_label.py --images ~/grasp_ds/raw --out ~/grasp_ds/labels \
        --model ~/astra_models/yolo26s-seg.onnx --conf 0.25

输出：
    labels/img_0000.txt   # 每行: class_id x1 y1 x2 y2 ... （归一化 0-1，mask 轮廓）
    preview/img_0000.png  # 叠了 mask 的预览图，方便人工检查
"""
import argparse
import glob
import os

import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astra_seg_cloud.yolo_seg import YoloSegOnnx, COCO_CLASSES, resolve_class_ids  # noqa: E402


def mask_to_polygon(mask, eps_ratio=0.002):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 60:
        return None
    eps = eps_ratio * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return poly if len(poly) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=os.path.expanduser('~/astra_models/yolo26s-seg.onnx'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--classes', default='')
    ap.add_argument('--names', default='',
                    help='目标类别顺序，逗号分隔；标签 id 用它在列表里的下标（解决模型 id 与训练 id 不一致）')
    ap.add_argument('--keep-unknown', action='store_true',
                    help='不在 --names 里的类别也保留（用模型原 id）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prev = os.path.join(args.out, 'preview')
    os.makedirs(prev, exist_ok=True)

    allowed = None
    if args.classes:
        allowed = resolve_class_ids([c for c in args.classes.split(',') if c])
    model = YoloSegOnnx(args.model, imgsz=640, conf_thres=args.conf,
                        iou_thres=0.45, allowed_class_ids=allowed)
    target_names = [n.strip() for n in args.names.split(',') if n.strip()]
    name2id = {n: i for i, n in enumerate(target_names)}

    paths = sorted(glob.glob(os.path.join(args.images, '*.png')) +
                   glob.glob(os.path.join(args.images, '*.jpg')))
    total_obj = 0
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        lines = []
        vis = img.copy()
        for d in model.infer(img):
            poly = mask_to_polygon(d.mask)
            if poly is None:
                continue
            pts = poly.astype(np.float64).copy()
            pts[:, 0] /= w
            pts[:, 1] /= h
            if target_names:
                cid = name2id.get(d.label)
                if cid is None:
                    if not args.keep_unknown:
                        continue
                    cid = int(d.class_id)
            else:
                cid = int(d.class_id)
            coords = ' '.join(f'{x:.5f} {y:.5f}' for x, y in pts)
            lines.append(f'{cid} {coords}')
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
            x1, y1, x2, y2 = [int(v) for v in d.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1)
            cv2.putText(vis, f'{d.label} {d.score:.2f}', (x1, max(y1 - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(args.out, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
        cv2.imwrite(os.path.join(prev, name + '.png'), vis)
        total_obj += len(lines)

    print(f'处理 {len(paths)} 张，产生 {total_obj} 个实例标签')
    print(f'标签 -> {args.out}\n预览 -> {prev}')


if __name__ == '__main__':
    main()
