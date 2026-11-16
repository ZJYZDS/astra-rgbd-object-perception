#!/usr/bin/env python3
"""把 图片 + YOLO-seg 标签 组装成训练目录（含 train/val 划分和 data.yaml）。

用法：
    python3 scripts/build_dataset.py --images ~/grasp_ds/raw \
        --labels ~/grasp_ds/labels --out ~/grasp_ds/dataset \
        --names book,bottle,keyboard,pen,remote,stapler --val-ratio 0.15
"""
import argparse
import glob
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--names', required=True)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(',') if n.strip()]
    imgs = sorted(glob.glob(os.path.join(args.images, '*.png')) +
                  glob.glob(os.path.join(args.images, '*.jpg')))
    pairs = []
    skipped = 0
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(args.labels, stem + '.txt')
        if not os.path.exists(lbl):
            skipped += 1
            continue
        if os.path.getsize(lbl) == 0:      # 空标签（无实例）→ 不去训练，避免学背景
            skipped += 1
            continue
        pairs.append((p, lbl))
    if not pairs:
        print('没有可用的 图/标签 对'); return

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio)) if len(pairs) > 6 else 0
    split = {'val': pairs[:n_val], 'train': pairs[n_val:]}

    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    for part, items in split.items():
        for img, lbl in items:
            shutil.copy2(img, os.path.join(args.out, 'images', part, os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(args.out, 'labels', part, os.path.basename(lbl)))

    with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(args.out)}\n')
        f.write('train: images/train\nval: images/val\n')
        f.write(f'nc: {len(names)}\nnames: {names}\n')

    print(f'训练 {len(split["train"])} 张，验证 {len(split["val"])} 张，跳过 {skipped} 张')
    print(f'data.yaml -> {os.path.join(args.out, "data.yaml")}')
    print(f'classes: {names}')


if __name__ == '__main__':
    main()
