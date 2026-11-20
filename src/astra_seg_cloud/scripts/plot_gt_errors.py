#!/usr/bin/env python3
"""把 pose_gt_aruco.py 记的 CSV 画成"长时序误差曲线"。

用法:
    python3 scripts/plot_gt_errors.py <csv> [--out png]
输出:
    - Δx / Δy / Δz 随时间的曲线（含 ±1σ 带与滑动均值）
    - 位置漂移（预测 - GT，按窗口均值）
    - 统计表：均值/标准差/p95/最大，以及"30 帧滑动 std"的中位数
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    import csv
    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if args.label and r['label'] != args.label:
                continue
            try:
                rows.append({k: float(v) for k, v in r.items() if k != 'label'}
                            | {'label': r['label']})
            except ValueError:
                continue
    if not rows:
        print('CSV 为空（可能没匹配到实例）')
        return 1
    t = np.array([r['t'] for r in rows])
    for ax, key in (('x', 'dx_cm'), ('y', 'dy_cm'), ('z', 'dz_cm')):
        pass
    dx = np.array([r['dx_cm'] for r in rows])
    dy = np.array([r['dy_cm'] for r in rows])
    dz = np.array([r['dz_cm'] for r in rows])
    dp = np.array([r['dp_cm'] for r in rows])
    yaw = np.array([r['dyaw_deg'] for r in rows])

    print(f'样本 {len(t)} 帧，时长 {t[-1]:.1f} s，平均 {len(t)/max(t[-1],1e-6):.1f} Hz')
    print(f'{"量":<8}{"均值":>10}{"std":>9}{"p95|·|":>10}{"max|·|":>10}')
    for name, v in (('dx(cm)', dx), ('dy(cm)', dy), ('dz(cm)', dz),
                    ('|dp|(cm)', dp), ('dyaw(°)', yaw)):
        print(f'{name:<8}{v.mean():10.3f}{v.std():9.3f}'
              f'{np.percentile(np.abs(v),95):10.3f}{np.abs(v).max():10.3f}')

    def slide_std(v, w=30):
        if len(v) < w:
            return np.array([v.std()])
        k = np.ones(w) / w
        m = np.convolve(v, k, mode='valid')
        m2 = np.convolve(v * v, k, mode='valid')
        return np.sqrt(np.maximum(m2 - m * m, 0))

    print(f'30 帧滑动 std 中位: x {np.median(slide_std(dx))*10:.2f} mm, '
          f'y {np.median(slide_std(dy))*10:.2f} mm, z {np.median(slide_std(dz))*10:.2f} mm')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, v, name in ((axes[0], dx, 'Δx'), (axes[1], dy, 'Δy'), (axes[2], dz, 'Δz')):
        ax.plot(t, v, lw=0.6, color='0.55', label='per frame')
        if len(v) >= 30:
            k = np.ones(30) / 30
            ax.plot(t[29:], np.convolve(v, k, 'valid'), lw=1.6, color='tab:blue',
                    label='30-frame mean')
        ax.axhline(v.mean(), ls='--', lw=0.8, color='tab:red',
                   label=f'mean {v.mean():+.2f} cm')
        ax.fill_between(t, v.mean() - v.std(), v.mean() + v.std(), color='tab:blue', alpha=0.12)
        ax.set_ylabel(f'{name} [cm]')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    axes[3].plot(t, dp, lw=0.6, color='0.55')
    axes[3].set_ylabel('|Δp| [cm]')
    axes[3].grid(alpha=0.3)
    axes[3].set_xlabel(f'time [s]   (samples: {len(t)}, {len(t)/max(t[-1],1e-6):.1f} Hz)')
    fig.suptitle('book pose error vs ArUco GT (long run)')
    fig.tight_layout()
    out = args.out or os.path.splitext(args.csv)[0] + '_errors.png'
    fig.savefig(out, dpi=130)
    print(f'图 -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
