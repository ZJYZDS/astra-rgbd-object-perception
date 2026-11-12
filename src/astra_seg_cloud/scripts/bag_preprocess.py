"""离线预处理：把 bag 里的实例点云清洗成可训练的样本（.npz）。

离线端才做重活，实时节点不跑这些：
    体素下采样 → 按平面分层（防止叠物被抹平）→ 每层做 MLS 平滑
    → 重新估法线（朝向相机）→ 存 npz（含 label/类别/尺寸/位姿/帧号）

用法：
    PYTHONNOUSERSITE=1 python3 scripts/bag_preprocess.py ~/bags/rec22 \
        --out ~/grasp_ds/v1 --mls-k 16 --mls-iters 2

产物：
    <out>/instances/<bag>/<frame>_<id>_<label>.npz
    <out>/report.json      每个实例的处理前后指标与耗时
    <out>/summary.md       汇总
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

from astra_seg_interfaces.msg import ObjectClouds
from astra_seg_cloud.obb import ransac_plane, split_raised, voxel_downsample

CLASS_INDEX = {'cup': 0, 'keyboard': 1, 'book': 2, 'bottle': 3}


def mls_project(pts, k=16, iters=2):
    """最简 MLS：每点用 k 近邻拟合局部平面，把点投影上去。迭代 iters 次。"""
    P = pts.astype(np.float64).copy()
    k = min(k, len(P))
    for _ in range(iters):
        tree = cKDTree(P)
        _, idx = tree.query(P, k=k)
        out = np.empty_like(P)
        for i, nb in enumerate(idx):
            Q = P[nb]
            c = Q.mean(0)
            _, _, vh = np.linalg.svd(Q - c, full_matrices=False)
            n = vh[2]
            out[i] = P[i] - n * float((P[i] - c) @ n)
        P = out
    return P.astype(np.float32)


def estimate_normals(pts, k=12):
    tree = cKDTree(pts)
    k = min(k, len(pts))
    _, idx = tree.query(pts, k=k)
    N = np.empty_like(pts)
    for i, nb in enumerate(idx):
        Q = pts[nb]
        _, _, vh = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
        N[i] = vh[2]
    # 统一朝向相机（原点）
    flip = np.einsum('ij,ij->i', N, pts) > 0
    N[flip] *= -1
    return N.astype(np.float32)


def thickness(pts):
    p = voxel_downsample(pts, 0.008)
    r = ransac_plane(p, thresh=0.008, iters=80)
    if r is None:
        return None, None
    n, d, inl = r
    dist = p[inl] @ n + d
    return float(np.percentile(dist, 95) - np.percentile(dist, 5)), float(
        np.sqrt(np.mean(dist ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default=os.path.expanduser('~/grasp_ds/v1'))
    ap.add_argument('--voxel', type=float, default=0.006)
    ap.add_argument('--mls-k', type=int, default=16)
    ap.add_argument('--mls-iters', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--no-mls', action='store_true')
    args = ap.parse_args()

    bag_name = os.path.basename(args.bag.rstrip('/'))
    inst_dir = os.path.join(args.out, 'instances', bag_name)
    os.makedirs(inst_dir, exist_ok=True)

    r = SequentialReader()
    r.open(StorageOptions(uri=args.bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    report = []
    frames = 0
    while r.has_next():
        topic, data, ts = r.read_next()
        if not topic.endswith('/objects'):
            continue
        frames += 1
        if args.max_frames and frames > args.max_frames:
            break
        m = deserialize_message(data, ObjectClouds)
        for i, lab in enumerate(m.labels):
            P = np.frombuffer(bytes(m.clouds[i].data), np.float32).reshape(-1, 4)
            xyz, rgb = P[:, :3], P[:, 3]
            if len(xyz) < 200:
                continue
            t0 = time.perf_counter()
            t_raw, rms_raw = thickness(xyz)

            pts = voxel_downsample(xyz, args.voxel)
            base, raised = split_raised(pts)          # 按层分开，MLS 才不会跨层抹平
            layers = []
            if raised.sum() >= 80:
                layers = [(pts[base], 'base'), (pts[raised], 'top')]
            else:
                layers = [(pts, 'base')]

            out_pts, out_lbl = [], []
            for L, name in layers:
                Ls = L if args.no_mls else mls_project(L, args.mls_k, args.mls_iters)
                out_pts.append(Ls)
                out_lbl.append(np.full(len(Ls), 0 if name == 'base' else 1, np.uint8))
            P_all = np.vstack(out_pts)
            L_all = np.concatenate(out_lbl)
            Nrm = estimate_normals(P_all)
            t_all, rms_all = thickness(P_all)

            # 颜色按最近邻从原始点继承（离线，用 KD 树即可）
            tree = cKDTree(xyz)
            _, nn = tree.query(P_all)
            C_all = rgb[nn].astype(np.float32)
            ripg = C_all
            # rgb 是 float32 打包的 0x00RRGGBB，拆成 uint8
            packed = ripg.astype(np.uint32).ravel()
            col8 = np.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF,
                             packed & 0xFF], 1).astype(np.uint8)

            q = m.obb_poses[i]
            name = f'{ts:019d}_{int(m.ids[i]):03d}_{lab}.npz'
            np.savez_compressed(
                os.path.join(inst_dir, name),
                points=P_all.astype(np.float32),
                colors=col8,
                normals=Nrm,
                layers=L_all,
                label=lab,
                class_id=CLASS_INDEX.get(lab, 255),
                frame_ts=int(ts),
                track_id=int(m.ids[i]),
                obb_center=np.array([q.position.x, q.position.y, q.position.z], np.float32),
                obb_quat=np.array([q.orientation.x, q.orientation.y, q.orientation.z,
                                   q.orientation.w], np.float32),
                obb_dims=np.array([m.obb_dims[i].x, m.obb_dims[i].y, m.obb_dims[i].z],
                                  np.float32),
            )
            report.append({
                'frame': frames, 'ts': int(ts), 'id': int(m.ids[i]), 'label': lab,
                'n_raw': int(len(xyz)), 'n_out': int(len(P_all)),
                'layers': [n for _, n in layers],
                'thick_raw_cm': None if t_raw is None else round(t_raw * 100, 2),
                'thick_mls_cm': None if t_all is None else round(t_all * 100, 2),
                'rms_raw_cm': None if rms_raw is None else round(rms_raw * 100, 2),
                'rms_mls_cm': None if rms_all is None else round(rms_all * 100, 2),
                'ms': round((time.perf_counter() - t0) * 1000, 1),
            })

    json.dump(report, open(os.path.join(args.out, 'report.json'), 'w'),
              ensure_ascii=False, indent=1)

    # 汇总
    import collections
    agg = collections.defaultdict(list)
    for e in report:
        agg[e['label']].append(e)
    lines = [f'# 离线预处理汇总：{bag_name}', '', f'帧数 {frames}，实例 {len(report)}', '',
             '| 类别 | 实例 | 原始点数中位 | 输出点数中位 | 厚度 原始→MLS (cm) | RMS 原始→MLS | 单实例耗时 |',
             '|---|---|---|---|---|---|---|']
    for lab, v in agg.items():
        med = lambda k: np.median([x[k] for x in v if x[k] is not None])
        lines.append(
            f"| {lab} | {len(v)} | {int(np.median([x['n_raw'] for x in v]))} | "
            f"{int(np.median([x['n_out'] for x in v]))} | "
            f"{med('thick_raw_cm'):.2f} → {med('thick_mls_cm'):.2f} | "
            f"{med('rms_raw_cm'):.2f} → {med('rms_mls_cm'):.2f} | "
            f"{np.median([x['ms'] for x in v]):.1f} ms |")
    txt = '\n'.join(lines) + '\n'
    open(os.path.join(args.out, 'summary.md'), 'w').write(txt)
    print(txt)
    print(f'实例文件: {inst_dir}')


if __name__ == '__main__':
    main()
