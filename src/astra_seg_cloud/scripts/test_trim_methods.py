"""合成场景单测：比较"固定平面裁剪" vs "邻域区域生长" 对物体点云的保留率/污染率。

不依赖相机与 bag：直接构造一个已知尺寸的盒子（含侧面）+ 相邻污染面 + 深度噪声，
跑两种裁剪，再拟合 OBB，和真值比较。

用法：python3 scripts/test_trim_methods.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astra_seg_cloud import obb as O          # noqa: E402
from astra_seg_cloud.scene import grow_surface  # noqa: E402

rng = np.random.default_rng(0)


def make_scene():
    """返回 (pts, uv, truth_dims, mask_true, mask_contam)。"""
    # 桌面法线（和实机类似：朝上、略倾斜）
    n = np.array([-0.03, -0.62, 0.79], float)
    n /= np.linalg.norm(n)
    p0 = np.array([0.0, 0.10, 0.90], float)
    u = np.cross(n, [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    dims = np.array([0.31, 0.12, 0.03])          # 长 宽 高（米）
    c = p0 + 0.015 * n                            # 盒子中心（底面贴桌面）

    # ---- 顶面（密集）
    gu = np.linspace(-dims[0] / 2, dims[0] / 2, 62)
    gv = np.linspace(-dims[1] / 2, dims[1] / 2, 24)
    U, V = np.meshgrid(gu, gv)
    top = (c + 0.5 * dims[2] * n
           + U.ravel()[:, None] * u + V.ravel()[:, None] * v)
    # ---- 侧面（稀疏，4 面）
    sides = []
    hs = np.linspace(-dims[2] / 2, dims[2] / 2, 6)
    for t in np.linspace(-dims[0] / 2, dims[0] / 2, 20):
        for h in hs:
            sides.append(c + t * u + 0.5 * dims[1] * v + h * n)
            sides.append(c + t * u - 0.5 * dims[1] * v + h * n)
    for t in np.linspace(-dims[1] / 2, dims[1] / 2, 8):
        for h in hs:
            sides.append(c + 0.5 * dims[0] * u + t * v + h * n)
            sides.append(c - 0.5 * dims[0] * u + t * v + h * n)
    sides = np.array(sides)
    obj = np.vstack([top, sides])

    # ---- 邻接污染面（模拟旁边的书：低 3cm、紧贴盒子侧边）
    cu = np.linspace(dims[0] / 2 + 0.005, dims[0] / 2 + 0.20, 36)
    cv = np.linspace(-0.10, 0.10, 20)
    CU, CV = np.meshgrid(cu, cv)
    contam = (p0 + 0.002 * n
              + CU.ravel()[:, None] * u + CV.ravel()[:, None] * v)

    pts = np.vstack([obj, contam]).astype(np.float32)
    # 深度噪声：沿视线方向 ~2 mm
    pts += rng.normal(0, 0.002, pts.shape).astype(np.float32)

    # ---- 像素坐标（把点投影回一个虚拟 320x240 相机）
    f, cx, cy = 285.0, 159.5, 119.5
    z = pts[:, 2]
    uu = np.clip((f * pts[:, 0] / z + cx).round().astype(int), 0, 319)
    vv = np.clip((f * pts[:, 1] / z + cy).round().astype(int), 0, 239)
    uv = np.stack([uu, vv], axis=1)

    mask_true = np.zeros(len(pts), bool); mask_true[:len(obj)] = True
    mask_contam = ~mask_true
    return pts, uv, dims, mask_true, mask_contam, n, p0


def evaluate(tag, pts_kept, mask_kept, mask_true, mask_contam, dims, n):
    keep_true = float((mask_kept & mask_true).sum()) / max(int(mask_true.sum()), 1)
    keep_cont = float((mask_kept & mask_contam).sum()) / max(int(mask_contam.sum()), 1)
    box = None
    try:
        if int(mask_kept.sum()) >= 80:
            box = O.fit_obb(pts_kept[mask_kept], 'test', margin=0.002)
    except Exception as exc:
        print(f'  {tag}: fit 失败 {exc!r}')
        return
    if box is None:
        print(f'  {tag}: keep_true={keep_true*100:5.1f}% keep_contam={keep_cont*100:5.1f}% '
              f'-> fit 失败（点数 {int(mask_kept.sum())}）')
        return
    d = np.sort(np.asarray(box.dims))[::-1]          # 长宽高
    truth = np.sort(dims)[::-1]
    err = np.abs(d - truth) * 100
    ang = np.degrees(np.arccos(min(abs(float(box.normal @ n)), 1.0)))
    print(f'  {tag}: keep_true={keep_true*100:5.1f}% keep_contam={keep_cont*100:5.1f}% | '
          f'尺寸={np.round(d*100,1)} 误差={np.round(err,1)} cm | 法线偏差={ang:.1f}°')


def main():
    pts, uv, dims, mask_true, mask_contam, n, p0 = make_scene()
    print(f'合成场景：盒子 {dims*100} cm，点数 {len(pts)}（物体 {int(mask_true.sum())} / 污染 {int(mask_contam.sum())}）')

    # A. 固定平面裁剪（现状）
    keep_a, idx_a = O.clean_by_plane(pts, thresh=0.004)
    # clean_by_plane 返回 (points, keep)
    keepmask_a = np.zeros(len(pts), bool)
    # 用返回的 points 反推 keep 不方便，这里直接按距离复算一次，保证可比
    pl = O.ransac_plane_largest(O.voxel_downsample(pts, 0.012), thresh=0.0032, iters=60)
    if pl is not None:
        nn, dd, _ = pl
        keepmask_a = np.abs(pts @ np.asarray(nn) + float(dd)) < 0.004
    evaluate('A 固定平面裁剪 4mm', pts, keepmask_a, mask_true, mask_contam, dims, n)

    # B. 邻域区域生长
    # 种子取"物体最高处"（俯视时就是可见的上表面）
    h_all = pts @ n
    seed_h = float(np.percentile(h_all[mask_true], 99))
    keepmask_b = grow_surface(pts, uv, seed_h, n, eps_delta=0.006, eps_total=0.030)
    evaluate('B 邻域区域生长', pts, keepmask_b, mask_true, mask_contam, dims, n)


if __name__ == '__main__':
    main()
