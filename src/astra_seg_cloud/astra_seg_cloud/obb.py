"""6DoF / OBB 估计。

算法与 PCL 流程对应，但不依赖 PCL/Open3D，只用到 numpy + OpenCV：

    体素下采样            -> pcl::VoxelGrid
    统计离群点剔除        -> pcl::StatisticalOutlierRemoval
    kNN 局部 PCA 法线     -> pcl::NormalEstimation
    RANSAC 主平面         -> pcl::SACSegmentation
    平面内最小面积矩形    -> 凸包 + 旋转卡壳（OpenCV minAreaRect）
    尺寸 + 表面偏置       -> OBB

约定：所有坐标都在相机光学系（x 右、y 下、z 前，单位米）。
法线统一朝向相机（原点）。
"""

import cv2
import numpy as np

# 单视角看不到厚度的类别先验（米）
THICKNESS_PRIOR = {
    'book': 0.020,
    'keyboard': 0.020,
    'cup': 0.080,
    'bottle': 0.070,
    'box': 0.050,
    'laptop': 0.020,
    'default': 0.030,
}


class Obb:
    """有向包围盒：中心、朝向(3x3, 列为主轴)、尺寸(l,w,h)、平面质量。"""

    __slots__ = ('center', 'rotation', 'dims', 'normal', 'plane_ratio',
                 'rmse', 'n_points', 'source')

    def __init__(self, center, rotation, dims, normal,
                 plane_ratio=0.0, rmse=0.0, n_points=0, source='plane'):
        self.center = np.asarray(center, dtype=np.float32)
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.dims = np.asarray(dims, dtype=np.float32)
        self.normal = np.asarray(normal, dtype=np.float32)
        self.plane_ratio = float(plane_ratio)
        self.rmse = float(rmse)
        self.n_points = int(n_points)
        self.source = source

    def corners(self):
        """8 个角点，顺序：(-,-,-) (+,-,-) ... 与 12 条棱的索引表配合。"""
        l, w, h = self.dims / 2.0
        local = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h],
        ], dtype=np.float32)
        return (self.rotation @ local.T).T + self.center

    def quaternion(self):
        return quat_from_matrix(self.rotation)


def quat_from_matrix(R):
    """旋转矩阵 -> 四元数 (x, y, z, w)。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def voxel_downsample(points, voxel=0.005):
    if len(points) <= 200:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(out, inv, points)
    cnt = np.bincount(inv)
    return (out / cnt[:, None]).astype(np.float32)


def _knn(points, k):
    """返回每个点的 k 近邻索引（含自身）。优先 scipy，退化时用暴力法。"""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from scipy.spatial import cKDTree
        _, idx = cKDTree(points).query(points, k=min(k, len(points)))
        return np.atleast_2d(idx)
    except Exception:
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        return np.argsort(d, axis=1)[:, :min(k, len(points))]


def remove_outliers(points, k=10, std_mul=2.0):
    if len(points) <= k + 1:
        return points
    idx = _knn(points, k + 1)[:, 1:]
    d = np.linalg.norm(points[:, None, :] - points[idx], axis=2)
    mean_d = d.mean(axis=1)
    thr = mean_d.mean() + std_mul * mean_d.std()
    keep = mean_d <= thr
    return points[keep] if keep.sum() >= 10 else points


def estimate_normals(points, k=20):
    """逐点 PCA 法线，朝向相机（原点）。返回 (N,3)。"""
    idx = _knn(points, k)
    nb = points[idx]                                   # (N,k,3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb) / float(k)
    _, vecs = np.linalg.eigh(cov)                      # 升序
    normals = vecs[:, :, 0]
    # 朝向原点（相机）
    flip = np.einsum('ij,ij->i', normals, -points) < 0
    normals[flip] *= -1
    return normals.astype(np.float32)


def ransac_plane(points, thresh=0.005, iters=150, seed=0):
    """返回 (normal, d, inlier_mask)；失败返回 None。"""
    n_pts = len(points)
    if n_pts < 10:
        return None
    rng = np.random.default_rng(seed)
    best_cnt, best = 0, None
    for _ in range(iters):
        i = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = -float(nrm @ p0)
        dist = np.abs(points @ nrm + d)
        cnt = int((dist < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nrm, d, dist < thresh)
    if best is None or best_cnt < 10:
        return None
    nrm, d, mask = best
    # 用内点做一次最小二乘精修
    P = points[mask]
    c = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - c, full_matrices=False)
    nrm = vh[-1]
    d = -float(nrm @ c)
    dist = np.abs(points @ nrm + d)
    mask = dist < thresh
    return nrm, d, mask


def _basis_from_normal(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v


def _axis_candidates_iou(rect_pts, image_mask, n, base_axis, K, margin=0.005):
    """在 4 个 90° 候选里选"投影面与 mask 的 IoU 最大"的那个。

    动机：mask 最小矩形接近正方形时，cv2 的 w/h 长边判定是随机的，
    直接把它的角度当先验会差 90°；而 4 个候选 + IoU 判据不依赖长边判定。
    """
    if image_mask is None or base_axis is None:
        return None
    fx, fy, cx, cy = K
    h, w = image_mask.shape[:2]
    best = (-1.0, None, None, None)
    for k in range(4):
        ang = k * (np.pi / 2.0)
        c, s = np.cos(ang), np.sin(ang)
        # 绕 n 旋转 base_axis（Rodrigues）
        x = base_axis * c + np.cross(n, base_axis) * s
        x = x - n * float(np.dot(x, n))
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            continue
        x = x / nx
        y = np.cross(n, x)
        px = rect_pts @ x
        py = rect_pts @ y
        l = float(px.max() - px.min()) + 2 * margin
        ww = float(py.max() - py.min()) + 2 * margin
        if l < ww:
            x, y, l, ww = y, x, ww, l
            px, py = py, px
        cpx = 0.5 * (px.max() + px.min())
        cpy = 0.5 * (py.max() + py.min())
        fc = rect_pts.mean(axis=0) + x * (cpx - float(rect_pts.mean(axis=0) @ x)) \
            + y * (cpy - float(rect_pts.mean(axis=0) @ y))
        quad = np.array([fc - x * l / 2 - y * ww / 2, fc + x * l / 2 - y * ww / 2,
                         fc + x * l / 2 + y * ww / 2, fc - x * l / 2 + y * ww / 2])
        z = quad[:, 2]
        if np.any(z <= 1e-3):
            continue
        uv = np.stack([fx * quad[:, 0] / z + cx, fy * quad[:, 1] / z + cy], 1)
        poly = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(poly, np.int32(uv), 1)
        inter = int(((poly > 0) & image_mask).sum())
        union = int(((poly > 0) | image_mask).sum())
        iou = inter / max(union, 1)
        if iou > best[0]:
            best = (iou, x, y, (l, ww, fc, iou))
    if best[1] is None:
        return None
    x, y = best[1], best[2]
    l, ww, fc, iou = best[3]
    return x, y, l, ww, fc, iou


def _axis_from_image_hint(angle_deg, ref_pt, n, plane_d, K):
    """把图像里的方向反投影到平面上，得到平面内的 3D 轴。

    图像方向来自 mask 的最小面积矩形，比点云自身更接近真实轮廓。
    """
    fx, fy, cx, cy = K
    u0 = fx * ref_pt[0] / ref_pt[2] + cx
    v0 = fy * ref_pt[1] / ref_pt[2] + cy
    a = np.radians(angle_deg)
    pts3 = []
    for sign in (-1.0, 1.0):
        u = u0 + sign * np.cos(a) * 60.0
        v = v0 + sign * np.sin(a) * 60.0
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        denom = float(n @ ray)
        if abs(denom) < 1e-9:
            continue
        t = -plane_d / denom
        pts3.append(ray * t)
    if len(pts3) < 2:
        return None
    x = pts3[1] - pts3[0]
    x = x - n * float(np.dot(x, n))
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-9 else None


def _project_pt(p, K):
    fx, fy, cx, cy = K
    return np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])


def _unproject(u, v, n, plane_d, K):
    fx, fy, cx, cy = K
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    denom = float(n @ ray)
    if abs(denom) < 1e-9:
        return None
    return ray * (-plane_d / denom)


def clean_by_plane(points, thresh=0.005, voxel=0.012, iters=60):
    """剔除离主平面过远的点（键盘后方的背景/深度飞点）。

    mask 里常混入物体后方的桌面点，离平面 5cm 以上；不剔除会把框撑厚。
    """
    pts_in = np.asarray(points, dtype=np.float32)
    if len(pts_in) < 50:
        return pts_in, np.ones(len(pts_in), dtype=bool)
    p = voxel_downsample(pts_in, voxel)
    # RANSAC 的内点阈值比最终阈值更严一点，保证平面估计本身干净
    r = ransac_plane(p, thresh=min(thresh * 0.8, 0.006), iters=iters)
    if r is None:
        return pts_in, np.ones(len(pts_in), dtype=bool)
    n, d, mask = r
    if mask.mean() < 0.3:
        return pts_in, np.ones(len(pts_in), dtype=bool)
    dist = np.abs(np.asarray(points, dtype=np.float32) @ n + d)
    keep = dist < thresh
    if keep.sum() < 50:
        return np.asarray(points, dtype=np.float32), np.ones(len(points), dtype=bool)
    return np.asarray(points, dtype=np.float32)[keep], keep



# ---------------------------------------------------------------- 桌子平面
def ransac_plane_largest(points, thresh=0.012, iters=120, min_ratio=0.25):
    """从一片点云里找最大的平面（桌子）。返回 (n, d, ratio) 或 None。

    与 ransac_plane 的区别：这里专门给"整帧深度点云"用，允许大面积但要求内点率高。
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 200:
        return None
    r = ransac_plane(pts, thresh=thresh, iters=iters)
    if r is None:
        return None
    n, d, mask = r
    ratio = float(mask.mean())
    if ratio < min_ratio:
        return None
    # 法线朝向相机
    if float(n @ pts.mean(axis=0)) > 0:
        n, d = -n, -d
    return n.astype(np.float64), float(d), ratio


def depth_to_points(depth_mm, K, stride=4, min_depth=0.30, max_depth=1.60):
    """把深度图稀疏采样并反投影成 3D 点云（相机光学系），用于桌子平面估计。"""
    fx, fy, cx, cy = K
    h, w = depth_mm.shape[:2]
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    d = depth_mm[np.ix_(vs, us)].astype(np.float32) * 0.001
    vv, uu = np.meshgrid(vs, us, indexing='ij')
    keep = (d > min_depth) & (d < max_depth)
    if keep.sum() < 200:
        return None
    z = d[keep]
    u = uu[keep].astype(np.float32)
    v = vv[keep].astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------- 圆柱 OBB
def _circle_ransac(xy, thresh=0.012, iters=120, min_inliers=40):
    """3 点 RANSAC 拟合圆，返回 (cx, cy, r, inlier_ratio)。"""
    n = len(xy)
    if n < 3:
        return None
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p1, p2, p3 = xy[i]
        a = 2 * (p2 - p1)
        b = 2 * (p3 - p1)
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) < 1e-9:
            continue
        c2 = (p2 @ p2 - p1 @ p1)
        c3 = (p3 @ p3 - p1 @ p1)
        cx = (c2 * b[1] - c3 * a[1]) / det
        cy = (a[0] * c3 - b[0] * c2) / det
        r = float(np.hypot(p1[0] - cx, p1[1] - cy))
        if not np.isfinite(r) or r < 1e-3:
            continue
        d = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        inl = int((d < thresh).sum())
        if inl > best[0]:
            best = (inl, (cx, cy, r))
    if best[1] is None or best[0] < min_inliers:
        return None
    cx, cy, r = best[1]
    return float(cx), float(cy), float(r), best[0] / n


def fit_cylinder_obb(points, label, n_table, d_table, K=None, margin=0.002,
                     thresh_circle=0.012):
    """圆柱类物体（cup/bottle）的 OBB：z = 桌面法线，x = 把手方向（可见时）。

    流程（与设计一致）：
      1) 投影到桌面平面 → 米制 2D
      2) RANSAC 圆给稳健初值 r0（避免"被垫板污染的点"直接做最小二乘）
      3) 足迹过滤 |rr-r0| < 1.5cm → 精拟合圆 → r
      4) 径向残差 r(φ) 找把手：峰 > max(0.5cm, 0.15r) 且 > 1.5×中位 → yaw 有效
      5) 高度 h = p99 - p1（沿桌面法线，抗单点飞点）
      6) 框 = (2r + d_h) × 2r × h；中心沿把手方向偏 d_h/2
    返回 (Obb, meta) 或 (None, None)。meta = {'yaw_valid','handle_deg','r','d_handle','h'}
    """
    P = np.asarray(points, dtype=np.float32).astype(np.float64)
    if len(P) < 80:
        return None, None
    n = np.asarray(n_table, dtype=np.float64)
    n = n / np.linalg.norm(n)

    # 桌面内正交基
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, n)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - n * float(np.dot(u, n))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    c_ref = P.mean(axis=0)
    loc = np.stack([(P - c_ref) @ u, (P - c_ref) @ v], axis=1)

    # 半径两段式：RANSAC 初值 → 足迹过滤 → 精拟合
    init = _circle_ransac(loc.astype(np.float32), thresh=thresh_circle)
    if init is None:
        return None, None
    cx0, cy0, r0, _ratio = init
    rr0 = np.hypot(loc[:, 0] - cx0, loc[:, 1] - cy0)
    keep = np.abs(rr0 - r0) < 0.015
    if keep.sum() < 30:
        keep = np.ones(len(loc), dtype=bool)
    L = loc[keep]
    A = np.stack([L[:, 0], L[:, 1], np.ones(len(L))], axis=1)
    b = (L ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))
    if not np.isfinite(r) or r < 0.005:
        return None, None

    # ---- 用"杯身中段轮廓"重估半径与中心（半圈弧的圆拟合有偏，这里做交叉校验）----
    hn_all = P @ n
    h_lo, h_hi = np.percentile(hn_all, [20, 80])
    band = (hn_all >= h_lo) & (hn_all <= h_hi)
    if band.sum() >= 40:
        Lb = loc[band]
        best_ext, best_mid = 0.0, None
        for deg in range(0, 180, 10):
            a = np.radians(deg)
            e1 = np.array([np.cos(a), np.sin(a)])
            s_ = Lb @ e1
            ext = float(s_.max() - s_.min())
            if ext > best_ext:
                best_ext = ext
                mid = (s_.max() + s_.min()) / 2.0
                best_mid = (mid * e1, e1, ext)
        if best_mid is not None and best_ext > 0.02:
            # 轮廓法：r_sil = 最大方向跨度的一半；中心 = 该方向跨度中点
            e1 = best_mid[1]
            e2 = np.array([-e1[1], e1[0]])
            s1 = Lb @ e1
            s2 = Lb @ e2
            c1 = (s1.max() + s1.min()) / 2.0
            c2 = (s2.max() + s2.min()) / 2.0
            r_sil = best_ext / 2.0
            cx_sil = c1 * e1[0] + c2 * e2[0]
            cy_sil = c1 * e1[1] + c2 * e2[1]
            # 交叉校验：两者差 >20% 时以轮廓法为准（半圈弧拟合会高估半径并把中心拉偏）
            if abs(r_sil - r) / max(r, 1e-6) > 0.20:
                r = r_sil
            else:
                r = 0.5 * (r + r_sil)
            if abs(cx_sil - cx) > 0.005 or abs(cy_sil - cy) > 0.005:
                cx, cy = cx_sil, cy_sil

    # 径向残差 → 把手
    Lk = loc[keep]
    dx = Lk[:, 0] - cx
    dy = Lk[:, 1] - cy
    rr = np.hypot(dx, dy)
    phi = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    bins = np.arange(0, 360, 15)
    prof = np.full(len(bins), np.nan)
    for i, b0 in enumerate(bins):
        sel = (phi >= b0) & (phi < b0 + 15)
        if sel.any():
            prof[i] = (rr[sel].max() - r)
    peak = float(np.nanmax(prof)) if np.isfinite(prof).any() else 0.0
    med = float(np.nanmedian(prof)) if np.isfinite(prof).any() else 0.0
    yaw_valid = bool(peak > max(0.005, 0.15 * r) and peak > 1.5 * max(med, 1e-3))
    if yaw_valid:
        k = int(np.nanargmax(prof))
        # 用峰附近点做圆周平均，把 15° 分箱细化到亚度
        near = (phi >= bins[k] - 15) & (phi <= bins[k + 1] + 15 if k + 1 < len(bins) else phi <= bins[k] + 30)
        near = near & (rr > r + 0.5 * max(peak, 1e-3))
        if near.sum() >= 5:
            a = np.radians(phi[near])
            handle_deg = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)
        else:
            handle_deg = float(bins[k] + 7.5)
        d_handle = float(max(peak, 0.0))
    else:
        handle_deg = 0.0
        d_handle = 0.0

    # 高度：沿 n 的 p99 - p1
    hn = P @ n
    # 高度：按最初定的口径 —— 过滤后点云的 z_max − z_min（沿桌面法线）
    hn = P @ n
    z_lo = float(hn.min())
    z_hi = float(hn.max())
    h = z_hi - z_lo
    if not np.isfinite(h) or h < 0.030 or keep.sum() < 60 or r < 0.015:
        return None, None
    z_center = z_lo + h / 2.0

    # 方向：x = 把手方向（有效时）；否则取规范（u 在平面上的方向）
    if yaw_valid:
        a = np.radians(handle_deg)
        x_axis = u * np.cos(a) + v * np.sin(a)
        x_axis = x_axis - n * float(np.dot(x_axis, n))
        x_axis /= np.linalg.norm(x_axis)
    else:
        x_axis = u.copy()
    y_axis = np.cross(n, x_axis)
    R = np.stack([x_axis, y_axis, n], axis=1).astype(np.float32)

    # 中心：圆心的平面坐标 + 沿把手偏 d_handle/2
    center_plane = c_ref + u * cx + v * cy
    if yaw_valid:
        center_plane = center_plane + x_axis * (d_handle / 2.0)
    # 沿法线放到 h/2 处
    center = center_plane + n * (z_center - float(center_plane @ n))

    dims = np.array([2.0 * r + d_handle + 2 * margin,
                     2.0 * r + 2 * margin,
                     h + 2 * margin], dtype=np.float32)
    obb = Obb(center.astype(np.float32), R, dims, n.astype(np.float32),
              plane_ratio=float(keep.mean()), rmse=0.0, n_points=int(keep.sum()),
              source='cylinder')
    meta = {'yaw_valid': yaw_valid, 'handle_deg': handle_deg,
            'r': r, 'd_handle': d_handle, 'h': h}
    return obb, meta


def fit_obb_on_plane(points, n_pl, d_pl, margin=0.002, band=0.010,
                     min_frac=0.25, min_keep=40):
    """物体站在某个平面（桌面，或另一个物体的顶面）上时的 OBB。

    关键点：不再让点云自己猜 z 轴，而是**用支撑面的法线当 z**，
    底面就压在支撑面上。这样 x/y 方向的污染点只影响长宽，不会把框掰歪。

    触发条件：点云里有 |h| < band 的点占比 >= min_frac（说明 mask 吃进了支撑面）。
    h = n·p + d，即相对支撑面的高度。
    """
    P = np.asarray(points, dtype=np.float32)
    if len(P) < min_keep:
        return None
    h = P @ np.asarray(n_pl, dtype=np.float32) + float(d_pl)
    frac = float((np.abs(h) < band).mean())
    if frac < min_frac:
        return None
    keep = h > band                       # 去掉贴在支撑面上的污染点
    if keep.sum() < min_keep:
        # 点云本身就是一层薄面（比如只看到书的顶面）：用全部点，
        # 高度取"可见厚度"，避免直接判失败。
        keep = np.ones(len(h), dtype=bool)
    Q = P[keep]
    hq = h[keep]

    n = np.asarray(n_pl, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # 平面内一组基
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    uv = Q.astype(np.float64) @ np.stack([u, v], axis=1)
    rect = cv2.minAreaRect(uv.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64)
    e1 = box[1] - box[0]
    e2 = box[3] - box[0]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if l1 < 0.01 or l2 < 0.01:
        return None
    ax1 = e1 / l1
    # 3D 方向
    X = ax1[0] * u + ax1[1] * v
    X /= np.linalg.norm(X)
    Y = np.cross(n, X)
    cx, cy = rect[0]

    h_top = float(np.percentile(hq, 99))
    if h_top < 0.01:
        return None
    p0 = cx * u + cy * v
    p0 = p0 - (float(n @ p0) + float(d_pl)) * n      # 投到支撑面上
    center = p0 + 0.5 * h_top * n

    R = np.stack([X, Y, n], axis=1)
    dims = np.array([max(l1, l2), min(l1, l2), h_top], dtype=np.float32) + margin
    obb = Obb(center, R, dims, n, plane_ratio=1.0, rmse=0.0,
              n_points=int(keep.sum()), source='support')
    return contain_points(obb, Q)


def split_raised(points, thresh=0.015, min_points=80, min_extent=0.02,
                 voxel=0.008, plane_thresh=0.006):
    """把「压在另一个物体上的物体」从同一片点云里分出来（单视角下的双层分离）。

    原理：主平面 = 下面那个物体的表面；沿法线方向离平面 > thresh、且**更靠近相机**
    的一片点，就是压在上面的物体（杯子/盒子/第二本书）。

    返回 (base_mask, raised_mask)：
        base_mask    下面物体的点（可直接拿去 fit_obb）
        raised_mask  上面物体的点（拿去 fit_obb 得到第二个 3D 框）
    判定不成立时 raised_mask 全 False，调用方按原逻辑处理。
    """
    pts = np.asarray(points, dtype=np.float32)
    n_all = len(pts)
    empty = np.zeros(n_all, dtype=bool)
    keep_all = np.ones(n_all, dtype=bool)
    if n_all < 200:
        return keep_all, empty

    p = voxel_downsample(pts, voxel)
    r = ransac_plane(p, thresh=plane_thresh, iters=80)
    if r is None:
        return keep_all, empty
    n, d, inl = r
    if inl.mean() < 0.45:          # 没有明显的主平面 → 不是"平面+凸起"场景
        return keep_all, empty

    s = pts @ n + d                # 有符号距离
    pos = s > thresh
    neg = s < -thresh
    if pos.sum() == 0 and neg.sum() == 0:
        return keep_all, empty

    # 哪一侧更靠近相机，哪一侧就是"上面那个物体"
    med = lambda m: float(np.median(pts[m, 2]))
    if pos.sum() and neg.sum():
        raised = pos if med(pos) < med(neg) else neg
    else:
        raised = pos if pos.sum() else neg

    if raised.sum() < min_points:
        return keep_all, empty

    # 凸起必须够"厚"够"大"，否则只是深度噪声/边缘飞点
    band = float(np.abs(s[raised]).max())
    ext = pts[raised].max(axis=0) - pts[raised].min(axis=0)
    if band < min_extent or float(np.sort(ext)[-2]) < min_extent * 1.5:
        return keep_all, empty

    return ~raised, raised


def tighten_to_points(obb, points, lo=1.0, hi=99.0, margin=0.001):
    """把框收紧到"点云自身的范围"（沿框轴的分位范围）。

    contain_points 只保证"包住"，尺寸仍由 mask 四边形决定，mask 偏大时框就松；
    这里再按点云范围收一次，让框贴着实际点云（遮挡时会偏小，由调用方决定是否启用）。
    """
    if obb is None or points is None or len(points) < 20:
        return obb
    rel = (np.asarray(points, dtype=np.float32) - obb.center) @ obb.rotation
    q_lo = np.percentile(rel, lo, axis=0)
    q_hi = np.percentile(rel, hi, axis=0)
    ext = np.maximum(q_hi - q_lo, 0.004)
    half_old = obb.dims / 2.0
    # 只收 x/y（footprint）：z 是厚度方向，单视角只看到上表面，
    # 点云在 z 上的范围是"表面粗糙度"而不是物体高度，不能用来收紧。
    new_dims = obb.dims.copy()
    new_dims[0] = min(ext[0] + margin, obb.dims[0])
    new_dims[1] = min(ext[1] + margin, obb.dims[1])
    shift = (q_lo + q_hi) / 2.0
    shift[2] = 0.0                     # 厚度方向不动
    shift = np.clip(shift, -half_old, half_old)
    obb.center = (obb.center + obb.rotation @ shift.astype(np.float32)).astype(np.float32)
    obb.dims = new_dims.astype(np.float32)
    return obb


def shrink_if_mask_oversized(obb, points, fill_gate=0.6, margin=0.004):
    """mask 四边形远大于点云时，按点云范围收 footprint。

    mask 会把"书 + 垫纸"识别成一个物体（COCO 常见），四边形跟着 mask 走就横跨两者；
    而点云（真有深度的部分）只覆盖书。用 fill = 点云范围 / 框尺寸 判断：
    正常物体 fill 0.75~1.0；mask 明显偏大时 fill < 0.6，此时按点云收紧。
    """
    if obb is None or points is None or len(points) < 30:
        return obb, 1.0
    rel = (np.asarray(points, dtype=np.float32) - obb.center) @ obb.rotation
    lo = np.percentile(rel, 1.0, axis=0)
    hi = np.percentile(rel, 99.0, axis=0)
    ext = np.maximum(hi - lo, 1e-4)
    fill = float(np.min(ext[:2] / np.maximum(obb.dims[:2], 1e-6)))
    if fill >= fill_gate:
        return obb, fill
    new = obb.dims.copy()
    shift = (lo + hi) / 2.0
    shift[2] = 0.0
    new[0] = min(ext[0] + margin, obb.dims[0])
    new[1] = min(ext[1] + margin, obb.dims[1])
    obb.center = (obb.center + obb.rotation @ shift.astype(np.float32)).astype(np.float32)
    obb.dims = new.astype(np.float32)
    return obb, fill


def contain_points(obb, points, margin=0.002):
    """把 OBB 扩到严格包住点云（在 OBB 自身坐标系里求紧包围盒）。

    框的朝向由 mask 决定，尺寸再被点云约束一次，保证「点云一定在框内」。
    """
    if obb is None or points is None or len(points) < 4:
        return obb
    rel = (np.asarray(points, dtype=np.float32) - obb.center) @ obb.rotation
    # 用 0.5/99.5 分位而不是 min/max：深度飞点不该把框撑大
    lo = np.percentile(rel, 0.5, axis=0)
    hi = np.percentile(rel, 99.5, axis=0)
    half = obb.dims / 2.0
    # 若点云超出当前框，则扩到点云并平移中心
    new_lo = np.minimum(lo, -half)
    new_hi = np.maximum(hi, half)
    shift = (new_hi + new_lo) / 2.0
    obb.center = (obb.center + obb.rotation @ shift.astype(np.float32)).astype(np.float32)
    obb.dims = (new_hi - new_lo + 2.0 * margin).astype(np.float32)
    return obb


def mask_quad(mask):
    """从 mask 提取实际四边形轮廓（比最小外接矩形更贴合物体的真实投影）。"""
    if mask is None or not mask.any():
        return None
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return None


def _obb_from_quad(obb, quad, K, thickness, margin):
    """由 mask 的四边形轮廓直接构造 OBB 的面。

    四边形 4 角反投影到平面后，用「对边向量平均」得到矩形的两轴与边长，
    再取平均得到面中心。无优化、无迭代，结果确定。
    """
    if obb is None or quad is None:
        return None
    n = np.asarray(obb.normal, dtype=np.float64)
    fx, fy, cx, cy = K
    face_old = np.asarray(obb.center, dtype=np.float64) + n * (thickness / 2.0)
    d_plane = -float(n @ face_old)

    P = []
    for u, v in np.asarray(quad, dtype=np.float64):
        p = _unproject(float(u), float(v), n, d_plane, K)
        if p is None:
            return None
        P.append(p)
    P = np.array(P)

    # 在平面坐标里把这 4 个点拟合成一个真正的矩形（2D 优化，收敛可靠）
    u0, v0 = _basis_from_normal(n)
    P2 = np.stack([(P - P.mean(0)) @ u0, (P - P.mean(0)) @ v0], axis=1)

    def rect_pts(cx2, cy2, th, ll, ww):
        c, s_ = np.cos(th), np.sin(th)
        R2 = np.array([[c, -s_], [s_, c]])
        local = np.array([[-ll / 2, -ww / 2], [ll / 2, -ww / 2],
                          [ll / 2, ww / 2], [-ll / 2, ww / 2]])
        return (R2 @ local.T).T + np.array([cx2, cy2])

    # 初值：对边平均
    e0, e1, e2, e3 = P2[1] - P2[0], P2[2] - P2[1], P2[3] - P2[2], P2[0] - P2[3]
    a0 = 0.5 * (e0 + e2) if np.dot(e0, e2) > 0 else 0.5 * (e0 - e2)
    b0 = 0.5 * (e1 + e3) if np.dot(e1, e3) > 0 else 0.5 * (e1 - e3)
    l0, w0 = np.linalg.norm(a0), np.linalg.norm(b0)
    th0 = np.arctan2(a0[1], a0[0])
    try:
        from scipy.optimize import least_squares
        from scipy.spatial import distance_matrix
        p0 = rect_pts(0, 0, th0, l0, w0)
        cost = distance_matrix(p0, P2)
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        order = np.empty(4, dtype=int); order[ri] = ci
        tgt = P2[order]

        def res(p_):
            return (rect_pts(p_[0], p_[1], p_[2], p_[3], p_[4]) - tgt).ravel()

        # 初值残差已经够小就别优化（省 2-4 ms/物体）
        r0 = np.linalg.norm(res([0, 0, th0, l0, w0])) / 4.0
        if r0 < 0.002:
            cx2, cy2, th, l, w = 0.0, 0.0, th0, l0, w0
        else:
            sol = least_squares(res, [0, 0, th0, l0, w0],
                                bounds=([-0.3, -0.3, -np.pi, 0.02, 0.01],
                                        [0.3, 0.3, np.pi, 0.8, 0.5]),
                                max_nfev=25)
            cx2, cy2, th, l, w = sol.x
    except Exception:
        cx2, cy2, th, l, w = 0.0, 0.0, th0, l0, w0

    x_ax = u0 * np.cos(th) + v0 * np.sin(th)
    x_ax = x_ax - n * float(np.dot(x_ax, n))
    x_ax /= np.linalg.norm(x_ax) + 1e-12
    y_ax = np.cross(n, x_ax)
    if l < w:
        x_ax, y_ax, l, w = y_ax, x_ax, w, l
    face_shift = u0 * cx2 + v0 * cy2

    center_face = P.mean(axis=0) + face_shift

    return x_ax, y_ax, l + 2 * margin, w + 2 * margin, center_face


def _fit_face_to_rect(obb, mask_rect, K, quad=None):
    """把可见面的 4 个角点拟合到 mask 矩形的 4 个角点。

    只对标边长不够：3D 面投影是梯形，边长对上了形状仍然歪。
    这里用最近邻确定角点对应，再最小二乘求 面中心 + 长 + 宽。
    """
    if obb is None or mask_rect is None:
        return obb
    try:
        from scipy.optimize import least_squares
        from scipy.spatial import distance_matrix
    except Exception:
        return obb

    fx, fy, cx, cy = K
    n = np.asarray(obb.normal, dtype=np.float64)
    x_ax = np.asarray(obb.rotation[:, 0], dtype=np.float64)
    y_ax = np.asarray(obb.rotation[:, 1], dtype=np.float64)
    h = float(obb.dims[2])
    target = (np.asarray(quad, dtype=np.float64) if quad is not None
              else cv2.boxPoints(mask_rect).astype(np.float64))

    def face_corners(c, l, w):
        return np.array([
            c - x_ax * l / 2 - y_ax * w / 2,
            c + x_ax * l / 2 - y_ax * w / 2,
            c + x_ax * l / 2 + y_ax * w / 2,
            c - x_ax * l / 2 + y_ax * w / 2,
        ])

    def proj(pts3):
        z = pts3[:, 2]
        if np.any(z <= 1e-3):
            return None
        return np.stack([fx * pts3[:, 0] / z + cx,
                         fy * pts3[:, 1] / z + cy], axis=1)

    c0 = np.asarray(obb.center, dtype=np.float64) + n * (h / 2.0)
    l0, w0 = float(obb.dims[0]), float(obb.dims[1])
    p0 = proj(face_corners(c0, l0, w0))
    if p0 is None:
        return obb

    # 最近邻确定对应关系（一次性，优化中固定）
    cost = distance_matrix(p0, target)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        order = np.empty(4, dtype=int)
        order[ri] = ci
    except Exception:
        order = np.argmin(cost, axis=1)
    tgt = target[order]

    def residuals(p):
        c = p[:3]
        pr = proj(face_corners(c, p[3], p[4]))
        if pr is None:
            return np.full(8, 1e3)
        return (pr - tgt).ravel()

    x0 = np.array([c0[0], c0[1], c0[2], l0, w0])
    lower = [c0[0] - 0.05, c0[1] - 0.05, c0[2] - 0.05, 0.02, 0.01]
    upper = [c0[0] + 0.05, c0[1] + 0.05, c0[2] + 0.05, 0.6, 0.4]
    try:
        res = least_squares(residuals, x0, bounds=(lower, upper),
                            loss='linear', max_nfev=30)
    except Exception:
        return obb
    if not res.success:
        return obb

    c = res.x[:3]
    obb.center = (c - n * (h / 2.0)).astype(np.float32)
    obb.dims = np.array([res.x[3], res.x[4], h], dtype=np.float32)
    return obb


def _match_mask_rect(obb, mask_rect, K, iters=3):
    """按 mask 最小矩形的投影尺寸标定 OBB 的 l/w，让框在图像上的投影严格贴合轮廓。

    点云平面投影会有系统偏差（侧裙点、法线误差），直接量投影长度再按比例修正最可靠。
    """
    if obb is None or mask_rect is None:
        return obb
    fx, fy, cx, cy = K
    (rw, rh) = mask_rect[1]
    if rw < 5 or rh < 5:
        return obb

    for _ in range(iters):
        corners = obb.corners()
        z = corners[:, 2]
        if np.any(z <= 1e-3):
            break
        uv = np.stack([fx * corners[:, 0] / z + cx,
                       fy * corners[:, 1] / z + cy], axis=1)
        lx = float(np.linalg.norm(uv[1] - uv[0]))
        ly = float(np.linalg.norm(uv[2] - uv[1]))
        if lx < 3 or ly < 3:
            break
        sx = float(np.clip(rw / lx, 0.75, 1.3))
        sy = float(np.clip(rh / ly, 0.75, 1.3))
        obb.dims[0] *= sx
        obb.dims[1] *= sy
        if abs(sx - 1) < 0.01 and abs(sy - 1) < 0.01:
            break
    return obb


def _grow_to_cover(obb, mask, K, target=0.995, max_scale=1.25, step=1.01):
    """把 OBB 逐步放大，直到它在图像上的投影覆盖 mask 的比例达到 target。

    单视角下键盘侧面不在拟合平面内，纯几何求出的框会漏掉边缘；
    这里只扩到刚好包住，避免像固定余量那样盲目放大。
    """
    if obb is None or mask is None:
        return obb
    fx, fy, cx, cy = K
    h_img, w_img = mask.shape
    # 只对最大连通区域算覆盖率，避免零星碎块把框强行撑大
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return obb
    core = np.zeros((h_img, w_img), np.uint8)
    cv2.drawContours(core, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    core = core.astype(bool)
    m_area = int(core.sum())
    if m_area == 0:
        return obb

    scale = 1.0
    for _ in range(24):
        corners = obb.corners()
        z = corners[:, 2]
        if np.any(z <= 1e-3):
            break
        uv = np.stack([fx * corners[:, 0] / z + cx,
                       fy * corners[:, 1] / z + cy], axis=1).astype(np.int32)
        hull = cv2.convexHull(uv)
        canvas = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(canvas, [hull], 1)
        cov = float(np.logical_and(core, canvas.astype(bool)).sum()) / m_area
        if cov >= target or scale >= max_scale:
            break
        scale *= step
        obb.dims = obb.dims * step
    return obb


def _obb_from_mask_contour(mask, n, plane_d, K, thickness, margin):
    """把 mask 轮廓逐点反投影到平面上，再求平面内的最小矩形。

    比「反投影矩形角点」正确：角点在物体外、深度不对，会被透视放大；
    轮廓点本身就贴合物体的可见边界。
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 8:
        return None
    pts2d = contour.reshape(-1, 2).astype(np.float64)
    if len(pts2d) > 600:
        step = len(pts2d) // 600 + 1
        pts2d = pts2d[::step]

    fx, fy, cx, cy = K
    rays = np.stack([(pts2d[:, 0] - cx) / fx,
                     (pts2d[:, 1] - cy) / fy,
                     np.ones(len(pts2d))], axis=1)
    denom = rays @ n
    good = np.abs(denom) > 1e-9
    if good.sum() < 8:
        return None
    t = -plane_d / denom[good]
    P = rays[good] * t[:, None]

    ref = P.mean(axis=0)
    u0, v0 = _basis_from_normal(n)
    coords = np.stack([(P - ref) @ u0, (P - ref) @ v0], axis=1).astype(np.float32)
    (c2x, c2y), (rw, rh), ang = cv2.minAreaRect(coords)

    a = np.radians(ang)
    e1 = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
    e2 = np.array([-np.sin(a), np.cos(a)], dtype=np.float32)
    ax1 = u0 * e1[0] + v0 * e1[1]
    ax2 = u0 * e2[0] + v0 * e2[1]
    if rw >= rh:
        x_axis, y_axis, l, w = ax1, ax2, rw, rh
    else:
        x_axis, y_axis, l, w = ax2, ax1, rh, rw

    if float(np.dot(x_axis, x_axis)) < 1e-12:
        return None
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)

    face_center = ref + u0 * float(c2x) + v0 * float(c2y)
    center = face_center - n * (thickness / 2.0)
    return x_axis, y_axis, l + 2.0 * margin, w + 2.0 * margin, center


def _obb_from_mask_rect(rect, n, plane_d, K, thickness, margin):
    """把 mask 的最小矩形反投影到平面，直接得到贴合轮廓的 OBB 尺寸与朝向。

    这样框在图像上的投影就等于 mask 的矩形，不会偏大或偏小。
    """
    corners = cv2.boxPoints(rect)           # 4x2 图像坐标
    pts = []
    for u, v in corners:
        p = _unproject(float(u), float(v), n, plane_d, K)
        if p is None:
            return None
        pts.append(p)
    pts = np.array(pts)

    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    e23 = pts[3] - pts[2]
    e30 = pts[0] - pts[3]
    x_dir = e01 + e23
    y_dir = e12 + e30
    if np.linalg.norm(x_dir) < 1e-9 or np.linalg.norm(y_dir) < 1e-9:
        return None

    l = 0.5 * (np.linalg.norm(e01) + np.linalg.norm(e23))
    w = 0.5 * (np.linalg.norm(e12) + np.linalg.norm(e30))

    x = x_dir - n * float(np.dot(x_dir, n))
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x = x / nx
    y = np.cross(n, x)

    if l < w:                                # 长边作为 x 轴
        x, y = y, x
        l, w = w, l

    face_center = pts.mean(axis=0)
    center = face_center - n * (thickness / 2.0)
    return x, y, l + 2.0 * margin, w + 2.0 * margin, center


def _refine_axis(rect_pts, ref, n, base_axis, K, hint_deg,
                 span=10.0, step=1.0):
    """绕法线微调长轴，使投影后的长边角度最接近 mask 的方向。"""
    best_err, best = 1e9, None
    perp = np.cross(n, base_axis)
    for dphi in np.arange(-span, span + 1e-6, step):
        a = np.radians(dphi)
        x = base_axis * np.cos(a) + perp * np.sin(a)
        x = x - n * float(np.dot(x, n))
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            continue
        x = x / nx
        y = np.cross(n, x)
        px = rect_pts @ x
        py = rect_pts @ y
        l = float(px.max() - px.min())
        w = float(py.max() - py.min())
        if l < w:
            x, y = y, x
            l, w = w, l
            px, py = py, px
        cpx = 0.5 * (px.max() + px.min())
        cpy = 0.5 * (py.max() + py.min())
        fc = ref + x * (cpx - float(ref @ x)) + y * (cpy - float(ref @ y))
        p0 = fc - x * (l / 2.0) - y * (w / 2.0)
        p1 = fc + x * (l / 2.0) - y * (w / 2.0)
        if p0[2] <= 1e-3 or p1[2] <= 1e-3:
            continue
        u0, v0 = _project_pt(p0, K)
        u1, v1 = _project_pt(p1, K)
        ang = np.degrees(np.arctan2(v1 - v0, u1 - u0))
        err = abs(((ang - hint_deg + 90.0) % 180.0) - 90.0)
        if err < best_err:
            best_err, best = err, (x, y, l, w, fc)
    return best


def _pca_obb(points, label, thickness):
    """平面拟合失败时的退化方案：全局 PCA。"""
    c = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    R = vh.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    proj = (points - c) @ R
    dims = np.array([proj[:, i].max() - proj[:, i].min() for i in range(3)])
    dims[2] = max(dims[2], thickness)
    return Obb(c, R, dims, R[:, 2], plane_ratio=0.0,
               rmse=float(np.std(proj[:, 2])), n_points=len(points), source='pca')


def filter_to_obb(points, obb, margin=0.012, min_keep=0.3):
    """返回 (保留的点, 布尔掩码)，去掉落在 OBB 外的背景离群点。

    仅对平面拟合成功的实例生效；过滤后点数太少则不过滤，避免误删。
    """
    keep = np.ones(len(points), dtype=bool)
    if obb is None or obb.source != 'plane':
        return points, keep
    rel = (np.asarray(points, dtype=np.float32) - obb.center) @ obb.rotation
    inside = np.all(np.abs(rel) <= (obb.dims / 2.0 + margin), axis=1)
    if inside.sum() < max(20, min_keep * len(points)):
        return points, keep
    return points[inside], inside


def fit_obb(points, label='', thickness=None, voxel=0.008, plane_thresh=0.008,
            max_points=2000, ransac_iters=60, margin=0.005,
            angle_hint=None, K=None, mask_rect=None, image_mask=None,
            mask_quad_pts=None):
    """对单个实例点云求 OBB。返回 Obb 或 None。

    为了满足 15fps，先随机抽稀到 max_points，再体素下采样；
    离群点由 RANSAC 自带的内点筛选处理，不再单独跑 SOR。
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 50:
        return None

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), max_points, replace=False)]

    pts = voxel_downsample(points, voxel)
    if len(pts) < 30:
        return None

    prior = thickness if thickness is not None else THICKNESS_PRIOR.get(
        label, THICKNESS_PRIOR['default'])

    plane = ransac_plane(pts, thresh=plane_thresh, iters=ransac_iters)
    if plane is None:
        return _pca_obb(pts, label, prior)

    n, d, mask = plane
    inliers = pts[mask]
    ratio = mask.mean()
    if ratio < 0.25 or len(inliers) < 30:
        return _pca_obb(pts, label, prior)

    # 法线朝向相机
    centroid = inliers.mean(axis=0)
    if float(n @ centroid) > 0:
        n = -n
        d = -d

    u, v = _basis_from_normal(n)
    # 用「平面附近带宽内」的全部点，而不是只用平面内点
    # 只用平面内点求矩形：带宽会把键盘侧裙（离平面 2cm 左右）也算进来，
    # 投影到平面上会把长度方向撑长 20%+，导致框比物体长。
    rect_pts = inliers
    ref = rect_pts.mean(axis=0)

    x_axis = None
    refined = None
    if angle_hint is not None and K is not None:
        x_axis = _axis_from_image_hint(angle_hint, ref, n, d, K)
        # mask 近正方形时 hint 的长边判定会差 90°，先用 4 候选 + IoU 选一次
        picked = _axis_candidates_iou(rect_pts, image_mask, n, x_axis, K, margin)
        if picked is not None:
            x_axis, y_axis, l, w, face_center, _iou = picked
            refined = (x_axis, y_axis, l, w, face_center)
        elif x_axis is not None:
            refined = _refine_axis(rect_pts, ref, n, x_axis, K, angle_hint)
    if refined is not None:
        x_axis, y_axis, l, w, face_center = refined
    elif x_axis is None:
        rel = rect_pts - ref
        uv = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)
        import cv2
        (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(uv)
        a = np.radians(angle)
        e1 = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
        e2 = np.array([-np.sin(a), np.cos(a)], dtype=np.float32)
        ax1 = u * e1[0] + v * e1[1]
        ax2 = u * e2[0] + v * e2[1]
        x_axis, y_axis = (ax1, ax2) if rw >= rh else (ax2, ax1)
    else:
        y_axis = np.cross(n, x_axis)

    if refined is None:
        # 沿两轴投影求范围
        px = rect_pts @ x_axis
        py = rect_pts @ y_axis
        l = float(px.max() - px.min())
        w = float(py.max() - py.min())
        if l < w:                      # 长边作为 x 轴
            x_axis, y_axis = y_axis, x_axis
            l, w = w, l
            px, py = py, px
        cpx = 0.5 * (px.max() + px.min())
        cpy = 0.5 * (py.max() + py.min())
        face_center = ref + x_axis * (cpx - float(ref @ x_axis)) \
            + y_axis * (cpy - float(ref @ y_axis))

    l += 2.0 * margin
    w += 2.0 * margin

    # 厚度：类别先验；若点云在法线方向有明显厚度，取较大者
    # 厚度只用平面内点算（全部点会被背景离群点撑大）
    proj_n = (inliers - face_center) @ n
    observed = float(np.percentile(proj_n, 95) - np.percentile(proj_n, 5))
    h = max(prior, min(observed, 0.15))

    # 可见面在外侧，中心沿 -n 往物体内部偏移 h/2
    center = face_center - n * (h / 2.0)

    R = np.stack([x_axis, y_axis, n], axis=1).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
        y_axis = -y_axis

    rmse = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    obb = Obb(center, R, np.array([l, w, h], dtype=np.float32), n,
              plane_ratio=float(ratio), rmse=rmse,
              n_points=len(inliers), source='plane')
    if mask_quad_pts is not None and K is not None:
        got = _obb_from_quad(obb, mask_quad_pts, K, h, margin)
        if got is not None:
            x_axis, y_axis, l, w, face_center = got
            R = np.stack([x_axis, y_axis, n], axis=1).astype(np.float32)
            if np.linalg.det(R) < 0:
                R[:, 1] *= -1
                y_axis = -y_axis
                R = np.stack([x_axis, y_axis, n], axis=1).astype(np.float32)
            obb = Obb(face_center - n * (h / 2.0), R,
                      np.array([l, w, h], dtype=np.float32), n,
                      plane_ratio=float(ratio), rmse=rmse,
                      n_points=len(inliers), source='quad')
            return contain_points(obb, points)
    return contain_points(obb, points)
