"""场景级点云处理：整帧反投影 → 过滤 → 去平面 → 远到近认领。

和"每个 mask 各自反投影"相比，这一版先把整帧变成一份点云，
再把每个物体从这份点云里**认领**走，认领过的点直接从场景删除。
这样后面的物体（更近的）不会拿到前面（更远）物体的表面点，
从源头消除"遥控器的点云里混进书面点"。
"""

import numpy as np

from .obb import ransac_plane_largest, voxel_downsample


def deproject_frame(depth_mm, fx, fy, cx, cy, stride=2,
                    min_depth=0.5, max_depth=1.3):
    """整帧反投影。返回 (pts[N,3] float32, uv[N,2] int32, depth_valid[N] bool)。

    uv 是像素坐标（col, row），用于之后和 mask 对齐。
    """
    depth = np.asarray(depth_mm)
    h, w = depth.shape[:2]
    if stride > 1:
        depth = depth[::stride, ::stride]
        hh, ww = depth.shape
    else:
        hh, ww = h, w
    z = depth.astype(np.float32) / 1000.0
    ok = (z > min_depth) & (z < max_depth)
    if not np.any(ok):
        return None, None, None
    rows, cols = np.nonzero(ok)
    zz = z[rows, cols]
    uu = cols.astype(np.float32) * stride
    vv = rows.astype(np.float32) * stride
    x = (uu - cx) * zz / fx
    y = (vv - cy) * zz / fy
    pts = np.stack([x, y, zz], axis=1).astype(np.float32)
    uv = np.stack([uu.astype(np.int32), vv.astype(np.int32)], axis=1)
    return pts, uv, zz


def remove_big_planes(pts, n_planes=2, thresh=0.006, iters=80,
                      min_ratio=0.20, up_prior=0.45,
                      ref_n=None, ref_d=None, max_angle_deg=10.0,
                      max_offset=0.03):
    """迭代删除大平面（桌面、背景板）。

    返回 (keep_mask, planes, plane_pts_list)：
      keep_mask    —— 非平面点
      planes       —— [(n, d, 内点数), ...]
      plane_pts    —— 每个平面被删掉的点（用于 RViz 显示桌面）
    """
    keep = np.ones(len(pts), dtype=bool)     # 场景里保留的点（真桌子才会被删）
    search = np.ones(len(pts), dtype=bool)   # RANSAC 的搜索范围（可排除假平面）
    planes = []
    plane_pts = []
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(n_planes + 2):          # 允许跳过几个"假桌面"（比如键盘/书顶面）
        if len(planes) >= n_planes:
            break
        idx = np.nonzero(search)[0]
        if len(idx) < 500:
            break
        sub = pts[idx]
        # RANSAC 只在大范围降采样点上跑，阈值不变（保持严格）
        if len(sub) > 1500:
            sub_s = voxel_downsample(sub, 0.010)
            if len(sub_s) >= 300:
                sub = sub_s
        pl = ransac_plane_largest(sub, thresh=thresh, iters=iters,
                                  min_ratio=min_ratio)
        if pl is None:
            break
        n, d, _r = pl
        n = np.asarray(n, dtype=np.float32)
        if float(n @ up) < 0:
            n, d = -n, -float(d)
        if abs(float(n @ up)) < up_prior:
            break
        dist = pts @ n + float(d)
        # 与上一帧桌面一致性检查：法线角度、平面偏移都不能跳太多，
        # 否则会把"键盘顶面/书顶面"当成桌面，把那块点从场景里删掉。
        if ref_n is not None:
            ang = float(np.degrees(np.arccos(np.clip(abs(float(n @ ref_n)), -1.0, 1.0))))
            if ang > max_angle_deg:
                search &= ~(np.abs(dist) < thresh)
                continue
            if ref_d is not None and abs(float(d) - float(ref_d)) > max_offset:
                search &= ~(np.abs(dist) < thresh)
                continue
        # 真桌面：上面必须站着东西（>15% 的点在它上方），否则可能是"键盘/书的顶面"
        above = float((search & (dist > 0.010)).mean())
        if above < 0.15:
            # 只从"搜索范围"里排除这块平面，**不要**把场景里这些点删掉
            search &= ~(np.abs(dist) < thresh)
            continue
        on = search & (np.abs(dist) < thresh)
        if on.sum() < 300:
            break
        keep &= ~on
        planes.append((n, float(d), int(on.sum())))
        plane_pts.append(pts[on])
    return keep, planes, plane_pts


def trim_by_height_gap(h, base, gap=0.020, cap=0.25):
    """沿高度方向找"最大断崖"，把断崖以上的点当背景裁掉。

    物体本体（桌面往上）是连续的一层；背景/其他物体通常和它之间有明显空隙。
    返回布尔掩码（True = 保留）。
    """
    h = np.asarray(h, dtype=np.float64)
    keep = (h >= base - 0.02) & (h <= base + cap)
    idx = np.nonzero(keep)[0]
    if len(idx) < 20:
        return keep
    hs = h[idx]
    order = np.argsort(hs)
    hs_s = hs[order]
    d = np.diff(hs_s)
    if len(d) >= 3:
        j = int(np.argmax(d))
        if d[j] > gap and j >= len(d) * 0.5:
            lim = float(hs_s[j])
            if lim > base + 0.01:
                keep &= (h <= lim)
    return keep


def trim_inplane_outliers(pts, n, base, iters=2, k=3.0, min_r=0.05):
    """在平面内做鲁棒裁剪：以中位点为心，去掉离群远的点（背景碎片）。"""
    P = np.asarray(pts, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64); n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = P @ np.stack([u, v], axis=1)
    keep = np.ones(len(P), dtype=bool)
    for _ in range(iters):
        cur = uv[keep]
        if len(cur) < 20:
            break
        c = np.median(cur, axis=0)
        d = np.linalg.norm(cur - c, axis=1)
        r = max(k * float(np.median(d)), min_r)
        new_keep = np.zeros(len(P), dtype=bool)
        idx = np.nonzero(keep)[0]
        new_keep[idx[d <= r]] = True
        if new_keep.sum() < 20:
            break
        keep = new_keep
    return keep


def trim_depth_outliers(pts, k=3.0, min_band=0.030):
    """剔除深度飞点：和本实例中位深度差太远的点（反光/边缘常出这种点）。"""
    P = np.asarray(pts, dtype=np.float32)
    if len(P) < 20:
        return np.ones(len(P), dtype=bool)
    z = P[:, 2].astype(np.float64)
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med))) * 1.4826
    band = max(k * mad, min_band)
    return np.abs(z - med) <= band


def mask_uv_index(mask, uv, h=240, w=320):
    """把 (u,v) 像素坐标映射到 mask 的布尔值。"""
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    return mask[v, u]


def depth_order(dets, pts, uv, min_pts=40):
    """按 mask 内中位深度从远到近排序，返回 (order, masks, depth_of)。"""
    masks = {k: np.asarray(dets[k].mask, dtype=bool) for k in range(len(dets))}
    depth_of = {}
    for k in range(len(dets)):
        sel = mask_uv_index(masks[k], uv)
        depth_of[k] = float(np.median(pts[sel, 2])) if sel.sum() >= min_pts else -1.0
    order = sorted(range(len(dets)), key=lambda k: -depth_of[k])
    return order, masks, depth_of


def claim_mask(masks, depth_of, k, uv, alive, eps=0.005):
    """第 k 个物体的有效 mask：自己的 mask 减掉所有更近物体的 mask。"""
    m = masks[k].copy()
    dk = depth_of.get(k, -1.0)
    for j, dj in depth_of.items():
        if j == k or dj <= 0 or dk <= 0:
            continue
        if dj < dk - eps:
            m &= ~masks[j]
    return m


def claim_in_order(pts, uv, dets, n_table, d_table,
                   slab=0.015, min_pts=40, protect_first=True):
    """按"mask 内点中位深度从远到近"的顺序认领场景点。

    返回 list，与 dets 同序，每项 (mask_index, claimed_pts, claimed_uv, claimed_cols)。
    """
    n_t = np.asarray(n_table, dtype=np.float32)
    d_t = float(d_table)
    h_all = pts @ n_t + d_t

    # 每个检测的"代表深度"：mask 内有效点的中位深度（比 mask 中心 depth 稳）
    keys = []
    for k, det in enumerate(dets):
        sel = mask_uv_index(det.mask, uv)
        if sel.sum() >= min_pts:
            keys.append((float(np.median(pts[sel, 2])), k))
        else:
            keys.append((-1.0, k))
    order = [k for _, k in sorted(keys, key=lambda x: -x[0])]

    alive = np.ones(len(pts), dtype=bool)
    out = []
    first_done = False
    # 每个检测的"代表深度"（mask 内中位深度），用于判断谁在谁前面
    depth_of = {k: keys[k][0] for k in range(len(dets))}
    masks = {k: np.asarray(dets[k].mask, dtype=bool) for k in range(len(dets))}
    cut = 0.10     # 自身顶面以上 10cm 外的点视为背景
    for k in order:
        det = dets[k]
        # 近处物体的 mask 从当前（更远）物体的 mask 里减掉：
        # 这样书不会把"压在它上面的遥控器"那块像素算进自己。
        m_k = masks[k].copy()
        for j, dj in depth_of.items():
            if j == k or dj <= 0:
                continue
            if dj < depth_of.get(k, 1e9) - 0.005:      # j 更近
                m_k &= ~masks[j]
        sel = mask_uv_index(m_k, uv) & alive
        if sel.sum() >= min_pts:
            # 第一个（最远的）物体额外用 slab 约束：
            # 防止它的 mask 覆盖到前面物体身上、把前面的点抢走。
            hk = h_all[sel]
            span = float(hk.max() - hk.min())
            if span > 0.02:
                bins = max(int(span / 0.004), 4)
                hist, edges = np.histogram(hk, bins=bins)
                j = int(np.argmax(hist))
                mode_h = 0.5 * (edges[j] + edges[j + 1])
                # 单边裁剪：保留"不高于该物体顶面 + cut"的点，
                # 既拿走它自己的整个厚度，又不会把上面的物体/背景抢走。
                lim = mode_h + (slab if (protect_first and not first_done) else cut)
                keep = hk < lim
                if keep.sum() >= min_pts:
                    idx = np.nonzero(sel)[0]
                    sel_idx = idx[keep]
                    sel = np.zeros_like(sel)
                    sel[sel_idx] = True
            alive &= ~sel
            out.append((k, sel))
            first_done = True
        else:
            out.append((k, None))
    return out


def fit_slab_obb(obb_mod, points, n, base_h, margin=0.002):
    """用"支撑面法线 + 底面高度"直接构造 OBB（不再让点云自己猜 z）。"""
    if points is None or len(points) < 40:
        return None
    return obb_mod.fit_obb_on_plane(points, n, -float(base_h),
                                    margin=margin, band=0.004, min_frac=0.0)
