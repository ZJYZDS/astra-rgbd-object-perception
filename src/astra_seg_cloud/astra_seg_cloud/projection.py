"""mask 腐蚀 + 深度反投影。

depth_align: true 时深度图已经硬件对齐到彩色图，两路共用同一套内参，
所以直接用 (u, v, depth) 反投影即可，无需再做像素坐标映射。
"""

import cv2
import numpy as np


def erode_mask(mask, kernel=3):
    """腐蚀 mask 边缘，滤掉分割边界上混入的背景像素/深度飞点。

    如果腐蚀后 mask 变空（细长物体），退回原始 mask，避免整条目标丢失。
    """
    if kernel is None or kernel <= 1:
        return mask
    k = np.ones((kernel, kernel), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)
    if not eroded.any():
        return mask
    return eroded


def deproject(mask, depth_mm, rgb_bgr, fx, fy, cx, cy,
              min_depth=0.15, max_depth=6.0, stride=1, min_points=30):
    """把 mask 内的像素按内参反投影成相机光学系下的 3D 点。

    返回 (points[N,3] float32 米, colors[N,3] uint8 BGR)，点数不足时返回 (None, None)。
    """
    h, w = depth_mm.shape[:2]

    if stride > 1:
        vs = np.arange(0, h, stride)
        us = np.arange(0, w, stride)
        sub = mask[np.ix_(vs, us)]
        vi, ui = np.nonzero(sub)
        v = vs[vi]
        u = us[ui]
    else:
        v, u = np.nonzero(mask)

    if u.size == 0:
        return None, None

    z = depth_mm[v, u].astype(np.float32) * 0.001  # mm -> m
    valid = (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return None, None

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack((x, y, z), axis=1).astype(np.float32)
    if pts.shape[0] < min_points:
        return None, None

    colors = rgb_bgr[v.astype(np.int32), u.astype(np.int32)]
    return pts, colors
