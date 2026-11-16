"""按跟踪 ID 对 OBB 位姿做时序平滑。

逐帧独立估计会带来 2~7° 的平面内抖动，看着像"框在歪"。
这里对旋转做四元数 slerp、对中心/尺寸做一阶低通，同一 ID 连续帧共享状态。
"""

import numpy as np


def _quat_from_R(R):
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                      0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return q / (np.linalg.norm(q) + 1e-12)


def _R_from_quat(q):
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    d = np.clip(d, -1.0, 1.0)
    if d > 0.9995:
        return q0 + t * (q1 - q0)
    theta0 = np.arccos(d)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return s0 * q0 + s1 * q1


class PoseSmoother:
    def __init__(self, alpha=0.35, max_age=30,
                 dim_max_jump=0.35, dim_max_step=0.12,
                 center_max_step=0.05):
        """alpha 越小越平滑（0~1）。

        dim_max_jump : 单帧尺寸相对变化超过它 -> 判为异常测量，整帧丢弃（不跟）
        dim_max_step : 即使正常，每帧尺寸最多变化这个相对比例（防止视觉上的"缩放"）
        center_max_step : 每帧中心最多移动多少米
        """
        self.alpha = float(alpha)
        self.max_age = int(max_age)
        self.dim_max_jump = float(dim_max_jump)
        self.dim_max_step = float(dim_max_step)
        self.center_max_step = float(center_max_step)
        self._state = {}

    def update(self, track_id, obb):
        """返回平滑后的 (R, center, dims)；输入不合法时原样返回。"""
        if obb is None:
            return None

        R = np.asarray(obb.rotation, dtype=np.float64).copy()
        prev = self._state.get(track_id)

        # 规范化解：盒子的朝向绕法线有 180° 等价歧义（x/y 同时取反），
        # 先对齐到上一帧的 x 轴，否则 slerp 会绕远路，看起来就是"框在歪"。
        if prev is not None and float(np.dot(R[:, 0], prev['x'])) < 0.0:
            R[:, 0] *= -1.0
            R[:, 1] *= -1.0

        q = _quat_from_R(R)
        c = np.asarray(obb.center, dtype=np.float64)
        d = np.asarray(obb.dims, dtype=np.float64)

        if prev is None:
            self._state[track_id] = {'q': q, 'c': c, 'd': d, 'x': R[:, 0], 'age': 0}
            return obb

        a = self.alpha
        q_s = _slerp(prev['q'], q, a)
        q_s = q_s / (np.linalg.norm(q_s) + 1e-12)

        # 尺寸：只做异常值拒绝（单帧变化过大时丢弃该测量），不做限速
        prev_d = np.asarray(prev['d'], dtype=np.float64)
        rel = np.abs(d - prev_d) / np.maximum(prev_d, 1e-6)
        if float(np.max(rel)) > self.dim_max_jump:
            d_s = prev_d.copy()
        else:
            d_s = (1 - a) * prev_d + a * d

        # 中心：普通低通（不限速）
        c_s = (1 - a) * prev['c'] + a * c
        R_s = _R_from_quat(q_s)
        self._state[track_id] = {'q': q_s, 'c': c_s, 'd': d_s,
                                 'x': R_s[:, 0].astype(np.float64), 'age': 0}

        obb.rotation = R_s
        obb.center = c_s.astype(np.float32)
        obb.dims = d_s.astype(np.float32)
        obb.normal = obb.rotation[:, 2].copy()
        return obb

    def prune(self, alive_ids):
        for k in list(self._state):
            if k not in alive_ids:
                self._state[k]['age'] += 1
                if self._state[k]['age'] > self.max_age:
                    del self._state[k]
