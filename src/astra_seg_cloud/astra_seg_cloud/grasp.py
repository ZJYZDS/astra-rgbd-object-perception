"""抓取位姿生成：吸盘 / 平行夹爪，自动选最优。

坐标系约定（夹爪本体）：
    z 轴 = 接近方向，指向物体内部
    x 轴 = 闭合方向（平行夹爪的两指沿此轴收拢；吸盘时无意义）
    y 轴 = z × x
    position = 夹爪中心，停在物体表面外侧 standoff 处

所有几何量都在相机光学系（米）。
"""

from dataclasses import dataclass

import numpy as np

GRASP_NONE = 0
GRASP_SUCTION = 1
GRASP_PARALLEL = 2

GRASP_NAMES = {GRASP_NONE: 'none', GRASP_SUCTION: 'suction', GRASP_PARALLEL: 'parallel'}


@dataclass
class GripperConfig:
    suction_radius: float = 0.03     # 吸盘半径（米）
    jaw_max_open: float = 0.08       # 夹爪最大开口（米）
    jaw_min_open: float = 0.00
    jaw_width: float = 0.04          # 手指宽度（米）
    standoff: float = 0.02           # 抓取点距表面的退让距离


@dataclass
class Grasp:
    kind: int
    position: np.ndarray
    rotation: np.ndarray
    width: float
    score: float
    mode: str

    @property
    def name(self):
        return GRASP_NAMES.get(self.kind, 'unknown')


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _make_frame(z_approach, x_close, contact, standoff, kind, width, score, mode):
    """由接近方向与闭合方向构造抓取位姿。"""
    z = _norm(z_approach)
    x = x_close - z * float(np.dot(x_close, z))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x = _norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    position = contact - z * standoff
    return Grasp(kind, position.astype(np.float32), R.astype(np.float32),
                 float(width), float(score), mode)


def _width_score(width, cfg):
    span = max(cfg.jaw_max_open - cfg.jaw_min_open, 1e-6)
    mid = 0.5 * (cfg.jaw_max_open + cfg.jaw_min_open)
    return float(np.clip(1.0 - abs(width - mid) / (0.5 * span), 0.0, 1.0))


def evaluate(obb, cfg: GripperConfig):
    """枚举候选抓取，返回得分最高的一个。"""
    l, w, h = (float(v) for v in obb.dims)
    n = np.asarray(obb.normal, dtype=np.float64)          # 朝相机
    ax = np.asarray(obb.rotation[:, 0], dtype=np.float64)  # 长轴
    ay = np.asarray(obb.rotation[:, 1], dtype=np.float64)
    center = np.asarray(obb.center, dtype=np.float64)
    if np.linalg.det(np.stack([ax, ay, n], axis=1)) < 0:
        ay = -ay

    top_center = center + n * (h / 2.0)
    cands = []

    # ---------- 1. 吸盘：需要半径 r 的圆盘落在可见面内 ----------
    r = cfg.suction_radius
    if min(l, w) * 0.5 >= r:
        margin = float(np.clip((min(l, w) * 0.5 - r) / max(r, 1e-6), 0, 1))
        score = 0.55 * obb.plane_ratio + 0.45 * margin
        cands.append(_make_frame(-n, ax, top_center, cfg.standoff,
                                 GRASP_SUCTION, 2 * r, score, 'suction'))

    # ---------- 2. 平行夹爪：俯视，横跨较短边 ----------
    width_top = min(l, w)
    if cfg.jaw_min_open <= width_top <= cfg.jaw_max_open:
        close_axis = ay if w <= l else ax
        score = 0.6 * _width_score(width_top, cfg) + 0.4 * obb.plane_ratio
        cands.append(_make_frame(-n, close_axis, top_center, cfg.standoff,
                                 GRASP_PARALLEL, width_top, score, 'top'))

    # ---------- 3. 平行夹爪：侧向，沿长轴进入，夹厚度 ----------
    if cfg.jaw_min_open <= h <= cfg.jaw_max_open and w >= cfg.jaw_width:
        to_cam = -center
        sign = 1.0 if float(np.dot(ax, to_cam)) >= 0 else -1.0
        contact = center + ax * (sign * l / 2.0)
        score = (0.5 * _width_score(h, cfg)
                 + 0.3 * float(np.clip((w - cfg.jaw_width) / max(cfg.jaw_width, 1e-6), 0, 1))
                 + 0.2 * obb.plane_ratio)
        cands.append(_make_frame(-sign * ax, n, contact, cfg.standoff,
                                 GRASP_PARALLEL, h, score, 'side_x'))

    # ---------- 4. 平行夹爪：侧向，沿短轴进入，夹厚度 ----------
    if cfg.jaw_min_open <= h <= cfg.jaw_max_open and l >= cfg.jaw_width:
        to_cam = -center
        sign = 1.0 if float(np.dot(ay, to_cam)) >= 0 else -1.0
        contact = center + ay * (sign * w / 2.0)
        score = (0.5 * _width_score(h, cfg)
                 + 0.3 * float(np.clip((l - cfg.jaw_width) / max(cfg.jaw_width, 1e-6), 0, 1))
                 + 0.2 * obb.plane_ratio)
        cands.append(_make_frame(-sign * ay, n, contact, cfg.standoff,
                                 GRASP_PARALLEL, h, score, 'side_y'))

    if not cands:
        return Grasp(GRASP_NONE, center.astype(np.float32),
                     np.eye(3, dtype=np.float32), 0.0, 0.0, 'none')

    best = max(cands, key=lambda g: g.score)
    return best
