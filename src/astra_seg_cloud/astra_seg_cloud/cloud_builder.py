"""构造 sensor_msgs/PointCloud2：字段 x/y/z/rgb（rgb 为 float32 打包的 0x00RRGGBB）。"""

import array

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

_POINT_STEP = 16

_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
]


def build_cloud(header, points, colors_bgr):
    """points: (N,3) float32 米；colors_bgr: (N,3) uint8。"""
    n = int(points.shape[0])
    buf = np.empty((n, 4), dtype=np.float32)
    buf[:, :3] = points

    packed = (colors_bgr[:, 2].astype(np.uint32) << 16
              | colors_bgr[:, 1].astype(np.uint32) << 8
              | colors_bgr[:, 0].astype(np.uint32))
    buf[:, 3] = packed.view(np.float32)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = _FIELDS
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * n
    msg.is_dense = True
    # 注意：不能直接赋 bytes —— rclpy 会逐元素转成 uint8[]，3 万点要 30ms 以上。
    # array.array('B', ...) 走的是连续内存，几乎零开销。
    msg.data = array.array('B', buf.tobytes())
    return msg


def obb_edge_points(obb, step=0.004, max_per_edge=90):
    """把 OBB 的 12 条棱离散成点（RViz 里"看到"3D 框）。

    step: 点间距（米），默认 4mm → 框看起来是连续线。
    长边会按 max_per_edge 自动放宽，避免大物体点数爆炸。
    """
    c = obb.corners()
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = []
    for a, b in edges:
        pa, pb = c[a], c[b]
        L = float(np.linalg.norm(pb - pa))
        n = int(np.clip(round(L / step), 4, max_per_edge))
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(pa[None, :] * (1 - t) + pb[None, :] * t)
    return np.vstack(pts).astype(np.float32)


def instance_color(idx):
    """按实例序号给一个高对比、互不相同的颜色（BGR），黄金角散开色相。"""
    h = (float(idx) * 0.61803398875) % 1.0
    s, v = 1.0, 1.0
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p_, q_, t_ = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = [(v, t_, p_), (q_, v, p_), (p_, v, t_), (p_, q_, v), (t_, p_, v), (v, p_, q_)][i]
    r, g, b = [int(round(255 * x)) for x in rgb]
    return np.array([b, g, r], dtype=np.uint8)   # BGR
