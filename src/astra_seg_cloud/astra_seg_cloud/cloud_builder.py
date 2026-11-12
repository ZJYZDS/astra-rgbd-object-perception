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


def obb_edge_points(obb, step=0.012):
    """把 OBB 的 12 条棱离散成点（用于 RViz 里"看到"3D 框）。"""
    c = obb.corners()
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = []
    for a, b in edges:
        pa, pb = c[a], c[b]
        n = max(int(np.linalg.norm(pb - pa) / step), 2)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(pa[None, :] * (1 - t) + pb[None, :] * t)
    return np.vstack(pts).astype(np.float32)
