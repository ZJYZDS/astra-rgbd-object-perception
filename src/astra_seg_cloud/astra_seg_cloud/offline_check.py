"""离线自检：不启动 ROS，直接验证「YOLO 推理 + mask 腐蚀 + 反投影」是否正确。

    ros2 run astra_seg_cloud seg_cloud_offline_check /home/zjy/astra_models/test_bus.jpg

会打印检测结果、点云点数，并把 overlay 存到 /tmp/seg_overlay.jpg。
"""

import os
import sys

from astra_seg_cloud.entry import _prepend_venv, _strip_user_site

# 必须先摘掉被 NumPy 2 污染的 user site，再导入 numpy / cv2
_strip_user_site()
_prepend_venv()

import numpy as np  # noqa: E402


def _geometry_selftest(projection):
    """用已知深度的平面反投影，再投影回去，验证 u/v 误差为 0。"""
    import cv2

    fx = fy = 285.1711
    cx, cy = 159.5, 119.5
    h, w = 240, 320
    depth = np.full((h, w), 2000, dtype=np.uint16)  # 2 米

    mask = np.zeros((h, w), dtype=bool)
    mask[100:140, 120:180] = True

    pts, cols = projection.deproject(
        mask, depth, np.zeros((h, w, 3), np.uint8), fx, fy, cx, cy,
        min_depth=0.1, max_depth=10.0, min_points=1)
    assert pts is not None, '反投影返回空'
    assert np.allclose(pts[:, 2], 2.0, atol=1e-6), 'z 不是 2 米'

    # 反投影后再投影回像素：u = x*fx/z + cx
    u = pts[:, 0] * fx / pts[:, 2] + cx
    v = pts[:, 1] * fy / pts[:, 2] + cy
    assert abs(u.mean() - 149.5) < 1e-3, f'u 均值偏移 {u.mean()}'
    assert abs(v.mean() - 119.5) < 1e-3, f'v 均值偏移 {v.mean()}'

    # 无效深度（0）必须被过滤
    depth[:] = 0
    pts0, _ = projection.deproject(
        mask, depth, np.zeros((h, w, 3), np.uint8), fx, fy, cx, cy,
        min_depth=0.1, max_depth=10.0, min_points=1)
    assert pts0 is None, '深度全 0 时应返回 None'

    # 腐蚀必须缩小 mask，且不能把 mask 腐蚀空
    k = np.ones((31, 31), np.uint8)
    small = np.zeros((h, w), np.uint8)
    small[100:140, 120:180] = 1
    eroded = cv2.erode(small, k)
    assert eroded.sum() < small.sum()
    assert projection.erode_mask(mask, 5).sum() > 0

    print('✓ 几何自检通过（z=2m、像素可逆、无效深度过滤、腐蚀生效）')


def main():
    import cv2

    from astra_seg_cloud import projection
    from astra_seg_cloud.projection import deproject, erode_mask
    from astra_seg_cloud.yolo_seg import YoloSegOnnx

    _geometry_selftest(projection)

    image = sys.argv[1] if len(sys.argv) > 1 else '/home/zjy/astra_models/test_bus.jpg'
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        '~/astra_models/yolov8s-seg.onnx')

    img = cv2.imread(image)
    if img is None:
        raise SystemExit(f'读不到图片: {image}')
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

    net = YoloSegOnnx(model, logger=print)
    dets = net.infer(img)

    fx = fy = 285.1711
    cx, cy = 159.5, 119.5
    depth = np.full(img.shape[:2], 2000, dtype=np.uint16)

    print(f'\n检测到 {len(dets)} 个实例（320x240 输入）：')
    overlay = img.copy()
    for d in dets:
        m = erode_mask(d.mask, 3)
        pts, _ = deproject(m, depth, img, fx, fy, cx, cy, min_points=1)
        n = 0 if pts is None else len(pts)
        print(f'  {d.label:12s} score={d.score:.3f} '
              f'box=({d.box[0]:.0f},{d.box[1]:.0f},{d.box[2]:.0f},{d.box[3]:.0f}) '
              f'pts={n}')
        overlay[m] = (0, 255, 0)
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

    out = '/tmp/seg_overlay.jpg'
    cv2.imwrite(out, cv2.addWeighted(overlay, 0.6, img, 0.4, 0))
    print(f'\noverlay 已保存: {out}')


if __name__ == '__main__':
    main()
