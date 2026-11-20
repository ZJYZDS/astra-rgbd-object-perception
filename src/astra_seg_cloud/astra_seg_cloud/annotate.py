"""把 OBB / 抓取位姿画到 RGB 图上。"""

import cv2
import numpy as np

# OBB 12 条棱，基于 Obb.corners() 的角点顺序
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def project(points, fx, fy, cx, cy):
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    ok = z > 1e-3
    u = np.full(len(z), -1, dtype=np.int32)
    v = np.full(len(z), -1, dtype=np.int32)
    u[ok] = np.round(fx * pts[ok, 0] / z[ok] + cx).astype(np.int32)
    v[ok] = np.round(fy * pts[ok, 1] / z[ok] + cy).astype(np.int32)
    return np.stack([u, v], axis=1), ok


def draw_mask_overlay(img, mask, color, alpha=0.35, contour=True):
    """把实例 mask 半透明叠在 RGB 上（用实例颜色），并用 1px 描边。"""
    if mask is None or not mask.any():
        return img
    m = mask.astype(bool)
    col = np.array(color, dtype=np.float32)
    region = img[m].astype(np.float32)
    img[m] = (region * (1.0 - alpha) + col * alpha).astype(np.uint8)
    if contour:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, tuple(int(v) for v in color), 1, cv2.LINE_AA)
    return img


def draw_obb(img, obb, fx, fy, cx, cy, color=(0, 255, 255), thickness=1):
    corners = obb.corners()
    uv, ok = project(corners, fx, fy, cx, cy)
    h, w = img.shape[:2]
    for a, b in EDGES:
        if not (ok[a] and ok[b]):
            continue
        pa = tuple(np.clip(uv[a], [-10, -10], [w + 10, h + 10]))
        pb = tuple(np.clip(uv[b], [-10, -10], [w + 10, h + 10]))
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    return img


def points_pixel_mask(pts, fx, fy, cx, cy, shape, block=3):
    """把 3D 点投影成像素掩码（"3D 点覆盖了哪些像素"）。

    stride 采样后点云是稀疏网格，所以每个点按 block×block 铺开，再闭运算补缝，
    得到的就是"这份 3D 点云在图像上的覆盖面"。
    """
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    P = np.asarray(pts, dtype=np.float64)
    if len(P) == 0:
        return m
    z = P[:, 2]
    ok = z > 1e-3
    if not ok.any():
        return m
    u = (fx * P[ok, 0] / z[ok] + cx).round().astype(np.int32)
    v = (fy * P[ok, 1] / z[ok] + cy).round().astype(np.int32)
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m[v[keep], u[keep]] = 255
    if block > 1:
        m = cv2.dilate(m, np.ones((block, block), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def blend_mask(img, mask, color, alpha=0.28):
    if mask is None or not mask.any():
        return img
    ov = img.copy()
    ov[mask > 0] = color
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)
    return img


def draw_obb_outline(img, obb, fx, fy, cx, cy, color=(0, 255, 255), thickness=1):
    """只画框的投影轮廓（细边，不填充）。"""
    corners = obb.corners()
    uv, ok = project(corners, fx, fy, cx, cy)
    if not ok.all():
        return img
    h, w = img.shape[:2]
    pts = np.clip(uv, [0, 0], [w - 1, h - 1]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.polylines(img, [hull], True, color, thickness, cv2.LINE_AA)
    return img


def draw_obb_mask(img, obb, fx, fy, cx, cy, color=(0, 255, 255), alpha=0.22,
                  outline=True):
    """把 3D 框的投影轮廓画成**浅色面片**（代替线框）。

    RGB 上用线框判断 3D 框好不好本身就不可靠（透视下前后面重叠、棱边位置随视角变），
    改成"这个框在图像上覆盖了哪块面积"，配合细轴看朝向，判读更直接。
    """
    corners = obb.corners()
    uv, ok = project(corners, fx, fy, cx, cy)
    if not ok.all():
        return img
    h, w = img.shape[:2]
    pts = np.clip(uv, [0, 0], [w - 1, h - 1]).astype(np.int32)
    hull = cv2.convexHull(pts)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    if outline:
        cv2.polylines(img, [hull], True, color, 1, cv2.LINE_AA)
    return img


def draw_grasp(img, grasp, fx, fy, cx, cy, color=(0, 0, 255)):
    h, w = img.shape[:2]
    pos = np.asarray(grasp.position, dtype=np.float64)
    if pos[2] <= 1e-3:
        return img

    # 抓取中心
    uv, ok = project([pos], fx, fy, cx, cy)
    if not ok[0]:
        return img
    p = (int(uv[0, 0]), int(uv[0, 1]))
    cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)

    # 接近方向箭头（z 轴，指向物体内部）
    z = grasp.rotation[:, 2]
    tip = pos + z * 0.04
    uv2, ok2 = project([tip], fx, fy, cx, cy)
    if ok2[0]:
        cv2.arrowedLine(img, p, (int(uv2[0, 0]), int(uv2[0, 1])),
                        color, 1, cv2.LINE_AA, tipLength=0.4)

    # 平行夹爪：画出两指与开口宽度
    if grasp.kind == 2:
        x = grasp.rotation[:, 0]
        half = grasp.width / 2.0
        a = pos + x * half
        b = pos - x * half
        uva, oka = project([a], fx, fy, cx, cy)
        uvb, okb = project([b], fx, fy, cx, cy)
        if oka[0] and okb[0]:
            cv2.line(img, (int(uva[0, 0]), int(uva[0, 1])),
                     (int(uvb[0, 0]), int(uvb[0, 1])), color, 1, cv2.LINE_AA)
            cv2.circle(img, (int(uva[0, 0]), int(uva[0, 1])), 2, color, -1)
            cv2.circle(img, (int(uvb[0, 0]), int(uvb[0, 1])), 2, color, -1)
    return img


def draw_label(img, text, org, color, scale=0.34):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)
    return img


def draw_axes_plain(img, obb, fx, fy, cx, cy, length=0.05, thickness=1):
    """把实例自身三轴画成**不带箭头**的线段（红=x, 绿=y, 蓝=z）。"""
    R = np.asarray(obb.rotation, dtype=np.float64)
    c = np.asarray(obb.center, dtype=np.float64)
    colors = ([(0, 0, 255), (0, 255, 0), (255, 0, 0)])
    h, w = img.shape[:2]
    for k, name in enumerate('xyz'):
        tip = c + R[:, k] * length
        uv, ok = project(np.array([c, tip]), fx, fy, cx, cy)
        if not ok.all():
            continue
        p0 = tuple(np.int32(np.clip(uv[0], [0, 0], [w - 1, h - 1])))
        p1 = tuple(np.int32(np.clip(uv[1], [0, 0], [w - 1, h - 1])))
        cv2.line(img, p0, p1, colors[k], thickness, cv2.LINE_AA)
        cv2.putText(img, name, (int(p1[0]) + 2, int(p1[1]) + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, colors[k], 1, cv2.LINE_AA)
    return img


def caption_strip(lines, width, line_h=12, pad=3, colors=None, min_lines=0):
    """在图下方拼一条黑底说明条（不遮挡画面）。

    colors: 可选，每行的实例颜色（BGR），会在文字前画一个小色块，
            方便把文字和 RViz 里框的颜色对上。
    """
    lines = list(lines)
    # 固定行数：物体数量变化时不改变整张图的高度（否则显示上会"跳"）
    while len(lines) < max(min_lines, 1):
        lines.append('')
    h = line_h * max(len(lines), 1) + 2 * pad
    strip = np.zeros((h, width, 3), np.uint8)
    for i, text in enumerate(lines):
        x0 = 4
        if colors is not None and i < len(colors) and colors[i] is not None:
            c = tuple(int(v) for v in colors[i])
            cv2.rectangle(strip, (x0, pad + line_h * i + 1),
                          (x0 + 7, pad + line_h * i + line_h - 2), c, -1)
            x0 += 10
        cv2.putText(strip, text, (x0, pad + line_h * (i + 1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (220, 220, 220), 1, cv2.LINE_AA)
    return strip
