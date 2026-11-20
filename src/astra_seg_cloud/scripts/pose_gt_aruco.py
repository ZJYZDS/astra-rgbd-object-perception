#!/usr/bin/env python3
"""方案 B：用 ArUco 提供物体实时位姿 GT，并与 /astra_seg/objects 逐帧比对。

原理：
    T_cam_obj = T_cam_marker · T_marker_obj
    T_cam_marker 由 solvePnP(IPPE_SQUARE) 从 4 个角点解出；
    T_marker_obj 是"marker 贴在物体哪个位置"的固定变换，量一次写进 yaml。

用法：
    python3 scripts/pose_gt_aruco.py --config config/gt_markers.yaml
    python3 scripts/pose_gt_aruco.py --selftest          # 合成图自测，不需要打印机
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

DEFAULT_K = (285.1711, 285.1711, 159.5, 119.5)


# ---------------------------------------------------------------- 几何小工具
def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


UP = np.array([0.0, -1.0, 0.0])          # 相机 y 向下，桌面朝上 = -y
_E1 = np.array([1.0, 0.0, 0.0])          # 平面内基 1（相机 x）
_E2 = np.cross(UP, _E1)                  # 平面内基 2


def yaw_up(R):
    """绕桌面法线的 yaw（度）。不依赖 box 的 z 正负/xy 命名，避免口径混用。"""
    x = np.asarray(R, dtype=np.float64)[:, 0]
    x = x - UP * float(x @ UP)
    n = np.linalg.norm(x)
    if n < 1e-9:
        return 0.0
    x /= n
    return math.degrees(math.atan2(float(x @ _E2), float(x @ _E1)))


def rot_err_deg(Ra, Rb):
    R = Ra @ Rb.T
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def yaw_of(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


# ---------------------------------------------------------------- marker 检测
def make_detector(dict_name='DICT_4X4_50'):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        return cv2.aruco.ArucoDetector(dic, params)
    params = cv2.aruco.DetectorParameters_create()
    return ('legacy', dic, params)


def detect(detector, gray):
    if isinstance(detector, tuple):
        _, dic, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.flatten(), corners)]


def solve_planar(uv, obj, R_base, K=DEFAULT_K, p0=(0.0, 0.0, 0.9, 0.0), iters=30):
    """已知 marker 面朝向基 R_base（法线已知），只解 yaw + 平移（4 自由度）。

    平面标记用普通 6DoF PnP 会有"镜像解"二义性：角点噪声 0.3 px 时旋转误差能跳到
    25-30°。把"贴在同一平面"这个物理约束加进去就稳了（实测 ~0.5°）。
    用数值雅可比的 Gauss-Newton，不依赖 scipy。
    """
    fx, fy = K[0], K[1]
    p = np.array(p0, dtype=np.float64)

    def project(p):
        R = R_base @ rpy_to_R(0, 0, p[3])
        cam = (R @ obj.T).T + p[:3]
        return np.stack([fx * cam[:, 0] / cam[:, 2] + K[2],
                         fy * cam[:, 1] / cam[:, 2] + K[3]], axis=1)

    for _ in range(iters):
        r = (project(p) - uv).ravel()
        J = np.zeros((len(r), 4))
        eps = 1e-5
        for k in range(4):
            q = p.copy(); q[k] += eps
            J[:, k] = ((project(q) - uv).ravel() - r) / eps
        dp, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + 0.7 * dp
        p[2] = min(max(p[2], 0.2), 2.0)
        if np.linalg.norm(dp) < 1e-7:
            break
    return p


def pose_of_marker(img_pts, marker_size, K):
    """marker 系（中心为原点，z 朝外，x 右 y 下）到相机系的位姿。"""
    s = marker_size / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float32), img_pts.astype(np.float32),
                                  K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


# ---------------------------------------------------------------- 自测
def selftest(marker_size=0.10, K=DEFAULT_K, dict_name='DICT_4X4_50'):
    """把 marker 按已知位姿"打印"到合成图里，再解回来，验证误差量级。"""
    fx, fy, cx, cy = K
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    tex = cv2.aruco.generateImageMarker(dic, 0, 400) if hasattr(cv2.aruco, 'generateImageMarker') \
        else cv2.aruco.drawMarker(dic, 0, 400)
    tex = cv2.copyMakeBorder(tex, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)

    # 正面朝向相机的基（marker 的 +z 指向相机）+ 少量倾斜
    T_true = np.eye(4)
    T_true[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ rpy_to_R(0.15, -0.25, 0.35)
    T_true[:3, 3] = [0.03, -0.02, 0.90]
    s = marker_size / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    cam_pts = (T_true[:3, :3] @ obj.T).T + T_true[:3, 3]
    uv = np.stack([fx * cam_pts[:, 0] / cam_pts[:, 2] + cx,
                   fy * cam_pts[:, 1] / cam_pts[:, 2] + cy], axis=1)

    pad = 60
    tex_uv = np.array([[pad, pad], [pad + 400, pad], [pad + 400, pad + 400],
                       [pad, pad + 400]], np.float32)
    H, _ = cv2.findHomography(tex_uv, uv.astype(np.float32))
    canvas = np.full((240, 320), 200, np.uint8)
    warped = cv2.warpPerspective(tex, H, (320, 240), dst=canvas,
                                 borderMode=cv2.BORDER_TRANSPARENT)
    det = make_detector(dict_name)
    found = detect(det, warped)
    if not found:
        print('自测失败：合成图里没检测到 marker'); return False
    mid, pts = found[0]
    # 用"已知平面法线"的约束解（GT 工具默认就是这个模式）
    n_reg = T_true[:3, 2] / np.linalg.norm(T_true[:3, 2])
    u_ax = np.cross(n_reg, [1.0, 0.0, 0.0])
    if np.linalg.norm(u_ax) < 1e-6:
        u_ax = np.cross(n_reg, [0.0, 0.0, 1.0])
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(n_reg, u_ax)
    R_base = np.stack([u_ax, v_ax, n_reg], axis=1)
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    p4 = solve_planar(pts, obj, R_base)
    T = np.eye(4); T[:3, :3] = R_base @ rpy_to_R(0, 0, p4[3]); T[:3, 3] = p4[:3]
    dp = np.linalg.norm(T[:3, 3] - T_true[:3, 3])
    dr = rot_err_deg(T[:3, :3], T_true[:3, :3])
    print(f'自测: id={mid}  平移误差={dp*1000:.1f} mm  旋转误差={dr:.2f}°（约束解）')
    Tf = pose_of_marker(pts, marker_size, Km)
    if Tf is not None:
        print(f'      对照 6DoF: {np.linalg.norm(Tf[:3,3]-T_true[:3,3])*1000:.1f} mm / '
              f'{rot_err_deg(Tf[:3,:3], T_true[:3,:3]):.2f}°')
    cv2.imwrite('/tmp/aruco_selftest.png', cv2.resize(warped, None, fx=3, fy=3,
                                                       interpolation=cv2.INTER_NEAREST))
    # 合成图只有 ~30 px，重采样本身带 ~1 px 角点偏差 → 深度误差 ≈ 1/px。
    # 所以这里按"距离的 4%"判平移，旋转按 2°（旋转对角点偏差不敏感）。
    ok = dp < 0.04 * T_true[2, 3] and dr < 2.0
    print(f'自测 ' + ('PASS' if ok else 'FAIL') +
          f'（平移阈值 {0.04*T_true[2,3]*1000:.0f} mm，旋转 2°）')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/gt_markers.yaml')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--marker-size', type=float, default=0.10)
    ap.add_argument('--csv', default=os.path.expanduser('~/gt_aruco/gt_errors.csv'))
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(args.marker_size) else 1)

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    dict_name = cfg.get('dict', 'DICT_4X4_50')
    marker_size = float(cfg.get('marker_size_m', args.marker_size))
    markers = cfg.get('markers', {}) or {}
    if not markers:
        print(f'配置里没有 markers，先填 {args.config}'); sys.exit(1)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from astra_seg_interfaces.msg import ObjectClouds

    K = DEFAULT_K
    Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], np.float64)

    class Gt(Node):
        def __init__(self):
            super().__init__('pose_gt_aruco')
            self.prev_c = None
            self.det = make_detector(dict_name)
            self.latest = {}
            self.errs = []
            self.f = open(args.csv, 'w')
            self.f.write('t,label,dx_cm,dy_cm,dz_cm,dp_cm,dyaw_deg,'
                         'gt_x,gt_y,gt_z,pred_x,pred_y,pred_z\n')
            self.t0 = None
            self.create_subscription(Image, '/camera/color/image_raw', self.on_img,
                                     qos_profile_sensor_data)
            self.create_subscription(ObjectClouds, '/astra_seg/objects', self.on_obj, 10)
            self.create_timer(2.0, self.report)

        def on_obj(self, msg):
            self.latest = {}
            for i, lab in enumerate(msg.labels):
                p = msg.obb_poses[i].position
                q = msg.obb_poses[i].orientation
                R = rpy_to_R(0, 0, 0)
                # 四元数 → R
                x, y, z, w = q.x, q.y, q.z, q.w
                n = x * x + y * y + z * z + w * w
                s = 2.0 / max(n, 1e-12)
                R = np.array([
                    [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                    [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                    [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])
                self.latest.setdefault(lab, []).append((np.array([p.x, p.y, p.z]), R,
                                                        float(msg.scores[i])))

        def on_img(self, msg):
            bgr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
            gray = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
            for mid, pts in detect(self.det, gray):
                info = markers.get(mid) or markers.get(str(mid))
                if not info:
                    continue
                label = info['label']
                T_mo = T_from(info.get('t_marker_obj', [0, 0, 0]),
                              info.get('rpy_marker_obj', [0, 0, 0]))
                mode = info.get('mode', cfg.get('mode', 'planar'))
                if mode == 'planar':
                    n_reg = np.asarray(info.get('normal', cfg.get('normal', [0, 1, 0])),
                                       dtype=np.float64)
                    n_reg /= (np.linalg.norm(n_reg) + 1e-12)
                    # 构造 R_base：x 轴在平面内，z 轴为法线（朝相机）
                    u_ax = np.cross(n_reg, [1.0, 0.0, 0.0])
                    if np.linalg.norm(u_ax) < 1e-6:
                        u_ax = np.cross(n_reg, [0.0, 0.0, 1.0])
                    u_ax /= np.linalg.norm(u_ax)
                    v_ax = np.cross(n_reg, u_ax)
                    R_base = np.stack([u_ax, v_ax, n_reg], axis=1)
                    p4 = solve_planar(pts, np.array(
                        [[-marker_size / 2, marker_size / 2, 0],
                         [marker_size / 2, marker_size / 2, 0],
                         [marker_size / 2, -marker_size / 2, 0],
                         [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float64),
                        R_base)
                    T = np.eye(4)
                    T[:3, :3] = R_base @ rpy_to_R(0, 0, p4[3])
                    T[:3, 3] = p4[:3]
                else:
                    T = pose_of_marker(pts, marker_size, Km)
                if T is None:
                    continue
                # 时间连续性：marker 不可能一帧跳 5 cm（多半是角点配错/二义性）
                c_now = T[:3, 3]
                if self.prev_c is not None and np.linalg.norm(c_now - self.prev_c) > 0.05:
                    continue
                self.prev_c = c_now
                T_obj = T @ T_mo
                cands = self.latest.get(label) or []
                if not cands:
                    continue
                c_gt = T_obj[:3, 3]
                best = min(cands, key=lambda c: np.linalg.norm(c[0] - c_gt))
                if np.linalg.norm(best[0] - c_gt) > 0.25:
                    continue                     # 对不上号，不算误差
                e = c_gt - best[0]
                dyaw = (yaw_up(best[1]) - yaw_up(T_obj[:3, :3]) + 180.0) % 360.0 - 180.0
                self.errs.append((e, abs(dyaw), dyaw))
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if self.t0 is None:
                    self.t0 = t
                self.f.write(
                    f'{t - self.t0:.3f},{label},{e[0]*100:.3f},{e[1]*100:.3f},{e[2]*100:.3f},'
                    f'{np.linalg.norm(e)*100:.3f},{self.errs[-1][2]:.3f},'
                    f'{c_gt[0]:.4f},{c_gt[1]:.4f},{c_gt[2]:.4f},'
                    f'{best[0][0]:.4f},{best[0][1]:.4f},{best[0][2]:.4f}\n')
                self.f.flush()

        def report(self):
            if not self.errs:
                return
            d = np.array([np.linalg.norm(e[0]) for e in self.errs]) * 100
            r = np.array([e[1] for e in self.errs])
            print(f'[gt] n={len(d)}  平移 p50={np.percentile(d,50):.2f} cm '
                  f'p95={np.percentile(d,95):.2f} cm | yaw p50={np.percentile(r,50):.1f}° '
                  f'p95={np.percentile(r,95):.1f}°', flush=True)
            self.errs.clear()

    rclpy.init()
    node = Gt()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as exc:
        if exc.__class__.__name__ not in ('KeyboardInterrupt', 'ExternalShutdownException'):
            raise
    finally:
        node.f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f'CSV -> {args.csv}')


if __name__ == '__main__':
    main()
