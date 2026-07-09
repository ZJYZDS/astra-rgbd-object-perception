#! /home/zjy/anaconda3/envs/yolo3D_py38/bin/python
# coding=utf-8

import rospy
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
import struct
import serial
from types import SimpleNamespace

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from utils.normal_utils import (
    pca_normal,
    cluster_normals,
    classify_faces,
    build_rotation_from_normal,
    cube_center_from_face,
)

FRAME_START = b'\x0B'
FRAME_END = b'\x1A'


def _ns(d):
    """递归将 dict 转为 SimpleNamespace."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _ns(v)
    return SimpleNamespace(**d)


class KFSGrabPoseEstimator:
    def __init__(self):
        rospy.init_node("kfs_pose_estimator")
        self._load_config()
        self.z_buffer = []
        self.camera_intrinsics = dict(
            fx=self.cfg.camera.fx,
            fy=self.cfg.camera.fy,
            cx=self.cfg.camera.cx,
            cy=self.cfg.camera.cy,
        )

        # ROS publishers
        self.pub_pose = rospy.Publisher(
            self.cfg.ros.pose_topic, PoseStamped, queue_size=5)
        self.pub_coord = rospy.Publisher(
            self.cfg.ros.coord_topic, Float32MultiArray, queue_size=5)

        # YOLO
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.cfg.yolo.model_path).to(self.device)
        rospy.loginfo("YOLO Model Loaded Successfully")

        # RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self._init_camera()

        # Serial
        self.ser = None
        if self.cfg.serial.port:
            try:
                self.ser = serial.Serial(
                    port=self.cfg.serial.port,
                    baudrate=self.cfg.serial.baudrate,
                    timeout=self.cfg.serial.timeout,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                )
            except Exception as e:
                rospy.logwarn("Serial open failed: %s", e)

        self._update_intrinsics()
        rospy.loginfo("Config loaded: %s",
                      {k: dict(vars(v)) if hasattr(v, '__dict__') else v
                       for k, v in vars(self.cfg).items()})

    # ========== 配置加载 ==========

    def _load_config(self):
        """从 rosparam 加载所有参数 (支持 launch 文件中的 YAML 导入)."""
        self.cfg = _ns({
            "camera": {
                "fx": rospy.get_param("~camera/fx", 321.6),
                "fy": rospy.get_param("~camera/fy", 321.6),
                "cx": rospy.get_param("~camera/cx", 320.7),
                "cy": rospy.get_param("~camera/cy", 174.2),
            },
            "yolo": {
                "model_path": rospy.get_param("~yolo/model_path",
                    "/home/zjy/roboncon2025_ws/src/yolo_depth_pkg/scripts/best.pt"),
                "img_sz": rospy.get_param("~yolo/img_sz", 640),
                "conf": rospy.get_param("~yolo/conf", 0.5),
                "iou": rospy.get_param("~yolo/iou", 0.5),
            },
            "serial": {
                "port": rospy.get_param("~serial/port", "/dev/ttyUSB0"),
                "baudrate": rospy.get_param("~serial/baudrate", 115200),
                "timeout": rospy.get_param("~serial/timeout", 1.0),
            },
            "mask": {
                "erode_kernel": rospy.get_param("~mask/erode_kernel", 5),
                "erode_iterations": rospy.get_param("~mask/erode_iterations", 2),
                "min_area": rospy.get_param("~mask/min_area", 50),
                "erosion_min_area": rospy.get_param("~mask/erosion_min_area", 100),
                "polygon_epsilon_factor": rospy.get_param("~mask/polygon_epsilon_factor", 0.015),
            },
            "normal": {
                "sample_half_window": rospy.get_param("~normal/sample_half_window", 8),
                "min_valid_points": rospy.get_param("~normal/min_valid_points", 30),
                "cluster_cosine_threshold": rospy.get_param(
                    "~normal/cluster_cosine_threshold", 0.6),
            },
            "depth": {
                "min_mm": rospy.get_param("~depth/min_mm", 0),
                "max_mm": rospy.get_param("~depth/max_mm", 3000),
            },
            "cube": {
                "side_mm": rospy.get_param("~cube/side_mm", 350.0),
            },
            "z_filter": {
                "buffer_size": rospy.get_param("~z_filter/buffer_size", 5),
                "anomaly_threshold_mm": rospy.get_param("~z_filter/anomaly_threshold_mm", 300),
            },
            "ros": {
                "pose_topic": rospy.get_param("~ros/pose_topic", "/kfs_pose"),
                "coord_topic": rospy.get_param("~ros/coord_topic", "/center_coord"),
                "frame_id": rospy.get_param("~ros/frame_id", "map"),
            },
            "axis_mapping": {
                "offset_x_mm": rospy.get_param("~axis_mapping/offset_x_mm", 0.0),
                "offset_y_mm": rospy.get_param("~axis_mapping/offset_y_mm", 0.0),
                "offset_z_mm": rospy.get_param("~axis_mapping/offset_z_mm", 0.0),
            },
        })

    # ========== 初始化 ==========

    def _init_camera(self):
        try:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            rospy.loginfo("Depth scale: %.4f", self.depth_scale)
        except Exception as e:
            rospy.logerr("Camera init failed: %s", e)
            raise

    def _update_intrinsics(self):
        profiles = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(
            profiles.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        self.camera_intrinsics["fx"] = intr.fx
        self.camera_intrinsics["fy"] = intr.fy
        self.camera_intrinsics["cx"] = intr.ppx
        self.camera_intrinsics["cy"] = intr.ppy
        rospy.loginfo("Camera intrinsics updated: %s", self.camera_intrinsics)

    # ========== YOLO 结果解析 ==========

    def dispose_result(self, results):
        mask_list = []
        for result in results:
            if result is None or result.masks is None or len(result.masks.data) == 0:
                continue
            for i in range(len(result.masks.data)):
                try:
                    mask = result.masks.data[i].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    if i >= len(result.boxes.cls):
                        continue
                    cls_id = int(result.boxes.cls[i])
                    mask_list.append({"mask": mask, "cls": cls_id})
                except Exception as e:
                    rospy.logerr("Failed to process mask %d: %s", i, e)
                    continue
        return mask_list if mask_list else None

    # ========== 采样点提取 ==========

    def _fit_hexagon(self, contour, perimeter):
        """尝试用不同 epsilon 拟合六边形，返回 6 个角点或 None."""
        for eps_factor in [0.015, 0.02, 0.025, 0.018, 0.01]:
            approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
            if len(approx) == 6:
                return approx.reshape(-1, 2).astype(np.float32)
        return None

    def extract_sample_points(self, mask):
        """从六边形 mask 中提取面内部采样点。

        思路：
        1. 腐蚀 → 六边形拟合 → 6个角点（环）
        2. 取右下角点(idx)，找右2(idx+2)%6 和 左2(idx-2)%6 → 3个关键角点(三角形)
        3. 关键角点间的连线中点必在 mask 面内部（非边缘）
        4. 过滤掉在 mask 外的中点 → 返回纯面内采样点

        若拟合不出 6 边形，回退到角点+边中点方案。
        """
        c = self.cfg.mask
        ksize = c.erode_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded = cv2.erode(mask, kernel, iterations=c.erode_iterations)

        if cv2.countNonZero(eroded) < c.erosion_min_area:
            eroded = mask

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        # 尝试六边形拟合
        corners = self._fit_hexagon(contour, perimeter)

        if corners is not None and len(corners) == 6:
            # 六边形方案: 三角形内点采样
            n = 6
            # 右下角点: max(x + y)
            idx = int(np.argmax(corners[:, 0] + corners[:, 1]))

            # 三个关键角点: 当前 + 右2 + 左2 (每隔一个取)
            i0 = idx
            i1 = (idx + 2) % n
            i2 = (idx - 2 + n) % n
            key = [corners[i0], corners[i1], corners[i2]]

            # 关键角点间连线中点 → 面内部采样
            sample_pts = []
            for i in range(3):
                j = (i + 1) % 3
                mid = (key[i] + key[j]) / 2.0
                mx, my = int(mid[0]), int(mid[1])
                if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
                    if mask[my, mx] > 0:
                        sample_pts.append(mid)

            if len(sample_pts) >= 2:
                rospy.loginfo("Hexagon interior sampling: %d pts", len(sample_pts))
                return np.array(sample_pts)

        # 回退: 角点 + 边中点
        approx = cv2.approxPolyDP(contour, c.polygon_epsilon_factor * perimeter, True)
        corners = approx.reshape(-1, 2).astype(np.float32)
        n = len(corners)
        midpoints = [(corners[i] + corners[(i + 1) % n]) / 2.0 for i in range(n)]
        return np.vstack([corners, np.array(midpoints)])

    # ========== 采样点法线计算 ==========

    def compute_sample_normal(self, cx, cy, depth_img):
        c = self.cfg.normal
        half = c.sample_half_window
        h, w = depth_img.shape

        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + half)
        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + half)

        if x1 - x0 < 5 or y1 - y0 < 5:
            return None, None

        ys, xs = np.mgrid[y0:y1, x0:x1]
        depths = depth_img[ys, xs].astype(np.float32)

        d_cfg = self.cfg.depth
        valid = (depths > d_cfg.min_mm) & (depths < d_cfg.max_mm)
        if np.sum(valid) < c.min_valid_points:
            return None, None

        xs_v = xs[valid]
        ys_v = ys[valid]
        depths_v = depths[valid]

        intr = self.camera_intrinsics
        x_3d = (xs_v - intr["cx"]) * depths_v / intr["fx"]
        y_3d = (ys_v - intr["cy"]) * depths_v / intr["fy"]
        z_3d = depths_v

        points = np.stack([x_3d, y_3d, z_3d], axis=1)
        return pca_normal(points)

    # ========== 6-DOF 位姿（正方体约束）==========

    def compute_6dof_pose(self, face_groups):
        if not face_groups:
            rospy.logwarn("No valid face group detected")
            return None, None

        if 'top' in face_groups:
            primary = face_groups['top']
            primary_axis = 'z'
        else:
            primary = face_groups['front']
            primary_axis = 'x'

        face_normal = primary['normal']
        face_center = primary['center']

        R = build_rotation_from_normal(face_normal, primary_axis)

        # 双面交叉验证
        if len(face_groups) >= 2 and 'top' in face_groups and 'front' in face_groups:
            n_top = face_groups['top']['normal']
            n_front = face_groups['front']['normal']
            dot_val = abs(np.dot(n_top, n_front))
            if dot_val > 0.3:
                rospy.logwarn("Top/front normals not orthogonal: %.3f", dot_val)
            R = build_rotation_from_normal(n_top, 'z')

        half = self.cfg.cube.side_mm / 2.0
        t_cube = cube_center_from_face(face_center, face_normal, half)
        return R, t_cube

    # ========== 坐标轴映射 ==========

    def axis_mapping(self, t):
        """相机系 → base_link 系: (x, y, z) → (z + ox, x + oy, -y + oz)."""
        x, y, z = t
        o = self.cfg.axis_mapping
        return np.array([z + o.offset_x_mm, x + o.offset_y_mm, -y + o.offset_z_mm])

    @staticmethod
    def axis_mapping_rotation(R):
        A = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, -1, 0]], dtype=np.float64)
        return A @ R

    # ========== 四元数 ==========

    @staticmethod
    def rot_to_quat(R):
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

        trace = m00 + m11 + m22
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return np.array([x, y, z, w])

    # ========== ROS 发布 ==========

    def publish_pose(self, R, t):
        t_mapped = self.axis_mapping(t)
        R_mapped = self.axis_mapping_rotation(R)
        q = self.rot_to_quat(R_mapped)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.cfg.ros.frame_id
        pose_msg.pose.position = Point(
            t_mapped[0] / 1000.0,
            t_mapped[1] / 1000.0,
            t_mapped[2] / 1000.0,
        )
        pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.pub_pose.publish(pose_msg)

        # 兼容旧的 center_coord topic
        msg_coord = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.size = 3
        dim.stride = 3
        dim.label = "cube_center"
        msg_coord.layout.dim.append(dim)
        msg_coord.data = [float(t[0]), float(t[1]), float(t[2])]
        self.pub_coord.publish(msg_coord)

    # ========== 串口 ==========

    def serial_pub(self, coord_mm):
        if not self.ser:
            return
        if coord_mm is None or any(v is None for v in coord_mm):
            return
        x, y, z = coord_mm
        try:
            data = struct.pack('<3f', x, y, z)
            self.ser.write(FRAME_START + data + FRAME_END)
            rospy.loginfo("Serial sent: %.1f, %.1f, %.1f", x, y, z)
        except Exception as e:
            rospy.logerr("Serial send failed: %s", e)

    # ========== Z 滤波 ==========

    def filter_z(self, coord):
        z = coord[2]
        c = self.cfg.z_filter
        if len(self.z_buffer) < c.buffer_size:
            self.z_buffer.append(z)
            return np.mean(self.z_buffer)
        mean_z = np.mean(self.z_buffer)
        if abs(mean_z - z) > c.anomaly_threshold_mm:
            rospy.logwarn("z anomaly: current %.1f, mean %.1f, using mean", z, mean_z)
            return mean_z
        else:
            self.z_buffer.pop(0)
            self.z_buffer.append(z)
        return np.mean(self.z_buffer)

    # ========== 主循环 ==========

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            # 1. YOLO 推理
            results = self.model.predict(
                source=color_img,
                imgsz=self.cfg.yolo.img_sz,
                conf=self.cfg.yolo.conf,
                iou=self.cfg.yolo.iou,
                verbose=False,
            )

            # 2. 解析 mask
            mask_list = self.dispose_result(results)
            if mask_list is None:
                continue

            # 3. 选择中心最近的 mask
            best_mask = None
            min_dx = np.inf
            for info in mask_list:
                mask = info["mask"]
                if cv2.countNonZero(mask) < self.cfg.mask.min_area:
                    continue
                ys, xs = np.where(mask > 0)
                cx = int(np.mean(xs))
                if abs(cx - 320) < min_dx:
                    min_dx = abs(cx - 320)
                    best_mask = mask

            if best_mask is None:
                continue

            # 4. 采样点提取
            sample_pts = self.extract_sample_points(best_mask)
            if sample_pts is None or len(sample_pts) < 4:
                continue

            # 5. 每个采样点计算法线
            items = []
            for (cx, cy) in sample_pts:
                normal, centroid = self.compute_sample_normal(cx, cy, depth_img)
                if normal is not None:
                    items.append((normal, centroid))

            if len(items) < 2:
                continue

            # 6. 聚类 + 分类
            clusters = cluster_normals(items, self.cfg.normal.cluster_cosine_threshold)
            face_groups = classify_faces(clusters)

            if face_groups is None:
                continue

            rospy.loginfo("Detected faces: %s", list(face_groups.keys()))

            # 7. 位姿计算
            R, t = self.compute_6dof_pose(face_groups)
            if R is None:
                continue

            # 8. Z 滤波
            self.filter_z(t)

            # 9. 发布
            t_mapped = self.axis_mapping(t)
            rospy.loginfo("Mapped pos (mm): %.1f, %.1f, %.1f",
                          t_mapped[0], t_mapped[1], t_mapped[2])
            self.publish_pose(R, t)
            self.serial_pub(t_mapped)


if __name__ == "__main__":
    node = None
    try:
        node = KFSGrabPoseEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node:
            node.pipeline.stop()
            if node.ser:
                node.ser.close()
        rospy.loginfo("Program exit, resources released")
