"""RGB-D 实例分割 -> 逐实例点云 -> rosbag2 主节点。

数据流：
    /camera/color/image_raw ┐
                            ├─ ApproximateTimeSynchronizer ──> 生产者回调（只入队，不推理）
    /camera/depth/image_raw ┘                                              │
                                                                  有界队列（可丢最旧）
                                                                           │
                                                                  消费者线程（推理 + 反投影）
                                                                           │
                                                          /astra_seg/objects (ObjectClouds)
                                                                           │
                                                                     rosbag2 落盘
"""

import os
import queue
import threading
import time

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from astra_seg_interfaces.msg import ObjectClouds
from geometry_msgs.msg import Point, Pose, Vector3

from .annotate import (caption_strip, draw_axes_plain, draw_grasp, draw_mask_overlay,
                       draw_label, draw_obb)
from .bag_recorder import BagRecorder
from .cloud_builder import build_cloud, instance_color, obb_edge_points
from .grasp import GripperConfig, evaluate
from .obb import (clean_by_plane, depth_to_points, filter_to_obb, fit_cylinder_obb,
                  fit_obb, fit_obb_on_plane, mask_quad, quat_from_matrix,
                  ransac_plane_largest, voxel_downsample)
from std_msgs.msg import UInt8MultiArray
from .pose_filter import PoseSmoother
from .projection import deproject, erode_mask
from .scene import (claim_mask, deproject_frame, depth_order, mask_uv_index,
                    remove_big_planes)
from .tracker import IouTracker
from . import yolo_seg
from .yolo_seg import YoloSegOnnx, resolve_class_ids, set_class_names

class _NamedDet:
    """把副模型的检测包装成和 Detection 一样的接口，但 label 由我们指定。"""

    __slots__ = ('box', 'score', 'class_id', 'mask', 'label')

    def __init__(self, det, label):
        self.box = det.box
        self.score = det.score
        self.class_id = det.class_id
        self.mask = det.mask
        self.label = label


COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'
OBJECTS_TOPIC = '/astra_seg/objects'
ANNOTATED_TOPIC = '/astra_seg/annotated'
BAG_COLOR_TOPIC = '/astra_seg/color'
BAG_DEPTH_TOPIC = '/astra_seg/depth'


class SegCloudNode(Node):
    def __init__(self):
        super().__init__('astra_seg_cloud')

        # ---------------- 参数 ----------------
        p = self.declare_parameter
        self.model_path = os.path.expanduser(
            p('model_path', '~/astra_models/yolov8s-seg.onnx').value)
        self.imgsz = int(p('imgsz', 640).value)
        self.conf_thres = float(p('conf_thres', 0.25).value)
        self.iou_thres = float(p('iou_thres', 0.45).value)
        # 类别白名单：空列表 = 不过滤。可写类别名（keyboard）或 COCO id（66）
        raw_targets = [str(x) for x in p('target_classes', ['']).value]
        self.target_classes = [x for x in raw_targets if x.strip()]

        self.fx = float(p('fx', 285.1711).value)
        self.fy = float(p('fy', 285.1711).value)
        self.cx = float(p('cx', 159.5).value)
        self.cy = float(p('cy', 119.5).value)

        self.erode_kernel = int(p('erode_kernel', 3).value)
        self.min_points = int(p('min_points', 30).value)
        self.min_depth = float(p('min_depth', 0.15).value)
        self.max_depth = float(p('max_depth', 6.0).value)
        self.pixel_stride = int(p('pixel_stride', 1).value)

        self.sync_slop = float(p('sync_slop', 0.05).value)
        self.sync_queue = int(p('sync_queue', 10).value)
        self.queue_size = int(p('queue_size', 4).value)

        self.record_bag = bool(p('record_bag', True).value)
        self.bag_dir = os.path.expanduser(p('bag_dir', '~/bags').value)
        self.bag_path_arg = os.path.expanduser(p('bag_path', '').value)
        self.save_depth = bool(p('save_depth', False).value)
        self.max_frames = int(p('max_frames', 0).value)

        self.track_iou = float(p('track_iou', 0.3).value)
        self.track_max_age = int(p('track_max_age', 15).value)

        # ---- 中心区域过滤（挡掉画面边缘的干扰物/背景）----
        self.roi_enable = bool(p('roi_enable', True).value)
        self.roi_margin = float(p('roi_margin', 0.10).value)
        self.roi_max_bbox_area = float(p('roi_max_bbox_area', 0.35).value)
        self.roi_min_inside = float(p('roi_min_inside', 0.80).value)
        self._roi_dropped = 0

        # ---- 圆柱类物体（用几何模型而不是平面拟合）----
        raw_cyl = [str(x) for x in p('cylinder_classes', ['cup', 'bottle']).value]
        self.cylinder_classes = {x for x in raw_cyl if x.strip()}
        self.table_stride = int(p('table_stride', 6).value)
        self._table_n = None
        self._table_d = None
        self._table_reject = 0
        self.table_max_jump_deg = float(p('table_max_jump_deg', 8.0).value)
        self._flat_normals = []          # 平面物体（键盘/书）的 z 轴缓存，用于统一桌面法线
        self._table_n_ref = None
        self._yaw_flags = {}
        self._yaw_deg = {}
        self.debug_detail = bool(p('debug_detail', False).value)
        self.draw_grasp_flag = bool(p('draw_grasp', False).value)
        self.debug_cloud = bool(p('debug_cloud', True).value)
        self.max_obj_z = float(p('max_obj_z', 1.10).value)   # 超过该深度(m)的实例丢弃，挡背景
        self.scene_pipeline = bool(p('scene_pipeline', True).value)  # 整帧点云 + 远到近认领
        self.debug_mode = bool(p('debug_mode', False).value)          # 调试模式（默认关）
        self.draw_mask_flag = (self.debug_mode
                               and bool(p('draw_mask', True).value))   # 仅调试模式才叠加 mask
        self.plane_clean_thresh = float(p('plane_clean_thresh', 0.004).value)  # true_pts 的平面阈值(m)
        self.scene_stride = int(p('scene_stride', 3).value)   # 整帧反投影步长（越大越快）
        self.class_conf_raw = [str(x) for x in p('class_conf', ['']).value]  # 形如 'remote:0.08'
        self.secondary_model = str(p('secondary_model', '').value)
        self.secondary_names = [str(x) for x in p('secondary_class_names', ['']).value if str(x).strip()]
        self.secondary_classes = [str(x) for x in p('secondary_classes', ['']).value if str(x).strip()]
        self.secondary_imgsz = int(p('secondary_imgsz', 1024).value)
        self.secondary_conf = float(p('secondary_conf', 0.25).value)
        self.secondary = None
        self.caption_lines = int(p('caption_lines', 5).value)   # 字幕固定行数（防图像高度跳变）

        # ---- OBB / 抓取 ----
        self.enable_obb = bool(p('enable_obb', True).value)
        self.obb_margin = float(p('obb_margin', 0.005).value)
        self.pose_alpha = float(p('pose_smooth_alpha', 0.30).value)
        self.pose_smoother = PoseSmoother(
            alpha=self.pose_alpha,
            dim_max_jump=float(p('pose_dim_max_jump', 0.35).value),
            dim_max_step=float(p('pose_dim_max_step', 0.12).value),
            center_max_step=float(p('pose_center_max_step', 0.05).value))
        self.gripper = GripperConfig(
            suction_radius=float(p('suction_radius', 0.03).value),
            jaw_max_open=float(p('jaw_max_open', 0.08).value),
            jaw_min_open=float(p('jaw_min_open', 0.0).value),
            jaw_width=float(p('jaw_width', 0.04).value),
            standoff=float(p('grasp_standoff', 0.02).value))
        self._grasp_counts = {}

        # ---------------- 模型 ----------------
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'找不到模型: {self.model_path}\n'
                '请先运行 scripts/export_model.sh 导出 yolov8s-seg.onnx')
        raw_names = [str(x) for x in p('class_names', ['']).value]
        raw_names = [x for x in raw_names if x.strip()]
        if raw_names:
            set_class_names(raw_names)
            self.get_logger().info(f'类别表已切换为自定义模型: {raw_names}')
        allowed = resolve_class_ids(self.target_classes) if self.target_classes else None
        if allowed is not None:
            names = [yolo_seg.CLASS_NAMES[i] if i < len(yolo_seg.CLASS_NAMES) else str(i)
                     for i in sorted(allowed)]
            self.get_logger().info(f'类别白名单: {names}')
        log = lambda m: self.get_logger().info(m)  # noqa: E731
        class_conf = {}
        for item in self.class_conf_raw:
            if ':' not in item:
                continue
            name, val = item.split(':', 1)
            try:
                cid = int(resolve_class_ids([name.strip()]).pop())
                class_conf[cid] = float(val)
            except Exception:
                pass
        if class_conf:
            self.get_logger().info(f'类别专属阈值: {class_conf}')
        self.model = YoloSegOnnx(
            self.model_path, imgsz=self.imgsz,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres,
            allowed_class_ids=allowed, logger=log, class_conf=class_conf)

        if self.secondary_model and self.secondary_classes:
            try:
                self.secondary = YoloSegOnnx(
                    self.secondary_model, imgsz=self.secondary_imgsz,
                    conf_thres=self.secondary_conf, iou_thres=self.iou_thres,
                    allowed_class_ids=None, logger=log,
                    class_conf={})
                self.get_logger().info(
                    f'副模型已加载: {self.secondary_model} '
                    f'(imgsz={self.secondary_imgsz}) 只负责 {self.secondary_classes}')
            except Exception as exc:
                self.get_logger().warn(f'副模型加载失败: {exc!r}')
                self.secondary = None

        self.tracker = IouTracker(
            iou_thres=self.track_iou, max_age=self.track_max_age)

        # ---------------- 通信 ----------------
        self.cb_group = ReentrantCallbackGroup()
        self.sub_color = message_filters.Subscriber(
            self, Image, COLOR_TOPIC, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, DEPTH_TOPIC, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=self.sync_queue, slop=self.sync_slop)
        self.sync.registerCallback(self._on_sync)

        self.pub_yaw_valid = self.create_publisher(
            UInt8MultiArray, '/astra_seg/yaw_valid', 10)
        self.pub_objects = self.create_publisher(ObjectClouds, OBJECTS_TOPIC, 10)
        self.pub_dbg_all = self.create_publisher(PointCloud2, '/astra_seg/debug/all', 10)
        self.pub_dbg_plane = self.create_publisher(PointCloud2, '/astra_seg/debug/plane', 10)
        self.pub_dbg_objects = self.create_publisher(PointCloud2, '/astra_seg/debug/objects', 10)
        self.pub_dbg_boxes = self.create_publisher(PointCloud2, '/astra_seg/debug/boxes', 10)
        self.pub_annotated = self.create_publisher(Image, ANNOTATED_TOPIC, 10)

        # 驱动偶发重复注册 publisher 时，同一时间戳会收到两份，去重避免重复处理
        self._last_stamp = None

        # ---------------- 生产者 / 消费者 ----------------
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._worker = None
        self._frames = 0        # 本统计周期内处理帧数
        self._total_frames = 0  # 累计处理帧数（用于 max_frames）
        self._dropped = 0
        self._clouds = 0
        self._latency_sum = 0.0
        self._t_conv = 0.0
        self._t_infer = 0.0
        self._t_cloud = 0.0
        self._t_bag = 0.0
        self._stats_t = time.time()

        # ---------------- 落包 ----------------
        self.recorder = None
        self.bag_path = None
        if self.record_bag:
            self.bag_path = self._make_bag_path()
            topics = {
                BAG_COLOR_TOPIC: 'sensor_msgs/msg/Image',
                OBJECTS_TOPIC: 'astra_seg_interfaces/msg/ObjectClouds',
            }
            if self.save_depth:
                topics[BAG_DEPTH_TOPIC] = 'sensor_msgs/msg/Image'
            if self.enable_obb:
                topics[ANNOTATED_TOPIC] = 'sensor_msgs/msg/Image'
            if self.debug_cloud:
                topics['/astra_seg/debug/all'] = 'sensor_msgs/msg/PointCloud2'
                topics['/astra_seg/debug/plane'] = 'sensor_msgs/msg/PointCloud2'
            self.recorder = BagRecorder(self.bag_path, topics)
            self.get_logger().info(f'记录到 rosbag2: {self.bag_path}')

        # 预热放在订阅生效前后都可以，但必须在工作线程开始消费前完成，
        # 否则首帧会把 CUDA 初始化的 3 秒算进延迟、并把队列灌满。
        self.model.warmup()

        self._worker = threading.Thread(target=self._consume_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f'订阅 {COLOR_TOPIC} + {DEPTH_TOPIC} | 内参 fx={self.fx} cx={self.cx} '
            f'| 腐蚀 k={self.erode_kernel} | 队列={self.queue_size}')
        self.get_logger().info(
            f'输出 {OBJECTS_TOPIC}，等待相机数据…')

    # ------------------------------------------------------------------ 工具
    def _filter_roi(self, dets, w, h):
        """只保留落在画面中心区域的检测。

        三重条件（都要满足）：
          1. 框中心在 ROI 内
          2. 框面积至少有 roi_min_inside 的比例落在 ROI 内
          3. 框面积不超过整幅图的 roi_max_bbox_area（挡掉"整幅背景"这类大框）
        """
        if not self.roi_enable or not dets:
            return dets
        m = self.roi_margin
        x0, y0, x1, y1 = m * w, m * h, (1.0 - m) * w, (1.0 - m) * h
        img_area = float(w * h)
        keep = []
        for d in dets:
            bx1, by1, bx2, by2 = [float(v) for v in d.box]
            bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
            area = bw * bh
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                self._roi_dropped += 1
                continue
            ix = max(0.0, min(bx2, x1) - max(bx1, x0))
            iy = max(0.0, min(by2, y1) - max(by1, y0))
            if (ix * iy) / area < self.roi_min_inside:
                self._roi_dropped += 1
                continue
            if area / img_area > self.roi_max_bbox_area:
                self._roi_dropped += 1
                continue
            keep.append(d)
        return keep

    def _make_bag_path(self):
        if self.bag_path_arg:
            return self.bag_path_arg
        os.makedirs(self.bag_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.bag_dir, f'astra_seg_{stamp}')

    # ------------------------------------------------------------------ 生产者
    def _on_sync(self, color_msg, depth_msg):
        """同步回调只做入队；队列满则丢弃最旧帧，保证实时性。"""
        if self._stop.is_set():
            return
        stamp = (color_msg.header.stamp.sec, color_msg.header.stamp.nanosec)
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp
        try:
            self._queue.put_nowait((color_msg, depth_msg))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((color_msg, depth_msg))
            except queue.Full:
                self._dropped += 1

    # ------------------------------------------------------------------ 消费者
    def _consume_loop(self):
        while not self._stop.is_set():
            try:
                color_msg, depth_msg = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process(color_msg, depth_msg)
            except Exception as exc:  # 单帧失败不能拖垮整条流水线
                self.get_logger().error(f'处理帧失败: {exc!r}',
                                        throttle_duration_sec=2.0)

    def _depth_array(self, msg):
        if msg.encoding not in ('16UC1', 'mono16'):
            self.get_logger().warn(
                f'深度编码是 {msg.encoding}，按 16UC1 处理', throttle_duration_sec=5.0)
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.step == w * 2:
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        rows = raw.reshape(h, msg.step)
        return rows[:, :w * 2].copy().view(np.uint16).reshape(h, w)

    def _process(self, color_msg, depth_msg):
        t0 = time.perf_counter()

        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        depth = self._depth_array(depth_msg)
        h, w = bgr.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        t_conv = time.perf_counter()

        # 桌面平面：depth -> pts -> 最大平面（每帧更新，stride 6 控制开销）
        try:
            dpts = depth_to_points(depth, (self.fx, self.fy, self.cx, self.cy),
                                   stride=self.table_stride)
            if dpts is not None:
                pl = ransac_plane_largest(dpts, thresh=0.012, iters=80, min_ratio=0.25)
                if pl is not None:
                    n_new, d_new, _r = pl
                    inl = np.abs(np.asarray(dpts) @ np.asarray(n_new) + float(d_new)) < 0.015
                    dp = np.asarray(dpts)[inl]
                    if len(dp) > 2500:
                        sel = np.random.default_rng(0).choice(len(dp), 2500, replace=False)
                        dp = dp[sel]
                    self._dbg_plane_pts = dp if len(dp) else None
                    # 先验①：相机俯视桌面，桌面法线应大致"朝上"（相机 -y 方向）
                    up = np.array([0.0, -1.0, 0.0])
                    # RANSAC 的法线方向是任意的：统一翻成"朝上"，否则高度会反号
                    if float(np.asarray(n_new) @ up) < 0:
                        n_new = -np.asarray(n_new)
                        d_new = -float(d_new)
                    ok_dir = abs(float(n_new @ up)) > 0.5
                    # 先验②：与上一帧夹角不能突变（防止偶尔锁到别的平面）
                    if self._table_n is not None:
                        cosang = abs(float(n_new @ self._table_n))
                        d_ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                        ok_jump = d_ang <= self.table_max_jump_deg
                    else:
                        ok_jump = True
                    if ok_dir and ok_jump:
                        self._table_n, self._table_d = n_new, d_new
                        self._table_reject = 0
                    else:
                        self._table_reject += 1
                        # 连续 15 帧都被拒 → 认为场景真的变了，接受新平面
                        if self._table_reject >= 15:
                            self._table_n, self._table_d = n_new, d_new
                            self._table_reject = 0
        except Exception as exc:      # 估计失败不影响主流程
            self.get_logger().warn(f'桌面平面估计失败: {exc}', throttle_duration_sec=5.0)

        self._yaw_flags = {}
        self._yaw_deg = {}
        dets = self.model.infer(bgr)
        # ---- 副模型只负责指定类别（例如 remote），避免"一个模型必须全会" ----
        if self.secondary is not None and self.secondary_classes:
            dets = [d for d in dets if d.label not in self.secondary_classes]
            try:
                extra = self.secondary.infer(bgr)
                for d in extra:
                    name = None
                    cid = int(d.class_id)
                    if self.secondary_names and 0 <= cid < len(self.secondary_names):
                        name = self.secondary_names[cid]
                    if name is None or name not in self.secondary_classes:
                        continue
                    dets.append(_NamedDet(d, name))
            except Exception as exc:
                self.get_logger().warn(f'副模型推理失败: {exc!r}',
                                       throttle_duration_sec=5.0)
        dets = self._filter_roi(dets, w, h)
        ids = self.tracker.update(dets)
        t_infer = time.perf_counter()

        header = Header()
        header.stamp = color_msg.header.stamp
        header.frame_id = color_msg.header.frame_id or 'camera_color_optical_frame'

        out = ObjectClouds()
        out.header = header
        annot = bgr.copy() if self.enable_obb else None
        self._captions = []
        self._caption_colors = []
        self._caption_order = []
        self._caption_rows = {}
        self._dbg_objs = []
        self._dbg_boxes = []
        self._dbg_box_colors = {}
        if not hasattr(self, '_dbg_plane_pts'):
            self._dbg_plane_pts = None
        self._dbg_last_all = None
        self._dbg_last_plane = None

        items = []
        if self.scene_pipeline and self._table_n is not None and len(dets) > 0:
            items = self._scan_scene(bgr, depth, dets, ids)
        if not items:
            items = self._scan_per_mask(bgr, depth, dets, ids)

        # ---- 第二遍：检测"物体踩在另一个平面（桌面/别的物体顶面）上"并重建 ----
        # 只看沿桌面法线的高度分布：众数那层就是它踩着的面。
        # 判据：该层占 >=25%，且它上方有点（本体）、下方几乎没有点。
        if self.enable_obb and not self.scene_pipeline and self._table_n is not None:
            n_t = np.asarray(self._table_n, dtype=np.float32)
            d_t = float(self._table_d)
            for it in items:
                if it['obb'] is None:
                    continue
                P = it['pts']
                if len(P) < 80:
                    continue
                h = P @ n_t + d_t
                span = float(h.max() - h.min())
                if span < 0.015:
                    continue
                bins = max(int(span / 0.004), 4)
                hist, edges = np.histogram(h, bins=bins)
                k = int(np.argmax(hist))
                mode_h = 0.5 * (edges[k] + edges[k + 1])
                frac_mode = hist[k] / len(h)
                above = float((h > mode_h + 0.006).mean())
                below = float((h < mode_h - 0.006).mean())
                if not (frac_mode >= 0.25 and above >= 0.10 and below <= 0.10):
                    continue
                new_obb = fit_obb_on_plane(P, n_t, d_t - mode_h, margin=self.obb_margin,
                                           band=0.008, min_frac=0.20)
                if new_obb is None:
                    continue
                it['obb'] = new_obb
                it['pts'], keep = filter_to_obb(it['pts'], new_obb)
                it['cols'] = it['cols'][keep]
                self.get_logger().info(
                    f'[支撑面] {it["det"].label}: 底面 {mode_h*100:.1f}cm，'
                    f'高度 {new_obb.dims[2]*100:.1f}cm，长宽 '
                    f'{new_obb.dims[0]*100:.1f}x{new_obb.dims[1]*100:.1f}cm')

        for it in items:
            det, tid, pts, cols = it['det'], it['tid'], it['pts'], it['cols']
            obb, grasp = it['obb'], it['grasp']
            out.ids.append(int(tid))
            out.labels.append(det.label)
            out.scores.append(float(det.score))
            out.clouds.append(build_cloud(header, pts, cols))
            self._clouds += 1
            if self.debug_cloud and self.enable_obb and obb is not None:
                col = instance_color(int(tid))
                self._dbg_objs.append((pts.astype(np.float32), cols.astype(np.uint8)))
                bp = obb_edge_points(obb, 0.004)
                self._dbg_boxes.append((bp, np.tile(col, (len(bp), 1))))
                self._dbg_box_colors[int(tid)] = col

            if self.enable_obb:
                self._fill_pose(out, obb, grasp)
                self._draw(annot, obb, grasp, det, out, it.get('mask_dbg'))

        self.pose_smoother.prune(set(int(i) for i in ids))
        self.pose_smoother.prune(set(int(i) for i in ids))
        t_cloud = time.perf_counter()

        ann_msg = None
        if self.enable_obb:
            ids_now = [int(i) for i in out.ids]
            order = [i for i in self._caption_order if i in ids_now]
            order += [i for i in ids_now if i not in order]
            self._caption_order = order
            rows = [self._caption_rows[i] for i in order if i in self._caption_rows]
            self._caption_rows = {k: v for k, v in self._caption_rows.items() if k in ids_now}
            if rows:
                annot = np.vstack([annot, caption_strip(
                    [r[0] for r in rows], annot.shape[1],
                    colors=[r[1] for r in rows],
                    min_lines=self.caption_lines)])
            ann_msg = self.bridge.cv2_to_imgmsg(annot, encoding='bgr8')
            ann_msg.header = header
            self.pub_annotated.publish(ann_msg)

        self.pub_objects.publish(out)

        if self.debug_cloud:
            # 平面点云（桌面/背景），灰色，抽样
            if self._dbg_plane_pts is not None and len(self._dbg_plane_pts) > 0:
                pp = self._dbg_plane_pts
                g = np.full((len(pp), 3), 110, dtype=np.uint8)
                _pmsg = build_cloud(header, pp.astype(np.float32), g)
                self._dbg_last_plane = _pmsg
                self.pub_dbg_plane.publish(_pmsg)
            self._dbg_last_all = None
            self._dbg_last_plane = None
            if self._dbg_objs:
                op = np.vstack([o[0] for o in self._dbg_objs])
                oc = np.vstack([o[1] for o in self._dbg_objs])
                self.pub_dbg_objects.publish(build_cloud(header, op, oc))
            if self._dbg_boxes:
                bp = np.vstack([b[0] for b in self._dbg_boxes])
                bc = np.vstack([b[1] for b in self._dbg_boxes])
                self.pub_dbg_boxes.publish(build_cloud(header, bp, bc))
                # 合并：平面 + 物体 + 框（一条话题看全部）
                tiles_p, tiles_c = [bp], [bc]
                if self._dbg_plane_pts is not None and len(self._dbg_plane_pts) > 0:
                    pp = self._dbg_plane_pts
                    tiles_p.append(pp.astype(np.float32))
                    tiles_c.append(np.full((len(pp), 3), 70, dtype=np.uint8))
                op = np.vstack([o[0] for o in self._dbg_objs])
                oc = np.vstack([o[1] for o in self._dbg_objs])
                tiles_p.append(op); tiles_c.append(oc)
                _amsg = build_cloud(header, np.vstack(tiles_p), np.vstack(tiles_c))
                self._dbg_last_all = _amsg
                self.pub_dbg_all.publish(_amsg)
        if self._yaw_flags:
            msg = UInt8MultiArray()
            msg.data = [1 if self._yaw_flags.get(int(i), 1) else 0 for i in out.ids]
            self.pub_yaw_valid.publish(msg)

        if self.recorder is not None:
            stamp_ns = (color_msg.header.stamp.sec * 1_000_000_000
                        + color_msg.header.stamp.nanosec)
            self.recorder.write(BAG_COLOR_TOPIC, color_msg, stamp_ns)
            if self.save_depth:
                self.recorder.write(BAG_DEPTH_TOPIC, depth_msg, stamp_ns)
            if ann_msg is not None:
                self.recorder.write(ANNOTATED_TOPIC, ann_msg, stamp_ns)
            self.recorder.write(OBJECTS_TOPIC, out, stamp_ns)
            if self.debug_cloud and self._dbg_last_all is not None:
                self.recorder.write('/astra_seg/debug/all', self._dbg_last_all, stamp_ns)
            if self.debug_cloud and self._dbg_last_plane is not None:
                self.recorder.write('/astra_seg/debug/plane', self._dbg_last_plane, stamp_ns)
        t_bag = time.perf_counter()

        self._frames += 1
        self._total_frames += 1
        self._latency_sum += t_bag - t0
        self._t_conv += t_conv - t0
        self._t_infer += t_infer - t_conv
        self._t_cloud += t_cloud - t_infer
        self._t_bag += t_bag - t_cloud

        if self.max_frames and self._total_frames >= self.max_frames:
            self.get_logger().info(
                f'已处理 {self._total_frames} 帧（max_frames 上限），准备退出')
            self._stop.set()

        self._log_stats()

    # ---------------- 点云扫描：两种模式 ----------------

    def _scan_per_mask(self, bgr, depth, dets, ids):
        """旧模式：每个 mask 各自反投影 + 去平面。"""
        items = []
        for det, tid in zip(dets, ids):
            mask = erode_mask(det.mask, self.erode_kernel)
            pts, cols = deproject(
                mask, depth, bgr, self.fx, self.fy, self.cx, self.cy,
                min_depth=self.min_depth, max_depth=self.max_depth,
                stride=self.pixel_stride, min_points=self.min_points)
            if pts is None:
                continue
            pts, keep = clean_by_plane(pts, thresh=0.010)
            cols = cols[keep]
            if len(pts) < 50:
                continue
            obb = grasp = None
            if self.enable_obb:
                obb, grasp = self._estimate(pts, det.label, tid, det.mask, depth)
                if obb is not None:
                    pts, keep = filter_to_obb(pts, obb)
                    cols = cols[keep]
            items.append(dict(det=det, tid=int(tid), pts=pts, cols=cols,
                              obb=obb, grasp=grasp))
        return items

    def _scan_scene(self, bgr, depth, dets, ids):
        """你的思路：整帧点云 → 去桌面 → 物体按"远到近"依次认领，认领完就删。

        每个物体的几何仍用原来的 _estimate()（mask 四边形/平面/圆柱分支），
        这里只改"点的归属"和"处理顺序"。
        """
        h, w = bgr.shape[:2]
        pts, uv, _ = deproject_frame(
            depth, self.fx, self.fy, self.cx, self.cy,
            stride=max(2, int(self.scene_stride)),
            min_depth=max(0.45, float(self.min_depth)),
            max_depth=min(1.35, float(self.max_depth)))
        if pts is None or len(pts) < 200:
            return []
        cols = bgr[np.clip(uv[:, 1], 0, h - 1), np.clip(uv[:, 0], 0, w - 1)]

        # 只删"桌面"这一个平面；桌面点单独下采样留给 RViz
        keep, planes, plane_pts = remove_big_planes(
            pts, n_planes=1, thresh=0.006,
            ref_n=(np.asarray(self._table_n, dtype=np.float32)
                   if self._table_n is not None else None),
            ref_d=(float(self._table_d) if self._table_n is not None else None))
        if plane_pts:
            tp = voxel_downsample(np.vstack(plane_pts), 0.05)
            if len(tp) > 4000:
                sel = np.random.default_rng(0).choice(len(tp), 4000, replace=False)
                tp = tp[sel]
            self._dbg_plane_pts = tp
        pts, uv, cols = pts[keep], uv[keep], cols[keep]
        if len(pts) < 150:
            return []

        # 高度基准：就用刚删掉的那个平面
        if planes:
            n_t = np.asarray(planes[0][0], dtype=np.float32)
            d_t = float(planes[0][1])
        else:
            n_t = np.asarray(self._table_n, dtype=np.float32)
            d_t = float(self._table_d)
        hh = pts @ n_t + d_t
        ok = (hh > -0.03) & (hh < 0.30)
        pts, uv, cols = pts[ok], uv[ok], cols[ok]
        if len(pts) < 120:
            return []

        # 认领时用"腐蚀后的 mask"：原始 mask 边界上的像素深度不可靠（飞点），
        # 会把这些点带进物体点云，把框撑大。
        dets_er = []
        for d in dets:
            m = erode_mask(d.mask, self.erode_kernel)
            if m.sum() < 20:
                m = d.mask
            dd = type(d)(d.box, d.score, d.class_id, m)
            dets_er.append(dd)
        # ---- 你的顺序：mask → pts → 平面拟合取 true_pts → 从总点云删 true_pts → 下一个 ----
        order, masks, depth_of = depth_order(dets_er, pts, uv, min_pts=40)
        alive = np.ones(len(pts), dtype=bool)
        items = []
        for k in order:
            det = dets[k]
            tid = int(ids[k])
            if depth_of.get(k, -1) <= 0:
                continue
            m_k = claim_mask(masks, depth_of, k, uv, alive)
            sel = mask_uv_index(m_k, uv) & alive
            if sel.sum() < 40:
                continue
            idx = np.nonzero(sel)[0]
            P, C = pts[idx], cols[idx]
            if det.label in self.cylinder_classes:
                # 圆柱是曲面：不能按"贴平面"筛，否则会把杯身砍掉只剩一条带
                P2, C2, idx2 = P, C, idx
            else:
                # 平面拟合 → 只保留属于该物体表面的点（true_pts）
                P2, keep = clean_by_plane(P, thresh=self.plane_clean_thresh)
                # 保护：严格阈值偶尔会把物体裁得只剩零星点（平面估计偏了），
                # 这时整帧会被丢弃，表现就是"框忽隐忽现"。留下 <40% 就回退。
                if int(keep.sum()) < max(40, int(0.4 * len(P))):
                    P2, C2, idx2 = P, C, idx
                else:
                    C2 = C[keep]
                    idx2 = idx[keep]
            if len(idx2) < 40:
                continue
            alive[idx2] = False          # 只删 true_pts，剩下的留给后面的物体
            UV2 = uv[idx2]
            obb = grasp = None
            if self.enable_obb:
                obb, grasp = self._estimate(P2, det.label, tid, det.mask, depth)
                if obb is not None:
                    P2, keep3 = filter_to_obb(P2, obb)
                    C2 = C2[keep3]
                    UV2 = UV2[keep3]
            mask_dbg = None
            if self.draw_mask_flag:
                mask_dbg = np.zeros((h, w), dtype=bool)
                uu = np.clip(UV2[:, 0], 0, w - 1)
                vv = np.clip(UV2[:, 1], 0, h - 1)
                mask_dbg[vv, uu] = True
                kk = 2 * max(1, int(self.pixel_stride)) + 1
                mask_dbg = cv2.dilate(mask_dbg.astype(np.uint8),
                                      np.ones((kk, kk), np.uint8), iterations=1).astype(bool)
            items.append(dict(det=det, tid=tid, pts=P2, cols=C2,
                              obb=obb, grasp=grasp, mask_dbg=mask_dbg))
        return items

    def _mask_rect(self, mask):
        """mask 的 2D 最小矩形；用于把轮廓反投影成 OBB。"""
        if mask is None or not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return None
        return cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))

    def _estimate(self, pts, label, tid, mask=None, depth=None):
        min_pts = 35 if label in self.cylinder_classes else 80
        if len(pts) < min_pts:
            return None, None
        try:
            rect = self._mask_rect(mask)
            quad = mask_quad(mask)
            hint = None
            if rect is not None:
                (rw, rh), ang = rect[1], rect[2]
                hint = float(ang if rw >= rh else ang + 90.0)
            if (label in self.cylinder_classes and self._table_n is not None):
                n_use = self._table_n_ref if self._table_n_ref is not None else self._table_n
                d_use = float(-float(np.asarray(n_use) @ pts.mean(axis=0))) if                     self._table_n_ref is not None else self._table_d
                obb, meta = fit_cylinder_obb(pts, label, n_use, d_use,
                                             K=(self.fx, self.fy, self.cx, self.cy),
                                             margin=self.obb_margin)
                if meta is not None:
                    self._yaw_flags[int(tid)] = bool(meta['yaw_valid'])
                    self._yaw_deg[int(tid)] = (float(meta['handle_deg'])
                                               if meta['yaw_valid'] else None)
                else:                      # 退化帧 → 回退到多边形分支
                    obb = fit_obb(pts, label, margin=self.obb_margin,
                                  angle_hint=hint, mask_rect=rect,
                                  mask_quad_pts=quad, image_mask=mask,
                                  K=(self.fx, self.fy, self.cx, self.cy))
            else:
                obb = fit_obb(pts, label, margin=self.obb_margin,
                              angle_hint=hint, mask_rect=rect,
                              mask_quad_pts=quad, image_mask=mask,
                              K=(self.fx, self.fy, self.cx, self.cy))
            if obb is not None and label not in self.cylinder_classes:
                nv = np.asarray(obb.normal, dtype=np.float64)
                nv = nv / np.linalg.norm(nv)
                if self._table_n_ref is not None and float(nv @ self._table_n_ref) < 0:
                    nv = -nv
                self._flat_normals.append(nv)
                if len(self._flat_normals) > 60:
                    self._flat_normals.pop(0)
                m = np.mean(np.asarray(self._flat_normals), axis=0)
                self._table_n_ref = m / np.linalg.norm(m)
            if obb is not None:
                if label in self.cylinder_classes:
                    # 回转体的 yaw 不可观测（把手方向也有 180° 歧义）：
                    # 框的朝向一律用"桌面法线 + 固定参考方向"的规范基，
                    # 保证框不随 mask 噪声乱转；把手角度只作为元数据。
                    obb = self._canonical_cylinder_obb(obb)
                obb = self.pose_smoother.update(tid, obb)
                return obb, evaluate(obb, self.gripper)
        except Exception as exc:
            self.get_logger().warn(
                f'OBB 估计失败: {exc!r}', throttle_duration_sec=5.0)
        return None, None

    def _canonical_cylinder_obb(self, obb):
        """回转体的框：朝向固定（与桌面法线绑定），footprint 取直径的正方形。

        原因：圆柱/瓶子的 yaw 不可观测（把手方向还有 180° 歧义），
        每帧让它自己估方向会 180° 翻来翻去，看起来就是"框在转"。
        """
        # 用"桌面法线"而不是圆柱自己拟合的轴：圆柱轴每帧会晃动，
        # 用它当基准的话规范化后的框还是会跟着转。
        n_src = self._table_n if self._table_n is not None else obb.normal
        n = np.asarray(n_src, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        # 统一朝上，避免 ±n 造成的 180° 翻转
        up = np.array([0.0, -1.0, 0.0])
        if float(n @ up) < 0:
            n = -n
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(n @ ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        x = ref - float(ref @ n) * n
        if np.linalg.norm(x) < 1e-6:
            return obb
        x /= np.linalg.norm(x)
        y = np.cross(n, x)
        R = np.stack([x, y, n], axis=1).astype(np.float32)
        # footprint 用近似直径的正方形（l≈w），消除长宽互换带来的跳动
        # 取较小的那个当直径：偏大的那个通常是部分视角拟合出的椭圆
        r2 = float(min(obb.dims[0], obb.dims[1]))
        obb.rotation = R
        obb.normal = R[:, 2].copy()
        obb.dims = np.array([r2, r2, float(obb.dims[2])], dtype=np.float32)
        return obb

    def _fill_pose(self, out, obb, grasp):
        pose = Pose()
        dims = Vector3()
        corners = np.zeros((8, 3), np.float32)
        if obb is not None:
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in obb.center)
            q = quat_from_matrix(obb.rotation)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
                float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            dims.x, dims.y, dims.z = (float(v) for v in obb.dims)
            corners = obb.corners()

        out.obb_poses.append(pose)
        out.obb_dims.append(dims)
        for c in corners:
            pt = Point()
            pt.x, pt.y, pt.z = float(c[0]), float(c[1]), float(c[2])
            out.obb_corners.append(pt)

        gp = Pose()
        if grasp is not None:
            gp.position.x, gp.position.y, gp.position.z = (float(v) for v in grasp.position)
            qg = quat_from_matrix(grasp.rotation)
            gp.orientation.x, gp.orientation.y, gp.orientation.z, gp.orientation.w = (
                float(qg[0]), float(qg[1]), float(qg[2]), float(qg[3]))
            out.grasp_types.append(int(grasp.kind))
            out.grasp_widths.append(float(grasp.width))
            out.grasp_scores.append(float(grasp.score))
            self._grasp_counts[grasp.name] = self._grasp_counts.get(grasp.name, 0) + 1
        else:
            out.grasp_types.append(0)
            out.grasp_widths.append(0.0)
            out.grasp_scores.append(0.0)
        out.grasp_poses.append(gp)

    def _draw(self, annot, obb, grasp, det, out, mask_dbg=None):
        if annot is None or obb is None or grasp is None:
            return
        # 框颜色 = 该实例在 RViz 里的颜色（和字幕色块、点云框一致）
        color = tuple(int(v) for v in instance_color(int(out.ids[-1])))
        if mask_dbg is not None:
            draw_mask_overlay(annot, mask_dbg, color, alpha=0.30)
        draw_obb(annot, obb, self.fx, self.fy, self.cx, self.cy, color)
        if self.draw_grasp_flag:
            draw_grasp(annot, grasp, self.fx, self.fy, self.cx, self.cy, color)
        draw_axes_plain(annot, obb, self.fx, self.fy, self.cx, self.cy, length=0.05)
        # 文字不压在画面上，统一放到下方说明条
        c = obb.center
        yaw = self._yaw_deg.get(int(out.ids[-1]))
        yaw_txt = f'{yaw:.0f}deg' if yaw is not None else 'n/a'
        self._caption_rows[int(out.ids[-1])] = (
            f'id{int(out.ids[-1])} {det.label} ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})m '
            f'{obb.dims[0]*100:.1f}x{obb.dims[1]*100:.1f}x{obb.dims[2]*100:.1f}cm '
            f'yaw {yaw_txt}', instance_color(int(out.ids[-1])))
        self._captions.append(
            f'id{int(out.ids[-1])} {det.label} ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})m '
            f'{obb.dims[0]*100:.1f}x{obb.dims[1]*100:.1f}x{obb.dims[2]*100:.1f}cm '
            f'yaw {yaw_txt}')

    def _log_stats(self):
        now = time.time()
        dt = now - self._stats_t
        if dt < 2.0:
            return
        n = max(1, self._frames)
        fps = self._frames / dt if dt > 0 else 0.0
        lat = self._latency_sum / n * 1000.0
        self.get_logger().info(
            f'帧={self._frames} 处理={fps:.1f}fps 延迟={lat:.0f}ms '
            f'[转换={self._t_conv/n*1000:.0f} 推理={self._t_infer/n*1000:.0f} '
            f'点云={self._t_cloud/n*1000:.0f} 落包={self._t_bag/n*1000:.0f}] '
            f'点云数={self._clouds} 丢弃={self._dropped} 轨迹={len(self.tracker.tracks)} '
            f'抓取={self._grasp_counts}')
        self._frames = 0
        self._clouds = 0
        self._latency_sum = 0.0
        self._t_conv = self._t_infer = self._t_cloud = self._t_bag = 0.0
        self._stats_t = now

    @property
    def finished(self):
        return self._stop.is_set()

    # ------------------------------------------------------------------ 收尾
    def stop(self):
        if getattr(self, '_stopped', False):
            return
        self._stopped = True
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        # 丢掉队列里残留的消息，避免 rclpy 关闭类型支持后再析构它们
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self.recorder is not None:
            self.recorder.close()
            self.get_logger().info(f'rosbag 已关闭: {self.bag_path}')

    def destroy_node(self):
        self.stop()
        super().destroy_node()

    @property
    def bridge(self):
        if not hasattr(self, '_bridge'):
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        return self._bridge


def main(args=None):
    rclpy.init(args=args)
    node = SegCloudNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
