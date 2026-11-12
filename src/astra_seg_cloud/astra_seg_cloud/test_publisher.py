"""合成话题发布器：没有相机时用它端到端验证整条链路。

用一张真实图片（含人/车等 COCO 目标）模拟 320x240 彩色流，
同时发布一张带噪声、带无效零值、带深度梯度的 16UC1 深度图，
并让图像缓慢平移，用来检验跟踪 ID 是否稳定。

    ros2 run astra_seg_cloud seg_cloud_test_pub --ros-args \
        -p image_path:=/home/zjy/astra_models/test_bus.jpg -p fps:=12.0
"""

from astra_seg_cloud.entry import _prepend_venv, _strip_user_site

# 必须先摘掉被 NumPy 2 污染的 user site，再导入 cv2 / cv_bridge
_strip_user_site()
_prepend_venv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

W, H = 320, 240


class TestPublisher(Node):
    def __init__(self):
        super().__init__('seg_cloud_test_pub')
        image_path = self.declare_parameter(
            'image_path', '/home/zjy/astra_models/test_bus.jpg').value
        fps = float(self.declare_parameter('fps', 12.0).value)
        depth_m = float(self.declare_parameter('depth_m', 2.0).value)
        self.motion = float(self.declare_parameter('motion_px', 40.0).value)

        self.bridge = CvBridge()
        self.phase = 0.0

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'读不到测试图片: {image_path}')
        self.base = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        yy, xx = np.mgrid[0:H, 0:W]
        self.depth = (depth_m * 1000.0 + xx * 1.5 + yy * 1.0).astype(np.float32)
        rng = np.random.default_rng(0)
        self.noise = rng.normal(0, 8.0, (H, W)).astype(np.float32)
        self.zero_zone = ((xx - 60) ** 2 + (yy - 60) ** 2) < 35 ** 2

        self.pub_c = self.create_publisher(Image, '/camera/color/image_raw', 5)
        self.pub_d = self.create_publisher(Image, '/camera/depth/image_raw', 5)
        self.timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f'测试发布器: {W}x{H} @ {fps}fps, 深度 {depth_m}m, 源图 {image_path}')

    def _tick(self):
        self.phase += 0.05
        shift = int(self.motion * np.sin(self.phase))
        frame = np.roll(self.base, shift, axis=1)
        if shift > 0:
            frame[:, :shift] = 114

        stamp = self.get_clock().now().to_msg()

        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = 'camera_color_optical_frame'

        depth = self.depth + self.noise
        depth[self.zero_zone] = 0.0
        depth_u16 = np.clip(depth, 0, 65535).astype(np.uint16)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_u16, encoding='16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'camera_depth_optical_frame'

        self.pub_c.publish(color_msg)
        self.pub_d.publish(depth_msg)


def main():
    rclpy.init()
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
