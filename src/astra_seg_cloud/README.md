# astra_seg_cloud

把 Astra 深度相机的 RGB-D 流，变成**带跨帧稳定 ID 的逐实例点云**，并原样落成 rosbag2，供后续离线处理。

```
/camera/color/image_raw  (320x240, rgb8)  ┐
                                          ├─ ApproximateTimeSynchronizer (slop 50ms)
/camera/depth/image_raw  (320x240, 16UC1) ┘
                    │ 生产者：只入队，不推理
              有界队列（满则丢最旧）
                    │ 消费者：独立线程
        YOLOv8s-seg ONNX 推理 → mask / 类别 / 置信度
                    │
        mask 腐蚀 3x3 → 与 depth 逐像素配对 → 内参反投影
                    │
        x=(u-159.5)·z/285.1711, y=(v-119.5)·z/285.1711, z=depth/1000
                    │
        IoU 跟踪分配跨帧稳定 ID
                    │
        /astra_seg/objects (ObjectClouds) ──┐
        /astra_seg/color   (Image)         ─┴─→ rosbag2
```

## 环境

| 项 | 值 |
|---|---|
| ROS | Humble (`/opt/ros/humble`) |
| Python | 3.10（系统解释器，rclpy/cv_bridge 来自 apt） |
| 推理 | `~/Astra_ws/.venv`：numpy 1.26.4 + onnxruntime-gpu 1.19.2（CUDA 12 / cuDNN 9） |
| 模型 | `~/astra_models/yolov8s-seg.onnx`（640×640，opset 12） |
| 相机 | Astra，`astra_mini.launch.py`，`depth_align: true` |

> **为什么有 `.venv`**：`~/.local` 里装了 NumPy 2.2.6，会顶掉系统的 NumPy 1.21，
> 导致 ROS Humble 的 `cv_bridge` 直接 `_ARRAY_API not found`。
> 本包在 `entry.py` 里自动摘除 user site、挂上 `.venv`，所以 `ros2 run` 直接可用，
> 不需要你改 `PYTHONPATH` 或卸载任何东西。

## 构建

```bash
cd ~/Astra_ws
colcon build --symlink-install --packages-select astra_seg_interfaces astra_seg_cloud
source install/setup.bash
```

首次若模型不存在，先导出（CPU 版 torch，约 1GB 临时空间）：

```bash
bash src/astra_seg_cloud/scripts/export_model.sh
```

## 运行

```bash
# 终端 1：相机
source /opt/ros/humble/setup.bash
source ~/Astra_ws/install/setup.bash
ros2 launch astra_camera astra_mini.launch.py

# 终端 2：本节点
source /opt/ros/humble/setup.bash
source ~/Astra_ws/install/setup.bash
ros2 run astra_seg_cloud astra_seg_cloud
```

或用 launch（默认录包到 `~/bags/astra_seg_<时间戳>`）：

```bash
ros2 launch astra_seg_cloud astra_seg_cloud.launch.py
ros2 launch astra_seg_cloud astra_seg_cloud.launch.py bag_path:=/tmp/my_bag
ros2 launch astra_seg_cloud astra_seg_cloud.launch.py record_bag:=false
```

不接相机时用合成数据自测（会自动平移图片以检验 ID 稳定性）：

```bash
ros2 run astra_seg_cloud seg_cloud_test_pub \
  --ros-args -p image_path:=~/astra_models/test_bus.jpg
ros2 run astra_seg_cloud astra_seg_cloud --ros-args -p max_frames:=60
```

## 输出

### 话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/astra_seg/objects` | `astra_seg_interfaces/msg/ObjectClouds` | 每帧一条，含所有实例点云 |
| `/astra_seg/color` | `sensor_msgs/msg/Image` | 同帧 RGB（仅写入 bag） |
| `/astra_seg/depth` | `sensor_msgs/msg/Image` | 可选，`save_depth:=true` 时写入 |

### ObjectClouds

```
std_msgs/Header header
uint32[] ids            # 跨帧稳定 ID
string[] labels         # COCO 类别名
float32[] scores
sensor_msgs/PointCloud2[] clouds   # 字段 x/y/z/rgb(float32 打包)，米，camera optical frame
```

同一帧的 RGB 与 ObjectClouds **共享同一个 header.stamp**，包里也按同一时间戳成对写入，回放时无需再做时间配对。

### 读包

```bash
ros2 bag info /tmp/final_bag

# 统计 / 导出 PLY
ros2 run astra_seg_cloud seg_cloud_bag_tool --bag /tmp/final_bag
ros2 run astra_seg_cloud seg_cloud_bag_tool --bag /tmp/final_bag --export ~/bag_ply
```

Python 里直接读：

```python
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from astra_seg_interfaces.msg import ObjectClouds
import numpy as np

r = SequentialReader()
r.open(StorageOptions(uri='/tmp/final_bag', storage_id='sqlite3'),
       ConverterOptions('cdr', 'cdr'))
while r.has_next():
    topic, data, ts = r.read_next()
    if topic.endswith('/objects'):
        m = deserialize_message(data, ObjectClouds)
        for tid, label, cloud in zip(m.ids, m.labels, m.clouds):
            xyz = np.frombuffer(bytes(cloud.data), np.float32).reshape(-1, 4)[:, :3]
            print(tid, label, xyz.shape)
```

## 参数（`config/params.yaml`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_path` | `~/astra_models/yolov8s-seg.onnx` | |
| `conf_thres` / `iou_thres` | 0.25 / 0.45 | 置信度与 NMS |
| `fx, fy, cx, cy` | 285.1711 / 159.5 / 119.5 | 与相机 `camera_info` 一致 |
| `erode_kernel` | 3 | mask 腐蚀核；细长物体建议 3，别开大 |
| `min_points` | 30 | 少于该点数丢弃该实例 |
| `min_depth` / `max_depth` | 0.15 / 6.0 | 米，范围外丢弃 |
| `pixel_stride` | 1 | 2 可把点数降到 1/4 |
| `sync_slop` | 0.05 | 近似同步容差（秒） |
| `queue_size` | 4 | 生产-消费队列，满则丢最旧 |
| `track_iou` / `track_max_age` | 0.3 / 15 | 跟踪阈值与 ID 回收帧数 |
| `record_bag` / `bag_dir` / `bag_path` | true / `~/bags` / 空 | |
| `save_depth` | false | 是否把深度图也写进包 |
| `max_frames` | 0 | >0 时处理 N 帧后自动退出 |
| `debug_detail` | false | 打印每个实例的分段耗时 |

## 实测（RTX 4060 Laptop，320×240，真实相机）

| 指标 | 值 |
|---|---|
| 吞吐 | 15–16 fps（相机彩色 15fps，无积压） |
| 单帧延迟 | 27–38 ms（推理 ~22ms、点云 ~4ms、落包 ~2ms） |
| 丢帧 | 0 |
| 120 帧包 | 71 MiB，263 个实例点云，29 万点 |

CPU-only 时推理约 86ms/帧（≈11fps），仍可跑但会持续丢帧，建议保持 GPU。

## 已知问题与设计取舍

1. **相机把 `/camera/color/image_raw` 注册了两个 publisher**（同节点、不同 GID）。
   每帧会收到两份同时间戳消息，节点内按时间戳去重，不会重复处理。
2. **彩色流 `rgb8`、15fps；深度流 `16UC1`、30fps。** 两者时间戳有几十毫秒抖动，
   用 `ApproximateTimeSynchronizer(0.05s)` 配对；实测每个彩色帧最近的深度帧偏差
   均值 0.2ms、最大 25ms，可接受。运动快的场景可调小 `sync_slop`。
3. **`msg.data` 不能直接赋 `bytes`。** rclpy 会把 `uint8[]` 逐元素转换，3 万点要 30–100ms；
   本包用 `array.array('B', ...)`，同样数据降到 0.5ms。这是本项目最大的性能坑。
4. **启动有约 3 秒预热**（CUDA 上下文 + 图优化），已在 `__init__` 里显式
   `warmup()`，避免首帧把队列灌满。
5. 深度为 0 的像素（遮挡、超量程）在反投影时被丢弃；腐蚀后 mask 为空会退回原 mask，
   防止细长目标整条丢失。
6. **ID 稳定性**：IoU + 速度预测的贪心匹配，12–15fps 下够用；
   快速移动或长时间遮挡会换 ID。需要更强可换 ByteTrack/BoT-SORT。

## 自检

```bash
# 不启动 ROS：验证 ONNX 解码 + 几何（z=2m、像素可逆、腐蚀生效）
ros2 run astra_seg_cloud seg_cloud_offline_check ~/astra_models/test_bus.jpg
# 结果写到 /tmp/seg_overlay.jpg
```
