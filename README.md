# astra_seg_cloud — RGB-D 实例分割 → 逐实例 6DoF 点云

把 **Orbbec Astra** 的 RGB-D 流变成**带跨帧稳定 ID 的逐实例点云 + 6DoF OBB + 抓取候选**，
并把同一帧的 RGB、点云、标注图原样落成 rosbag2。

面向**桌面抓取**场景：相机固定、物体摆在桌面上、单视角。

![RViz 调试图：左 RGB+3D 框，右点云+框点云](docs/media/demo_poster.png)

📹 **演示视频**：[docs/media/demo.mp4](docs/media/demo.mp4)（32 s，已去音轨；手机翻录的 RViz 实时画面）

---

## 效果（实测，320×240 / 相机 15 fps）

| 指标 | 值 |
|---|---|
| 硬件 | Orbbec Astra（USB2），color `rgb8` 15 fps，depth `16UC1` 30 fps，`depth_align: true` |
| 模型 | YOLO26s-seg → ONNX（40 MB），单帧推理 **13.5 ms**（RTX 4060 Laptop / CUDA EP） |
| 端到端 | 1–2 个物体 **15–16 fps**；3 个物体 **10–11 fps**（含调试点云发布），延迟 60–95 ms |
| 3D 框紧度 | 键盘/书 **框内点 ≥ 99.9%**，杯子 ≈ 99%；尺寸抖动 0.02–0.2 cm |
| 典型尺寸 | 键盘 31.6×10.5×2.9 cm、马克杯 6.8×6.2×10.3 cm、书 30.7×19.4×3.1 cm |
| 跟踪 | IoU + 速度预测，静止场景单 ID 连续 150 帧 |

---

## 制作思想

### 1. 用"分割 + 几何"而不是"端到端 3D 检测"

数据只有 320×240 的单视角 RGB-D，训练一个 3D 检测网络既缺数据也缺监督。
所以拆成两段：**2D 分割（成熟、可换模型）** + **几何拟合（可解释、可调试）**。
网络只负责"哪些像素是哪个物体"，3D 位姿/尺寸交给几何，出问题能一眼定位到是哪一段。

### 2. 单视角的物理限制要"认账"，而不是硬估

| 限制 | 处理 |
|---|---|
| 只看到物体一个面 → **厚度不可观测** | 用类别先验 + `contain_points()` 让框包住实际点云 |
| 圆柱/回转体 → **绕轴 yaw 不可观测** | 不硬估：输出规范 yaw，并用 `/astra_seg/yaw_valid` 标记"此朝向无意义" |
| 有把手的杯子 → yaw 可观测 | 从"径向残差单峰"里估把手方向，可见才置有效 |
| 物体可能压在别的物体上 | 高度用**点云 z_max − z_min**（物体自身范围），不假设它一定在桌面上 |

### 3. z 轴统一，不让每个物体各自"猜"

每个物体自己 RANSAC 平面，遇到圆柱会抓到侧壁（法线水平，框直接躺倒 85°）。
现在 **z 轴统一取"桌面法线"**：由同帧平面物体（键盘/书）的平面法线求均值，
深度图最大平面兜底。这样键盘、书、杯子的 z 轴互相一致（差 3° 以内）。

### 4. 按形状分两条分支

```
cup / bottle  → 圆柱分支：桌面平面内圆拟合 → r；高度 = z_max − z_min；把手残差定 yaw
keyboard/book → 多边形分支：mask 四边形（反投影）+ 主平面 → 最小矩形定 l/w
```

### 5. 实时链路只做轻量的，重活放离线

实时节点里只保留：体素下采样 → RANSAC 平面 → 圆拟合/四边形 → 包住点云收敛。
**MLS（移动最小二乘）平滑只放在离线预处理**（`scripts/bag_preprocess.py`）：
实测把厚度离散度从 0.65 → 0.39 cm、平面 RMS 0.20 → 0.13 cm，但单实例要 34–46 ms，
放进实时会直接把 15 fps 打到 10 fps 以下。而且 MLS 会抹平"物体压在物体上"的深度台阶，
所以离线也是**先分层再平滑**。

### 6. 一切都要能在 RViz 里"看见"

框不是图像，很难判断对错。所以把 3D 框的 12 条棱**离散成点云**，
和"平面点 + 物体点"一起作为调试点云发布（`/astra_seg/debug/*`），
配合 `config/debug.rviz` 一眼看出框歪没歪、包没包住。

---

## 系统架构

```
/camera/color/image_raw (320x240 rgb8, 15fps) ┐
/camera/depth/image_raw (16UC1, 30fps)        ┤ ApproxTimeSync(0.05s)
                                              └→ 生产者：只入队（队列 4，满丢最旧）
                                                   → 消费者线程：
                                                      YOLO26s-seg → mask ∩ depth → 反投影
                                                      → 桌面平面（depth→pts→RANSAC）
                                                      → 分支拟合 OBB（圆柱 / 多边形）
                                                      → IoU 跟踪分配稳定 ID
                                                      → 抓取候选（吸盘 / 平行夹爪）
                                                   → /astra_seg/objects + /color + /annotated
                                                      + /debug/*  → rosbag2
```

### 包结构

```
src/
├── astra_seg_interfaces/        # ObjectClouds.msg
└── astra_seg_cloud/             # 本包
    ├── astra_seg_cloud/
    │   ├── entry.py             # 启动引导（摘 user-site + 挂 .venv）
    │   ├── pipeline_node.py     # 主节点：同步/队列/推理/拟合/落包/调试
    │   ├── yolo_seg.py          # ONNX 解码 / NMS / mask 按框裁剪
    │   ├── projection.py        # mask 腐蚀 + 深度反投影
    │   ├── obb.py               # 桌面平面 / OBB（圆柱·多边形）/ 抓取候选
    │   ├── pose_filter.py       # 按 track id 的位姿平滑
    │   ├── tracker.py           # IoU 跟踪
    │   ├── grasp.py             # 抓取候选与打分
    │   ├── cloud_builder.py     # PointCloud2 构造（含框棱离散）
    │   ├── annotate.py          # 画框 / 三轴 / 底部文字条
    │   └── bag_recorder.py      # rosbag2 写入（同帧同 stamp）
    ├── config/params.yaml       # 全部参数
    ├── config/debug.rviz        # RViz 调试布局
    └── scripts/                 # 离线工具
        ├── bag_to_video.py      # bag → mp4
        ├── make_demo_video.py   # 左 RGB+框 / 右点云+框 合成视频
        ├── cloud_obb_tiles.py   # 实例点云+框 三视图
        ├── obb_axes_view.py     # RGB 上画框+三轴
        ├── bag_preprocess.py    # 离线 MLS 预处理 → 训练样本 .npz
        └── export_model.sh      # 导出 ONNX
```

### 消息

```
std_msgs/Header header
uint32[] ids                              # 跨帧稳定 ID
string[] labels ; float32[] scores
sensor_msgs/PointCloud2[] clouds          # x/y/z/rgb（float32 打包），米，camera optical frame
geometry_msgs/Pose[]    obb_poses         # 位置=中心；朝向 x=长, y=宽, z=法线
geometry_msgs/Vector3[] obb_dims          # l, w, h
geometry_msgs/Point[]   obb_corners       # 每实例 8 角点，扁平排列
geometry_msgs/Pose[]    grasp_poses       # z=接近方向，x=闭合方向
uint8[] grasp_types ; float32[] grasp_widths ; float32[] grasp_scores
```

调试话题（`debug_cloud: true` 时发布）：

| 话题 | 内容 |
|---|---|
| `/astra_seg/debug/plane` | 桌面平面内点（灰） |
| `/astra_seg/debug/objects` | 各物体点云（真实 RGB） |
| `/astra_seg/debug/boxes` | 3D 框 12 条棱离散点（每物体一色） |
| `/astra_seg/debug/all` | 平面 + 物体 + 框（一条话题看全部） |
| `/astra_seg/yaw_valid` | `UInt8MultiArray`，1=该实例 yaw 可信 |

---

## 快速开始

### 依赖

| 项 | 版本 |
|---|---|
| ROS 2 | Humble |
| Python | 3.10（系统解释器；`rclpy`/`cv_bridge` 来自 apt） |
| 推理环境 | `~/Astra_ws/.venv`：`numpy==1.26.4` + `onnxruntime-gpu==1.19.2`（CUDA 12 / cuDNN 9） |
| 模型 | `~/astra_models/yolo26s-seg.onnx`（`scripts/export_model.sh` 可重导） |
| 相机驱动 | `astra_camera`（Obreec 开源驱动，编译进同一 workspace） |

> **为什么有 `.venv`**：`~/.local` 里若装了 NumPy 2.x，会顶掉系统 1.21，导致 `cv_bridge` 报
> `_ARRAY_API not found`。`entry.py` 会自动摘除 user-site 并挂上 `.venv`，所以 `ros2 run` 直接可用。

### 编译 & 运行

```bash
cd ~/Astra_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select astra_seg_interfaces astra_seg_cloud

# 终端 1：相机
ros2 launch astra_camera astra_mini.launch.py

# 终端 2：本节点（默认录包到 ~/bags/astra_seg_<时间戳>）
ros2 run astra_seg_cloud astra_seg_cloud
```

### 调试（RViz）

```bash
ros2 run astra_seg_cloud astra_seg_cloud --ros-args \
  --params-file ~/Astra_ws/src/astra_seg_cloud/config/params.yaml -p record_bag:=false

rviz2 -d ~/Astra_ws/src/astra_seg_cloud/config/debug.rviz
```

左侧 `annotated_rgb` 面板是 RGB+框+三轴+文字条，右侧是点云+框点云。

### 常用工具

```bash
# 无相机自测
ros2 run astra_seg_cloud seg_cloud_test_pub --ros-args -p image_path:=~/astra_models/test_bus.jpg
ros2 run astra_seg_cloud seg_cloud_offline_check ~/astra_models/test_bus.jpg

# 读包 / 出图 / 出视频
ros2 run astra_seg_cloud seg_cloud_bag_tool --bag ~/bags/rec52
python3 scripts/bag_to_video.py      ~/bags/rec52 --scale 2
python3 scripts/make_demo_video.py   ~/bags/rec52 /tmp/demo.mp4
python3 scripts/cloud_obb_tiles.py   ~/bags/rec52
python3 scripts/bag_preprocess.py    ~/bags/rec52 --out ~/grasp_ds/v1
```

---

## 关键参数（`config/params.yaml`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_path` | `~/astra_models/yolo26s-seg.onnx` | 回退：`yolov8s-seg.onnx` + `conf 0.15` |
| `conf_thres` / `iou_thres` | 0.25 / 0.45 | |
| `target_classes` | `cup, keyboard, book, bottle` | 白名单 |
| `cylinder_classes` | `cup, bottle` | 走圆柱分支 |
| `roi_enable` / `roi_margin` | true / 0.10 | 只收画面中央 80% 的检测，挡边缘干扰 |
| `table_stride` / `table_max_jump_deg` | 6 / 8.0 | 桌面平面采样步长 / 相邻帧法线最大跳变 |
| `obb_margin` / `pose_smooth_alpha` | 0.001 / 0.30 | 外扩余量 / 位姿平滑 |
| `debug_cloud` / `draw_grasp` | true / false | 调试点云 / 图上画抓取标记 |

---

## 调试过程（AI 辅助，摘要）

整个项目是**人定方向 + AI 写代码/做实验**推进的。有效的做法只有一个：
**用数据定位，不靠感觉判断**。几条典型的"现象 → 根因 → 修法"：

| 现象 | 怎么定位的 | 根因 | 修法 |
|---|---|---|---|
| 键盘点云横跨 1 m | 把点云反投影回图像 | YOLOv8-seg 的 mask 是 prototype 线性组合，**没按检测框裁剪** | mask 只在框内取 |
| 单帧延迟 96 ms | 分段计时 | `msg.data = bytes` 让 rclpy 逐元素转换 3 万点 | 改 `array.array('B')`（→0.5 ms） |
| 框总是短一截 | 打印框外点比例 | `fit_obb` 某分支**提前 return**，标定没执行 | 收敛到 `contain_points()` |
| 偶发框歪 5.9° | 对比 `mask_quad` 是否被用上 | 参数名 `mask` 与内点变量撞名，传进来的 mask 从未使用 | 改名 `image_mask` |
| 杯子框躺倒 85° | 比较各实例 z 轴 | 圆柱被 RANSAC 抓到侧壁当主平面 | z 统一用桌面法线 |
| 桌面法线偶尔歪 37° | 逐帧打印法线夹角 | 深度图最大平面偶尔锁到背景 | 加"朝上先验 + 相邻帧 8° 防跳" |
| 杯子框顶矮 2 cm | 框内点比例 + 点云 z 分布 | 我把高度从 `z_max−z_min` 改成 `p5~p95`，**修了一个不存在的 bug** | 回到 `z_max−z_min` |
| 调试视频右栏全黑 | 打印 rgb 字段 | PointCloud2 的 rgb 是 float32 **位模式**，错用了数值转换 | `view(uint32)` 而非 `astype` |

教训也说清楚：
1. **"框内点 100%" 是弱指标**——它只证明没有点露在框外，不证明框紧、框正；
2. **不要拿 mask 去对齐框**，那是"按答案改答案"，会把误差藏起来；
3. 3D 框底面的近角投影本来就比物体接触线低几像素，**那是几何必然，不是 bug**。

---

## 已知限制

- 相机 320×240 是硬上限（640×480 时彩色掉到 1.2 fps，不可用）；深度有效距离 0.8–1.3 m；
- 深色/低对比的物体（黑键盘侧视、深色书封面）YOLO 分数会掉到 0.15 左右；
- 被完全遮住的部分不可见，框靠外轮廓 + 先验估计；
- 一帧内实例越多越慢（OBB 逐实例计算），3 个物体时约 10–11 fps。

## Roadmap

- [ ] 抓取**位置**交给学习网络（方式由类别查表，网络只出位置，OBB 提供宽度/朝向）
- [ ] 数据集：`bag_preprocess.py` 输出 `.npz`（点云/法线/分层/6DoF 标签）
- [ ] 几何基线（距离变换）→ 弱标签训练 → PyBullet 真标签
- [ ] 吸盘局部平面度判据（键帽缝隙场景）
- [ ] YOLO 微调（自有场景 200–300 帧）

## 致谢 / 许可

- 相机驱动：[orbbec/astra_camera](https://github.com/orbbec/astra_camera)（不包含在本仓库）
- 分割模型：Ultralytics YOLO26s-seg（AGPL-3.0，注意商用许可）
- 本仓库代码：MIT（见 `LICENSE`）
