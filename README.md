# astra_seg_cloud

**Orbbec Astra RGB-D → 逐实例点云 + 6DoF OBB + 抓取候选**（容器：ROS 2 Humble）。
分割用**通用 COCO 预训练模型**（`yolo26s-seg` ONNX），换场景只改 `target_classes`，不需要自采数据。

📹 演示：[docs/media/demo.mp4](docs/media/demo.mp4)（54 s，无音轨，720p 1.2 Mbps）
左：RViz（桌面平面 / 每实例点云 / 3D 框）右：RGB 叠加框与实时字幕（尺寸/yaw/抓取）

## 效果（320×240，3 个物体）

| 项 | 值 |
|---|---|
| 端到端 | **15 fps / 52 ms**（含 RViz 调试点云；相机本身 15 fps 是上限） |
| 推理 / 点云阶段 | 21 ms / 30 ms（桌面 RANSAC 9 ms + 逐物体拟合 8.6 ms） |
| 3D 框 | keyboard 31.0×11.9×2.8 cm（抖动 0.10）、book 25.6×20.7×2.7（抖动 0.06） |
| 出现率 | 两物体均 122/122 帧，不闪不消失；`quality=1` 比例 100% |

## 算法

```
整帧深度 → 反投影(stride 3) → passthrough → 体素
  → RANSAC 去桌面（6 mm；桌面点单独留给 RViz，并从场景删除）
  → 检测按 mask 内中位深度 远→近 排序
  → 每个物体：mask_k -= 更近物体的 mask
             sel = mask_k ∩ 场景
             true_pts = RANSAC 平面(4 mm) 内的点
             场景 -= true_pts        # 只删这些
             OBB = 多边形 / 圆柱分支
```

要点：只删 `true_pts`（删整个 mask 会把后面物体删没）；圆柱不套平面裁剪（曲面）；
桌面判定校验与上一帧法线(≤10°)/偏移(≤3 cm)一致；圆柱 yaw 用桌面法线的规范基（yaw 不可观测）。

**几何与质量的三个开关**

| 项 | 默认 | 说明 |
|---|---|---|
| `trim_mode` | `plane` | 4 mm 平面裁剪（实测最贴）。`grow` = 体素邻域区域生长，抗污染强（合成测试 0% vs 95.7% 污染点）但框偏松 +1.8 cm |
| `size_prior_enable` | `false` | 类别尺寸先验（中位数+MAD，3σ 外才替换）。会把偏松的尺寸学成"正常"，默认关 |
| `quality` | 逐实例 | 可解性门控：认领率/法线残差/平面内点比/圆柱长宽比/遮挡比例越界 → `0`（框画灰，下游别抓） |
| `label_vote_decay` | 0.85 | 按 track 类别投票，分数低于 `min_label_score` 直接记 `object`，避免 book↔cell phone 抖动 |

## 调试记录

| 现象（实测） | 根因 | 修法 |
|---|---|---|
| 键盘点云横跨 1 m | mask 未按检测框裁剪 | 只在框内取 mask |
| 单帧 96 ms | `msg.data = bytes` 逐元素转换 3 万点 | `array.array('B')` → 0.5 ms |
| 框外点 16.9% | `fit_obb` 分支提前 return，包点云没执行 | 收敛到 `contain_points()` |
| 框偶发歪 5.9° | 参数 `mask` 与内点变量撞名，传入 mask 未使用 | 改名 `image_mask` |
| 杯子框躺倒 85° | 圆柱被 RANSAC 抓侧壁当主平面 | z 统一用桌面法线 |
| 桌面法线偶歪 37° | 最大平面偶尔锁到背景 | 朝上先验 + 相邻帧 8° 防跳 |
| 杯子框顶矮 2 cm | 高度从 `z_max−z_min` 改成 `p5~p95`（修了不存在的 bug） | 回到 `z_max−z_min` |
| 键盘高 7.4（真实 3.0） | 新链路漏 `clean_by_plane()` | 补回（4 mm 严格阈值） |
| 高度算出 73 cm | 平面偏移传 `-base`，应为 `d_table − base` | 修正偏移 |
| 场景模式框仍被覆盖 | 第二遍"支撑面"覆盖新结果 | scene 模式禁用第二遍 |
| 键盘 71/122 帧消失 | RANSAC 把键盘顶面当桌面并把点从场景删掉 | 被拒平面只排除搜索范围 + 法线/偏移一致性 |
| 杯子时隐时现 | 圆柱套了平面裁剪（只剩 60 点）、拟合门槛 80 | 圆柱跳过裁剪 + 门槛 35 |
| 杯子框乱转（138.8°） | 用圆柱自身轴做规范基（轴在晃）+ ±n 有 180° 歧义 | 桌面法线 + 方形 footprint → 0.22° |
| 帧率 8.7（曾 15） | ORT/OpenBLAS 按核数(20)开线程并自旋，占 1800% CPU | 线程限 1 + 关自旋 → 15 fps |
| RViz 一开就崩宿主 | 从 agent shell 启动 Qt，继承环境/会话 | `systemd-run --user --collect` + 干净环境 |
| RViz 里 RGB 面板空白 | RViz 持久化了"显示项被取消勾选" | 清 `~/.rviz2` 后重启 |
| `book` 时有时无（7/152） | COCO 对这本教材打分只有 0.14–0.27，在 book/cell phone/remote 之间跳 | 白名单加 `cell phone` + 按 track 投票 + 低分归 `object` → 151/151 |
| `quality` 全 0（框全灰） | 认领率分母用了 mask 像素数，深色物体大量像素无深度 | 分母改成 mask 内**有深度的场景点**数 → 100% 恢复 |
| 框内点跑到框外 | `filter_to_obb` 只对 `source='plane'` 生效，quad 分支没裁 | 显示点云自己按框裁一次 |
| `_NamedDet() takes 3 args but 5 given` | 引入包装类后 `type(d)(...)` 构造出错 | 显式 `Detection(...)` 再包一层 |
| 框跑到画面边缘、检测全被 ROI 丢掉 | **相机驱动被启动了两次**（同话题 2 个发布者，帧交错） | 清掉多余实例，只留一个 |
| 55 ms 里有 18 ms 波动 | 桌面 RANSAC 最多跑 3 次（80 迭代） | 迭代 80→40、重试 +2→+1 → 9 ms |
| 四边形拟合每次都跑优化 | 初值（对边平均）通常已够好 | 初值残差 < 2 mm 直接用；否则 `max_nfev` 50→25 |

判据：**"框内点 100%" 是弱指标**（不证明框紧），要看 `框尺寸 vs 点云范围` 与真值；
**不要拿 2D mask 去对齐 3D 框**；框底面近角的投影本来就比接触线低几像素（几何必然）。

## 话题

| 话题 | 方向 | 类型 | 作用 |
|---|---|---|---|
| `/camera/color/image_raw` | 订阅 | `Image` | 彩色 320×240 rgb8，15 fps |
| `/camera/depth/image_raw` | 订阅 | `Image` | 深度 16UC1，30 fps（已与彩色对齐） |
| `/astra_seg/objects` | 发布 | `ObjectClouds` | **核心输出**：每实例点云 + 6DoF OBB + 抓取候选 |
| `/astra_seg/annotated` | 发布 | `Image` | RGB + 3D 框 + 三轴 + 字幕（调试模式下叠加 mask） |
| `/astra_seg/color` | 发布 | `Image` | 落包用的原始彩色帧 |
| `/astra_seg/depth` | 发布 | `Image` | 可选落包（`save_depth: true`） |
| `/astra_seg/yaw_valid` | 发布 | `UInt8MultiArray` | 每个实例的 yaw 是否可信（1/0） |

`ObjectClouds` 字段：`ids/labels/scores`、`clouds`（xyz+rgb 点云）、`obb_poses`（位置+四元数）、
`obb_dims`（l,w,h）、`obb_corners`（8 角点）、`grasp_poses/types/widths/scores`、
`quality`（1=可信 / 0=可疑，下游据此决定抓不抓）。
所有点云与位姿都在 `camera_color_optical_frame` 下（RViz 的 Fixed Frame 就设它）。

### 调试话题（`debug_cloud: true`）

| 话题 | 内容 | 用来查什么 |
|---|---|---|
| `/astra_seg/debug/plane` | 桌面平面内点（灰，≤2500 点） | 去桌面是否干净、桌面法线是否歪 |
| `/astra_seg/debug/objects` | 各实例**认领到的**点云（真实 RGB） | 点归属对不对（有没有混入邻居的点） |
| `/astra_seg/debug/boxes` | 3D 框 12 条棱离散点（4 mm，每实例一色） | 框是否贴合、朝向是否合理 |
| `/astra_seg/debug/all` | 上面三者合并成一条 | 一个 `PointCloud2` 看全场景 |

## 运行

```bash
colcon build --symlink-install --packages-select astra_seg_interfaces astra_seg_cloud
ros2 launch astra_camera astra_mini.launch.py
ros2 run astra_seg_cloud astra_seg_cloud --ros-args \
  --params-file ~/Astra_ws/src/astra_seg_cloud/config/params.yaml \
  -p target_classes:="['cup','keyboard','book','bottle']" \
  -p scene_pipeline:=true -p record_bag:=false

# RViz（本机必须这样起，否则会崩宿主）
systemd-run --user --collect --unit=rviz-gui \
  --setenv=DISPLAY=:1 --setenv=XAUTHORITY=$HOME/.Xauthority \
  --setenv=XDG_RUNTIME_DIR=/run/user/1000 --setenv=NO_AT_BRIDGE=1 \
  /bin/bash -lc 'source /opt/ros/humble/setup.bash; source ~/Astra_ws/install/setup.bash; \
                 exec rviz2 -d ~/Astra_ws/src/astra_seg_cloud/config/debug.rviz'
```

主要参数：`scene_pipeline`（远→近认领）、`scene_stride`（3）、`plane_clean_thresh`（4 mm）、
`debug_mode`（才在 RGB 上叠加 mask）、`caption_lines`（字幕固定 5 行，防图像高度跳变）、
`trim_mode`（plane/grow）、`size_prior_enable`、`tighten_to_cloud`、`occ_prior_thr`/`occ_quality_thr`（遮挡）、
`label_vote_decay`/`min_label_score`/`unknown_score`（标签稳定）。

## 真值与评测

物体**尺寸**可以直接量；**实时位姿 GT** 用 ArUco（`DICT_4X4_50`）：

```bash
python3 src/astra_seg_cloud/scripts/make_aruco_pdf.py --out ~/gt_aruco --size-cm 12   # 生成送印 PDF
python3 src/astra_seg_cloud/scripts/pose_gt_aruco.py --selftest                      # 合成图自测
python3 src/astra_seg_cloud/scripts/pose_gt_aruco.py --config src/astra_seg_cloud/config/gt_markers.yaml
```

关键点：**平面标记用 6DoF PnP 有二义性**（角点噪声 0.3 px 时旋转误差会跳到 25–34°），
必须加"标记贴在已知平面"的约束（只解 yaw+平移）。实测单帧精度：

| marker 边长 | 0.8 m | 0.9 m | 1.1 m |
|---|---|---|---|
| 10 cm | 8.9 mm / 0.62° | 11.2 mm / 0.70° | 16.8 mm / 0.86° |
| 12 cm | 7.4 mm / 0.52° | 9.4 mm / 0.58° | 14.0 mm / 0.71° |

静止物体取 100 帧中位数 → ~1 mm / 0.1°。注意 **marker 贴物体上会干扰检测**：
要做"干净"的精度验证，用 L 形载板（object 与 marker 同板、marker 偏在物体外）或直接手工量 GT。

另外 `scripts/test_trim_methods.py` 是合成场景单测：构造已知盒子 + 污染面，
比较两种裁剪的保留率/污染率（当前：平面裁剪 3.3% 真点 / 95.7% 污染点，
区域生长 85.8% / 0.0%）。

## 限制

相机 320×240 为硬上限；深度有效 0.8–1.3 m，反光表面有飞点；单视角被遮挡部分不可见（厚度/朝向靠先验）。

## 许可

MIT。相机驱动 [orbbec/astra_camera](https://github.com/orbbec/astra_camera) 不在本仓库；模型来自 Ultralytics（AGPL-3.0）。
