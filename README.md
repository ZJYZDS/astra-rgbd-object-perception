# mono_cube_pose

Monocular 6-DOF cube pose estimation for RoboCon 2025.  
Lightweight ROS Noetic package — single YOLO segmentation mask in, full 3D cube pose out.

## Pipeline

```
RealSense D455i (RGB+D)
        │
        ▼
  YOLOv8-seg ──► mask
        │
        ▼
  erode → hexagon fit → 3 interior sample points
        │
        ▼
  per-point PCA normal (17×17 neighborhood)
        │
        ▼
  cosine clustering → top / front face normals
        │
        ▼
  cube constraint (35 cm side) → full 6-DOF (R, t)
        │
        ▼
  ROS PoseStamped + serial UART output
```

**Approach**: Instead of processing the full point cloud, we use the hexagon mask geometry to pick 3 interior points guaranteed to lie on single faces — then compute normals only in small neighborhoods around them. Combined with the cube side-length constraint, a single detected face suffices for full 6-DOF.

## Install

```bash
cd ~/catkin_ws/src
git clone https://github.com/ZJYZDS/KFS_grab.git
# Python deps: ultralytics, pyrealsense2, torch, numpy, opencv-python
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

## Usage

```bash
# Launch with default config
roslaunch mono_cube_pose kfs_grab.launch

# Or override individual parameters
rosrun mono_cube_pose kfs_grab.py _cube/side_mm:=300 _yolo/conf:=0.7
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/kfs_pose` | `geometry_msgs/PoseStamped` | 6-DOF cube pose (position + quaternion) |
| `/center_coord` | `std_msgs/Float32MultiArray` | Cube center in camera frame [mm] |

## Configuration

All parameters in `config/default.yaml`, loaded via rosparam at launch.

| Namespace | Key | Default | Description |
|-----------|-----|---------|-------------|
| `~camera/` | fx, fy, cx, cy | 321.6, ... | Fallback intrinsics (auto-updated from camera) |
| `~yolo/` | model_path | — | YOLOv8-seg weights path |
| | img_sz | 640 | Inference image size |
| | conf | 0.5 | Detection confidence threshold |
| | iou | 0.5 | NMS IoU threshold |
| `~serial/` | port | /dev/ttyUSB0 | UART device path |
| | baudrate | 115200 | Baud rate |
| `~mask/` | erode_kernel | 5 | Erosion kernel size |
| | erode_iterations | 2 | Erosion passes |
| | polygon_epsilon_factor | 0.015 | approxPolyDP precision |
| `~normal/` | sample_half_window | 8 | PCA neighborhood half-size (17×17) |
| | min_valid_points | 30 | Minimum valid depth points per sample |
| | cluster_cosine_threshold | 0.6 | Normal clustering cosine threshold |
| `~depth/` | min_mm | 0 | Valid depth minimum [mm] |
| | max_mm | 3000 | Valid depth maximum [mm] |
| `~cube/` | side_mm | 350.0 | Cube side length [mm] |
| `~z_filter/` | buffer_size | 5 | Rolling median buffer size |
| | anomaly_threshold_mm | 300 | Z anomaly rejection threshold |
| `~axis_mapping/` | offset_x/y/z_mm | 0,0,0 | Camera→base_link translation offsets |

## Serial Protocol

Little-endian, 3×float32 (x, y, z in mm):

```
0x0B | float32 x | float32 y | float32 z | 0x1A
```

## Hardware

- Intel RealSense D455i (RGB + depth, 640×480 @30Hz)
- YOLOv8-seg model (`best.pt`)
- Serial UART to MCU (`/dev/ttyUSB0`)

## License

MIT
