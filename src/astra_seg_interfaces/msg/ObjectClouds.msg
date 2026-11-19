# 同一帧 RGB 图像中所有实例的分割点云 + 位姿 + 抓取建议。
# 一条消息 = 一帧，rgb 与各实例点云/位姿因此天然绑定，回放时无需再做时间戳配对。
std_msgs/Header header

# 每个实例的跟踪 ID（跨帧稳定，同一物体在连续帧中保持不变）
uint32[] ids

# 类别名与置信度，长度与 ids 一致
string[] labels
float32[] scores

# 每个实例一个点云：sensor_msgs/PointCloud2，字段 x/y/z/rgb(float32 打包)
# 坐标系为相机光学系（camera optical frame），单位米
sensor_msgs/PointCloud2[] clouds

# ---------------- 6DoF / OBB ----------------
# OBB 位姿：位置为 OBB 中心，朝向为 OBB 三轴（x=长, y=宽, z=表面法线方向）
geometry_msgs/Pose[] obb_poses

# OBB 尺寸：l, w, h（米），x/y/z 分别对应 obb_poses 的局部轴
geometry_msgs/Vector3[] obb_dims

# 每个实例 8 个 OBB 角点，按 ids 顺序扁平排列（8 × 实例数）
geometry_msgs/Point[] obb_corners

# ---------------- 抓取 ----------------
# 抓取位姿：z 轴为接近方向，x 轴为闭合方向（吸盘时无意义）
geometry_msgs/Pose[] grasp_poses

# 抓取方式：0=不可抓, 1=吸盘, 2=平行夹爪
uint8[] grasp_types

# 夹爪需要的开口宽度（米）；吸盘时为吸盘直径
float32[] grasp_widths

# 抓取质量分 0~1，越低越不可靠
float32[] grasp_scores

# ---------------- 质量门控 ----------------
# 每个实例的几何质量：1=可信（可用于抓取），0=可疑（点归属/平面/噪声越界）
# 由管线内部的"可解性 gate"给出，不参与几何拟合，只做下游过滤
uint8[] quality
