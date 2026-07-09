cmake_minimum_required(VERSION 3.0.2)
project(mono_cube_pose)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/kfs_grab.py
  scripts/utils/kfs_pointcloud.py
  scripts/utils/normal_utils.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
