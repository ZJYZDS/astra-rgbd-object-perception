import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('astra_seg_cloud')

    args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'params.yaml')),
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.expanduser('~/astra_models/yolov8s-seg.onnx')),
        DeclareLaunchArgument(
            'bag_path',
            default_value='',
            description='留空则自动写到 ~/bags/astra_seg_<时间戳>'),
        DeclareLaunchArgument('record_bag', default_value='true'),
        DeclareLaunchArgument('max_frames', default_value='0'),
    ]

    node = Node(
        package='astra_seg_cloud',
        executable='astra_seg_cloud',
        name='astra_seg_cloud',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'model_path': LaunchConfiguration('model_path'),
                'bag_path': LaunchConfiguration('bag_path'),
                'record_bag': LaunchConfiguration('record_bag'),
                'max_frames': LaunchConfiguration('max_frames'),
            },
        ],
    )

    return LaunchDescription(args + [node])
