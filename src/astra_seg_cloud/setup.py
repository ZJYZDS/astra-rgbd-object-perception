import os
from glob import glob

from setuptools import setup

package_name = 'astra_seg_cloud'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zjy',
    maintainer_email='zjy@example.com',
    description='RGB-D instance segmentation -> per-object point clouds with stable IDs, recorded to rosbag2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'astra_seg_cloud = astra_seg_cloud.entry:main',
            'seg_cloud_test_pub = astra_seg_cloud.test_publisher:main',
            'seg_cloud_offline_check = astra_seg_cloud.offline_check:main',
            'seg_cloud_bag_tool = astra_seg_cloud.bag_tools:main',
        ],
    },
)
