"""读取本节点产出的 rosbag，做统计或按帧导出 PLY 点云。

    # 只看统计
    ros2 run astra_seg_cloud seg_cloud_bag_tool --bag /tmp/final_bag

    # 导出每帧每个实例的 PLY（文件名含帧号、id、类别）
    ros2 run astra_seg_cloud seg_cloud_bag_tool --bag /tmp/final_bag \
        --export ~/bag_ply --limit 50
"""

import argparse
import os

from astra_seg_cloud.entry import _prepend_venv, _strip_user_site

_strip_user_site()
_prepend_venv()

import numpy as np  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # noqa: E402

from astra_seg_interfaces.msg import ObjectClouds  # noqa: E402


def read_frames(uri):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=uri, storage_id='sqlite3'),
                ConverterOptions('cdr', 'cdr'))
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic.endswith('/objects'):
            yield ts, deserialize_message(data, ObjectClouds)


def cloud_to_arrays(cloud):
    arr = np.frombuffer(bytes(cloud.data), dtype=np.float32).reshape(-1, 4)
    xyz = arr[:, :3]
    packed = arr[:, 3].view(np.uint32)
    rgb = np.stack([(packed >> 16) & 0xFF,
                    (packed >> 8) & 0xFF,
                    packed & 0xFF], axis=1).astype(np.uint8)
    return xyz, rgb


def write_ply(path, xyz, rgb):
    n = xyz.shape[0]
    head = (f'ply\nformat binary_little_endian 1.0\n'
            f'element vertex {n}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n').encode('ascii')
    body = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    body['x'], body['y'], body['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    body['red'], body['green'], body['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(head)
        f.write(body.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--export', default='')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    frames = 0
    total_clouds = 0
    total_points = 0
    ids = {}
    for idx, (ts, msg) in enumerate(read_frames(args.bag)):
        frames += 1
        for tid, label, cloud in zip(msg.ids, msg.labels, msg.clouds):
            total_clouds += 1
            total_points += cloud.width
            ids.setdefault(int(tid), [0, idx])
            ids[int(tid)][0] += 1
            if args.export and (not args.limit or idx < args.limit):
                os.makedirs(args.export, exist_ok=True)
                xyz, rgb = cloud_to_arrays(cloud)
                name = f'frame{idx:05d}_id{int(tid):03d}_{label}.ply'
                write_ply(os.path.join(args.export, name), xyz, rgb)
        if args.limit and idx + 1 >= args.limit and not args.export:
            break

    print(f'帧数: {frames}  实例点云: {total_clouds}  总点数: {total_points}')
    print(f'轨迹 id 数: {len(ids)}')
    for tid, (cnt, first) in sorted(ids.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f'  id={tid:<4d} 出现 {cnt} 帧, 首帧 {first}')
    if args.export:
        print(f'PLY 已导出到: {args.export}')


if __name__ == '__main__':
    main()
