"""rosbag2 写入器：同一帧的 rgb 与 ObjectClouds 用同一个时间戳成对写入。"""

import threading

import rosbag2_py
from rclpy.serialization import serialize_message


class BagRecorder:
    def __init__(self, uri, topics, storage_id='sqlite3'):
        """topics: dict[topic_name] = msg_type_string"""
        self._lock = threading.Lock()
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'))
        for name, type_str in topics.items():
            self._writer.create_topic(rosbag2_py.TopicMetadata(
                name=name,
                type=type_str,
                serialization_format='cdr'))
        self.uri = uri
        self._closed = False

    def write(self, topic, msg, stamp_ns):
        if self._closed:
            return
        with self._lock:
            self._writer.write(topic, serialize_message(msg), int(stamp_ns))

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._writer.close()
            except Exception:
                pass
