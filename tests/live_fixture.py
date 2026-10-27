#!/usr/bin/env python3
"""Synthetic LiDAR/IMU publisher; use only a private test ROS domain."""
import math
import struct
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField, Imu
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


def main():
    rclpy.init()
    node = Node("lidar_viewer_synthetic_fixture")
    points = node.create_publisher(PointCloud2, "/fixture/points", qos_profile_sensor_data)
    imu = node.create_publisher(Imu, "/fixture/imu", qos_profile_sensor_data)
    tf = node.create_publisher(TFMessage, "/tf_static", QoSProfile(depth=10,
        durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
    transform = TransformStamped()
    transform.header.frame_id = "pandar_xt32"
    transform.child_frame_id = "vn100_link_ned"
    transform.transform.rotation.x = 1.
    tf.publish(TFMessage(transforms=[transform]))
    started = time.monotonic()

    def send_imu():
        sample = Imu()
        sample.header.stamp = node.get_clock().now().to_msg()
        sample.header.frame_id = "vn100_link_ned"
        sample.orientation_covariance[0] = -1.
        sample.linear_acceleration.z = -9.80665
        imu.publish(sample)

    def send_cloud():
        t = time.monotonic() - started
        shift = 4 * math.sin(t * 1.7)
        cloud = PointCloud2()
        cloud.header.stamp = node.get_clock().now().to_msg()
        cloud.header.frame_id = "pandar_xt32"
        samples = [(x*.3, y*.3, -.57, 20.) for x in range(-20, 21) for y in range(-20, 21)]
        samples += [(shift+x*.08, 2+y*.08, z*.08, 70.+z*10.)
            for x in range(-5, 6) for y in range(-5, 6) for z in range(8)]
        cloud.height = 1
        cloud.width = len(samples)
        cloud.fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(["x", "y", "z", "intensity"])]
        cloud.point_step = 16
        cloud.row_step = cloud.width * cloud.point_step
        cloud.data = b"".join(struct.pack("<ffff", *sample) for sample in samples)
        points.publish(cloud)

    node.create_timer(.005, send_imu)
    node.create_timer(.05, send_cloud)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
