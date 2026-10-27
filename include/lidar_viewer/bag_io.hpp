#pragma once

#include "lidar_viewer/data_io.hpp"

#ifdef LIDAR_VIEWER_HAS_ROSBAG
#include <sensor_msgs/msg/point_cloud2.hpp>
#endif

// Offline storage access only. These functions never create a ROS node,
// participate in DDS discovery, or publish recorded robot commands.
bool IsRos2BagInput(const fs::path & input);
bool RosbagInputAvailable();
std::vector<FrameHandle> ResolveBagFrames(
  const fs::path & input, const std::string & topic = "/b2/points", const std::string & imu_topic = "");
FrameData LoadBagFrame(const FrameHandle & handle);

#ifdef LIDAR_VIEWER_HAS_ROSBAG
// Decode declared PointField offsets, row padding, and byte order rather than
// reinterpreting storage as an aligned C++ point struct.
FrameData DecodePointCloud2(const sensor_msgs::msg::PointCloud2 & cloud);
#endif
