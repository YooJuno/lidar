#pragma once

#include "lidar_viewer/data_io.hpp"

#include <cstddef>
#include <vector>

namespace lidar_viewer::algorithms
{
struct PointCloudView
{
  const std::vector<KittiPoint> * points{nullptr};
};

struct BoundingBox3D
{
  Vec3 min{};
  Vec3 max{};
};

struct OrientedBox3D
{
  Vec3 center{};
  Vec3 axis_x{1.0f, 0.0f, 0.0f};
  Vec3 axis_y{0.0f, 1.0f, 0.0f};
  float half_length{0.0f};
  float half_width{0.0f};
  float min_z{0.0f};
  float max_z{0.0f};
  bool valid{false};
};

struct Cluster
{
  std::vector<std::size_t> indices;
  BoundingBox3D bounds{};
  OrientedBox3D oriented_bounds{};
};

struct Detection
{
  BoundingBox3D bounds{};
  OrientedBox3D oriented_bounds{};
  float confidence{0.0f};
  int class_id{-1};
};
}  // namespace lidar_viewer::algorithms
