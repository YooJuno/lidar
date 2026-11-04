#pragma once

#include "lidar_viewer/data_io.hpp"
#include <algorithm>
#include <cmath>

namespace lidar_viewer
{
inline constexpr float kMinTopViewHalfHeight=.1f;
inline float TopViewHalfHeight(float distance) {return std::clamp(distance,kMinTopViewHalfHeight,5000.f);}
inline Vec3 TopViewUp(float yaw) { return {-std::sin(yaw), std::cos(yaw), 0}; }
inline Vec3 PanTopView(const Vec3 & target, float yaw, float dx, float dy, float scale)
{
  const Vec3 right{std::cos(yaw), std::sin(yaw), 0};
  return target - right * (dx * scale) + TopViewUp(yaw) * (dy * scale);
}
inline float RotateTopView(float yaw, float delta) { return std::remainder(yaw + delta, 2 * kPi); }
}  // namespace lidar_viewer
