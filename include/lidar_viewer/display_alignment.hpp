#pragma once

#include "lidar_viewer/imu_alignment.hpp"

namespace lidar_viewer
{
struct DisplayAlignment
{
  float yaw_deg{0}, pitch_deg{0}, roll_deg{0}, z_m{0};
};

struct ResolvedDisplayAlignment
{
  DisplayAlignment effective;
  bool imu_active{false};
};

inline ResolvedDisplayAlignment ResolveDisplayAlignment(const DisplayAlignment & manual,
  bool imu_requested, const ImuLevelingResult & imu)
{
  ResolvedDisplayAlignment result{manual,imu_requested && imu.applied};
  if (result.imu_active) {
    result.effective.pitch_deg=imu.pitch_deg;
    result.effective.roll_deg=imu.roll_deg;
  }
  return result;
}

inline void ResetDisplayRotation(DisplayAlignment & manual, bool & imu_requested)
{
  manual.yaw_deg=manual.pitch_deg=manual.roll_deg=0;
  imu_requested=false;
}
}  // namespace lidar_viewer
