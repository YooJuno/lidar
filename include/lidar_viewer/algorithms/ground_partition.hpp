#pragma once
#include "lidar_viewer/algorithms/ground_removal.hpp"
#include <algorithm>
#include <stdexcept>

namespace lidar_viewer::algorithms
{
// Preserve source order and attributes without reserving two full scans.
inline GroundRemovalResult PartitionGround(const std::vector<KittiPoint> & input,
  const std::vector<unsigned char> & mask)
{
  if (mask.size()!=input.size()) throw std::invalid_argument("Ground mask size differs from the scan");
  const auto count=static_cast<std::size_t>(std::count_if(mask.begin(),mask.end(),[](auto v){return v!=0;}));
  GroundRemovalResult result;result.ground_points.reserve(count);result.non_ground_points.reserve(input.size()-count);
  for (std::size_t i=0;i<input.size();++i)
    (mask[i] ? result.ground_points : result.non_ground_points).push_back(input[i]);
  return result;
}
}  // namespace lidar_viewer::algorithms
