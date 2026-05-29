#pragma once

#include "lidar_viewer/algorithms/clustering.hpp"

#include <cstdint>
#include <vector>

namespace lidar_viewer::algorithms
{
struct CudaCellAssignment
{
  std::int64_t key{0};
  std::size_t point_index{0};
};

bool BuildCellAssignmentsCuda(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params,
  std::vector<CudaCellAssignment> & assignments);
}  // namespace lidar_viewer::algorithms
