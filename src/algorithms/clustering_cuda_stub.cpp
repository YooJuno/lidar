#include "lidar_viewer/algorithms/clustering_cuda.hpp"

namespace lidar_viewer::algorithms
{
bool BuildCellAssignmentsCuda(
  const std::vector<KittiPoint> &,
  const ClusteringParams &,
  std::vector<CudaCellAssignment> &)
{
  return false;
}
}  // namespace lidar_viewer::algorithms
