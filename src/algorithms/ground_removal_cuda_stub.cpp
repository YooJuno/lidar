#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"

namespace lidar_viewer::algorithms
{
bool ScorePlanesCuda(
  const std::vector<KittiPoint> &,
  const std::vector<std::size_t> &,
  const std::vector<CudaPlaneModel> &,
  float,
  std::vector<int> &)
{
  return false;
}

bool BuildMinZGridCuda(
  const std::vector<KittiPoint> &,
  const GroundRemovalParams &,
  float,
  float,
  int,
  int,
  float,
  std::vector<float> &,
  std::vector<unsigned char> &)
{
  return false;
}

bool ClassifyGroundPointsGridCuda(
  const std::vector<KittiPoint> &,
  const GroundRemovalParams &,
  float,
  float,
  int,
  int,
  float,
  const std::vector<float> &,
  const std::vector<unsigned char> &,
  std::vector<unsigned char> &)
{
  return false;
}
}  // namespace lidar_viewer::algorithms
