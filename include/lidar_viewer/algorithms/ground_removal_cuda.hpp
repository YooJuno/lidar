#pragma once

#include "lidar_viewer/algorithms/ground_removal.hpp"

#include <vector>

namespace lidar_viewer::algorithms
{
struct CudaPlaneModel
{
  float a{0.0f};
  float b{0.0f};
  float c{1.0f};
  float d{0.0f};
};

bool ScorePlanesCuda(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  const std::vector<CudaPlaneModel> & planes,
  float threshold,
  std::vector<int> & inlier_counts);

bool BuildMinZGridCuda(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  std::vector<float> & min_z_grid,
  std::vector<unsigned char> & valid_grid);

bool ClassifyGroundPointsGridCuda(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  const std::vector<float> & accepted_ground_z_grid,
  const std::vector<unsigned char> & accepted_ground_valid_grid,
  std::vector<unsigned char> & is_ground_mask);
}  // namespace lidar_viewer::algorithms
