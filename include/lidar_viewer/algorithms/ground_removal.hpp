#pragma once

#include "lidar_viewer/algorithms/types.hpp"

#include <string>
#include <vector>

namespace lidar_viewer::algorithms
{
struct GroundRemovalParams
{
  bool enabled{false};
  bool use_cuda{false};
  std::string mode{"ransac"};
  float distance_threshold_m{0.18f};
  float final_distance_threshold_m{0.20f};
  float local_ground_margin_m{0.18f};
  float max_above_plane_m{0.22f};
  float max_below_plane_m{0.45f};
  float adaptive_threshold_scale{0.0025f};
  float max_adaptive_threshold_m{0.36f};
  float max_ground_angle_deg{15.0f};
  float min_candidate_z{-3.0f};
  float max_candidate_z{1.5f};
  float max_candidate_radius_m{160.0f};
  float candidate_grid_cell_m{0.8f};
  float grid_max_height_step_m{0.22f};
  float grid_max_local_slope_m_per_m{0.18f};
  float grid_seed_radius_m{12.0f};
  float grid_max_vertical_extent_m{0.35f};
  int max_iterations{240};
  int min_inliers{80};
};

struct GroundRemovalResult
{
  std::vector<KittiPoint> ground_points;
  std::vector<KittiPoint> non_ground_points;
};

class GroundRemovalAlgorithm
{
public:
  virtual ~GroundRemovalAlgorithm() = default;
  virtual GroundRemovalResult Run(
    const std::vector<KittiPoint> & input,
    const GroundRemovalParams & params) const = 0;
};

GroundRemovalResult RunGroundRemovalRansac(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);

GroundRemovalResult RunGroundRemovalGrid(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);

GroundRemovalResult RunGroundRemovalPlaceholder(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);
}  // namespace lidar_viewer::algorithms
