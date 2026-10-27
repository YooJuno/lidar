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
  // fast_grid uses the same classification rules on CPU and CUDA. gseg3d
  // selects the original third-party CPU implementation for comparison.
  std::string mode{"fast_grid"};
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

  // Patchwork++-style concentric zone model (CZM).  Each radial patch gets
  // its own low-point seeds and locally fitted ground plane, so slopes and
  // crowned/uneven roads are not forced onto one global plane.
  int patchwork_num_zones{4};
  int patchwork_num_sectors{32};
  int patchwork_min_points_per_patch{12};
  int patchwork_seed_count{20};
  int patchwork_refine_iterations{3};
  float patchwork_min_range_m{2.0f};
  float patchwork_seed_threshold_m{0.25f};
  float patchwork_distance_threshold_m{0.18f};
  float patchwork_max_flatness_m2{0.06f};

  // GSeg3D: two-pass 3D grid segmentation.  The viewer calibrates points
  // into its level world frame before this algorithm runs, so the default
  // ground height is zero and no ROS/IMU transform is required here.
  float gseg3d_cell_size_x_m{1.0f};
  float gseg3d_cell_size_y_m{1.0f};
  float gseg3d_cell_size_z_phase1_m{10.0f};
  float gseg3d_cell_size_z_phase2_m{1.0f};
  float gseg3d_slope_threshold_deg{20.0f};
  float gseg3d_ground_inlier_threshold_m{0.20f};
  float gseg3d_centroid_search_radius_m{5.0f};
  float gseg3d_ground_height_m{0.0f};
  float gseg3d_max_ground_height_deviation_m{0.10f};
  float gseg3d_cuda_seed_radius_m{15.0f};
  float gseg3d_cuda_cell_size_m{0.50f};
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

GroundRemovalResult RunGroundRemovalPatchworkPP(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);

GroundRemovalResult RunGroundRemovalGSeg3D(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);

GroundRemovalResult RunGroundRemovalFastGrid(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);

GroundRemovalResult RunGroundRemovalPlaceholder(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params);
}  // namespace lidar_viewer::algorithms
