#include "lidar_viewer/algorithms/clustering.hpp"
#include "lidar_viewer/algorithms/ground_removal.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

void RunAlgorithmRegressionTests();

namespace
{
using namespace lidar_viewer::algorithms;

void AddObject(std::vector<KittiPoint> & points, float center_x, float center_y)
{
  for (int x = -3; x <= 3; ++x) {
    for (int y = -3; y <= 3; ++y) {
      for (int z = 0; z < 4; ++z) {
        points.push_back({
          center_x + 0.08f * static_cast<float>(x),
          center_y + 0.08f * static_cast<float>(y),
          0.45f + 0.35f * static_cast<float>(z),
          1.0f});
      }
    }
  }
}
}  // namespace

int main()
{
  std::vector<KittiPoint> cloud;
  for (int x = -80; x <= 80; ++x) {
    for (int y = -80; y <= 80; ++y) {
      const float px = 0.25f * static_cast<float>(x);
      const float py = 0.25f * static_cast<float>(y);
      cloud.push_back({px, py, 0.025f * px, 0.0f});
    }
  }
  AddObject(cloud, 8.0f, 3.0f);

  GroundRemovalParams ground;
  ground.enabled = true;
  ground.mode = "patchworkpp";
  ground.max_candidate_radius_m = 40.0f;
  ground.max_candidate_z = 3.0f;
  ground.patchwork_min_range_m = 0.0f;
  ground.patchwork_min_points_per_patch = 6;
  ground.patchwork_num_sectors = 24;
  const GroundRemovalResult ground_result = RunGroundRemovalPlaceholder(cloud, ground);
  if (ground_result.ground_points.size() < 20000 || ground_result.non_ground_points.size() < 150) {
    std::cerr << "Patchwork++ smoke test failed: ground=" << ground_result.ground_points.size()
              << " non-ground=" << ground_result.non_ground_points.size() << '\n';
    return EXIT_FAILURE;
  }

  GroundRemovalParams gseg3d = ground;
  gseg3d.mode = "gseg3d";
  gseg3d.use_cuda = true;
  gseg3d.gseg3d_cell_size_x_m = 1.0f;
  gseg3d.gseg3d_cell_size_y_m = 1.0f;
  gseg3d.gseg3d_cell_size_z_phase1_m = 10.0f;
  gseg3d.gseg3d_cell_size_z_phase2_m = 1.0f;
  gseg3d.gseg3d_ground_inlier_threshold_m = 0.20f;
  const GroundRemovalResult gseg3d_result = RunGroundRemovalPlaceholder(cloud, gseg3d);
  if (gseg3d_result.ground_points.empty() || gseg3d_result.non_ground_points.size() < 150 ||
      gseg3d_result.ground_points.size() + gseg3d_result.non_ground_points.size() != cloud.size()) {
    std::cerr << "GSeg3D smoke test failed: ground=" << gseg3d_result.ground_points.size()
              << " non-ground=" << gseg3d_result.non_ground_points.size() << '\n';
    return EXIT_FAILURE;
  }

  std::vector<KittiPoint> objects;
  AddObject(objects, 10.0f, 0.0f);
  AddObject(objects, 11.2f, 0.0f);  // Separate objects: gap exceeds tolerance.
  ClusteringParams clustering;
  clustering.enabled = true;
  clustering.tolerance_m = 0.45f;
  clustering.max_tolerance_m = 0.45f;
  clustering.range_tolerance_scale = 0.0f;
  clustering.min_cluster_size = 10;
  clustering.min_height_m = 0.2f;
  const auto clusters = RunClusteringEuclidean(objects, clustering);
  if (clusters.size() != 2) {
    std::cerr << "Euclidean clustering smoke test failed: clusters=" << clusters.size() << '\n';
    return EXIT_FAILURE;
  }

  try {
    RunAlgorithmRegressionTests();
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
