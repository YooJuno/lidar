#pragma once

#include "lidar_viewer/algorithms/types.hpp"

#include <string>

namespace lidar_viewer::algorithms
{
struct ClusteringParams
{
  bool enabled{false};
  bool use_cuda{false};
  std::string mode{"euclidean"};
  float tolerance_m{0.55f};
  float grid_cell_m{0.25f};
  // At longer ranges the horizontal sampling interval increases.  This
  // bounded expansion prevents one object from being split without joining
  // neighbouring objects at close range.
  float range_tolerance_scale{0.006f};
  float max_tolerance_m{1.15f};
  float max_vertical_gap_m{2.5f};
  float min_height_m{0.2f};
  float max_height_m{5.0f};
  float min_width_m{0.1f};
  float max_width_m{8.0f};
  float min_length_m{0.1f};
  float max_length_m{20.0f};
  int min_cluster_size{8};
  int max_cluster_size{20000};
};

class ClusteringAlgorithm
{
public:
  virtual ~ClusteringAlgorithm() = default;
  virtual std::vector<Cluster> Run(
    const std::vector<KittiPoint> & input,
    const ClusteringParams & params) const = 0;
};

std::vector<Cluster> RunClusteringPlaceholder(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params);

std::vector<Cluster> RunClusteringEuclidean(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params);
}  // namespace lidar_viewer::algorithms
