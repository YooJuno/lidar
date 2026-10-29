#pragma once

#include "lidar_viewer/algorithms/clustering.hpp"
#include "lidar_viewer/algorithms/detection.hpp"
#include "lidar_viewer/algorithms/ground_removal.hpp"

namespace lidar_viewer::algorithms
{
struct PipelineConfig
{
  GroundRemovalParams ground_removal{};
  ClusteringParams clustering{};
  DetectionParams detection{};
};

struct PipelineOutput
{
  GroundRemovalResult ground_removal{};
  std::vector<Cluster> clusters;
};

// Synchronous geometry stages. LearnedDetector handles the full-scene neural
// detector independently; a cluster is never a semantic classification.
PipelineOutput RunGeometryPipeline(
  const std::vector<KittiPoint> & input,
  const PipelineConfig & config);
}  // namespace lidar_viewer::algorithms
