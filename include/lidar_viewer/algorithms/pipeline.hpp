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
  std::vector<Detection> detections;
};

PipelineOutput RunPipelinePlaceholder(
  const std::vector<KittiPoint> & input,
  const PipelineConfig & config);
}  // namespace lidar_viewer::algorithms
