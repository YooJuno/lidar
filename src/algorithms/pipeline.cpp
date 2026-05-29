#include "lidar_viewer/algorithms/pipeline.hpp"

namespace lidar_viewer::algorithms
{
PipelineOutput RunPipelinePlaceholder(
  const std::vector<KittiPoint> & input,
  const PipelineConfig & config)
{
  PipelineOutput output;
  output.ground_removal = RunGroundRemovalPlaceholder(input, config.ground_removal);
  output.clusters = RunClusteringPlaceholder(output.ground_removal.non_ground_points, config.clustering);
  output.detections = RunDetectionPlaceholder(
    output.ground_removal.non_ground_points,
    output.clusters,
    config.detection);
  return output;
}
}  // namespace lidar_viewer::algorithms
