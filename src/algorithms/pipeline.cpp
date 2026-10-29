#include "lidar_viewer/algorithms/pipeline.hpp"

namespace lidar_viewer::algorithms
{
PipelineOutput RunGeometryPipeline(
  const std::vector<KittiPoint> & input,
  const PipelineConfig & config)
{
  PipelineOutput output;
  output.ground_removal = RunGroundSegmentation(input, config.ground_removal);
  output.clusters = RunClustering(output.ground_removal.non_ground_points, config.clustering);
  // Learned detection consumes the full aligned scene asynchronously through
  // LearnedDetector. Geometry alone must never manufacture semantic labels.
  return output;
}
}  // namespace lidar_viewer::algorithms
