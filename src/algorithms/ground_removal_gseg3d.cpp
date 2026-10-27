#include "lidar_viewer/algorithms/ground_removal.hpp"

#include <ground_detection.hpp>

#include <Eigen/Geometry>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lidar_viewer::algorithms
{
namespace
{
struct PointKey
{
  std::uint32_t x{};
  std::uint32_t y{};
  std::uint32_t z{};
  std::uint32_t intensity{};

  bool operator==(const PointKey & other) const
  {
    return x == other.x && y == other.y && z == other.z && intensity == other.intensity;
  }
};

class PclWarningSilencer
{
public:
  PclWarningSilencer()
  : previous_level_(pcl::console::getVerbosityLevel())
  {
    // GSeg3D intentionally attempts plane fits in many small cells.  Some
    // valid sparse/linear cells make PCL emit a warning for every rejected
    // sample, which would otherwise flood the interactive viewer console.
    // The collinear-sample diagnostic is emitted at PCL's ERROR level; only
    // L_ALWAYS suppresses it. The algorithm handles that rejected sample and
    // continues with another one, so it is not actionable for a viewer user.
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  }

  ~PclWarningSilencer()
  {
    pcl::console::setVerbosityLevel(previous_level_);
  }

private:
  pcl::console::VERBOSITY_LEVEL previous_level_;
};

std::uint32_t FloatBits(float value)
{
  std::uint32_t bits{};
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

PointKey MakeKey(float x, float y, float z, float intensity)
{
  return {FloatBits(x), FloatBits(y), FloatBits(z), FloatBits(intensity)};
}

PointKey MakeKey(const KittiPoint & point)
{
  return MakeKey(point.x, point.y, point.z, point.intensity);
}

PointKey MakeKey(const pcl::PointXYZI & point)
{
  return MakeKey(point.x, point.y, point.z, point.intensity);
}

struct PointKeyHash
{
  std::size_t operator()(const PointKey & key) const
  {
    std::size_t hash = key.x;
    hash = hash * 16777619u ^ key.y;
    hash = hash * 16777619u ^ key.z;
    return hash * 16777619u ^ key.intensity;
  }
};

ground_segmentation::GridConfig MakeGridConfig(
  const GroundRemovalParams & params,
  float cell_size_z,
  std::uint16_t phase)
{
  ground_segmentation::GridConfig config;
  config.cellSizeX = std::max(static_cast<double>(params.gseg3d_cell_size_x_m), 0.1);
  config.cellSizeY = std::max(static_cast<double>(params.gseg3d_cell_size_y_m), 0.1);
  config.cellSizeZ = std::max(static_cast<double>(cell_size_z), 0.1);
  config.slopeThresholdDegrees = std::clamp(
    static_cast<double>(params.gseg3d_slope_threshold_deg), 0.0, 89.0);
  config.groundInlierThreshold = std::max(
    static_cast<double>(params.gseg3d_ground_inlier_threshold_m), 0.01);
  config.centroidSearchRadius = std::max(
    static_cast<double>(params.gseg3d_centroid_search_radius_m), 0.1);
  config.distToGround = params.gseg3d_ground_height_m;
  config.maxGroundHeightDeviation = std::max(
    static_cast<double>(params.gseg3d_max_ground_height_deviation_m), 0.01);
  config.processing_phase = phase;
  return config;
}

}  // namespace

GroundRemovalResult RunGroundRemovalGSeg3D(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {
    result.non_ground_points = input;
    return result;
  }

  using PointType = pcl::PointXYZI;
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
  cloud->points.reserve(input.size());
  for (const auto & point : input) {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      cloud->points.push_back({point.x, point.y, point.z, point.intensity});
    }
  }
  cloud->width = static_cast<std::uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = false;

  const Eigen::Quaterniond world_orientation = Eigen::Quaterniond::Identity();
  PclWarningSilencer silence_pcl_warnings;
  ground_segmentation::PointCloudGrid<PointType> phase_one(
    MakeGridConfig(params, params.gseg3d_cell_size_z_phase1_m, 1));
  phase_one.setInputCloud(cloud, world_orientation);
  auto phase_one_result = phase_one.segmentPoints();

  ground_segmentation::PointCloudGrid<PointType> phase_two(
    MakeGridConfig(params, params.gseg3d_cell_size_z_phase2_m, 2));
  phase_two.setInputCloud(phase_one_result.first, world_orientation);
  auto phase_two_result = phase_two.segmentPoints();

  // GSeg3D may omit cells with fewer than three points.  Reattach any such
  // original points as obstacles so this viewer's pipeline never drops data.
  std::unordered_map<PointKey, std::vector<std::size_t>, PointKeyHash> input_indices;
  input_indices.reserve(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    input_indices[MakeKey(input[index])].push_back(index);
  }
  std::vector<unsigned char> classified(input.size(), 0);

  auto append_classified = [&input, &input_indices, &classified](
      const pcl::PointCloud<PointType>::Ptr & points,
      std::vector<KittiPoint> & destination) {
      destination.reserve(destination.size() + points->size());
      for (const auto & point : points->points) {
        auto found = input_indices.find(MakeKey(point));
        if (found == input_indices.end() || found->second.empty()) {
          continue;
        }
        const std::size_t index = found->second.back();
        found->second.pop_back();
        classified[index] = 1;
        destination.push_back(input[index]);
      }
    };

  append_classified(phase_two_result.first, result.ground_points);
  append_classified(phase_one_result.second, result.non_ground_points);
  append_classified(phase_two_result.second, result.non_ground_points);
  for (std::size_t index = 0; index < input.size(); ++index) {
    if (classified[index] == 0) {
      result.non_ground_points.push_back(input[index]);
    }
  }
  return result;
}
}  // namespace lidar_viewer::algorithms
