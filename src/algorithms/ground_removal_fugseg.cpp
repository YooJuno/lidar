#include "lidar_viewer/algorithms/ground_removal.hpp"
#include <fugseg/fugseg.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace lidar_viewer::algorithms
{
GroundRemovalResult RunGroundRemovalFugSeg(const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {result.non_ground_points=input;return result;}
  if (input.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Point cloud exceeds FugSeg index capacity");
  for (const float value:{params.fugseg_bin_size_m,params.fugseg_inlier_threshold_m,
       params.fugseg_max_slope_change_deg,params.fugseg_max_gap_m,params.max_candidate_radius_m,
       params.fugseg_horizontal_noise_deg,params.fugseg_vertical_noise_deg,params.fugseg_range_noise_m,
       params.sensor_height_m,params.gseg3d_ground_height_m})
    if (!std::isfinite(value)) throw std::invalid_argument("FugSeg parameters must be finite");
  const int segments=std::clamp(params.fugseg_num_segments,30,720);
  const float bin=std::clamp(params.fugseg_bin_size_m,.25f,3.f);
  const float inlier=std::clamp(params.fugseg_inlier_threshold_m,.02f,.5f);
  const float slope=std::clamp(params.fugseg_max_slope_change_deg,1.f,45.f);
  const float gap=std::clamp(params.fugseg_max_gap_m,1.f,30.f);
  const float range=std::clamp(params.max_candidate_radius_m,3.f,200.f);
  const std::array<float,6> key{{static_cast<float>(segments),bin,inlier,slope,gap,range}};
  struct Cache {
    std::unique_ptr<fugseg::GroundExtractor> model;
    std::array<float,6> key{};
    fugseg::Cloud cloud;
    fugseg::Cluster3D ground,others;
  };
  static thread_local Cache cache;
  if (!cache.model || cache.key!=key) {
    cache.model=std::make_unique<fugseg::GroundExtractor>(segments,bin,inlier,slope,gap,.5f,range);
    cache.key=key;
  }
  const float height=std::clamp(params.sensor_height_m,.05f,20.f);
  cache.model->ConfigureSensor(height,std::clamp(params.fugseg_horizontal_noise_deg,.001f,1.f),
    std::clamp(params.fugseg_vertical_noise_deg,.001f,1.f),std::clamp(params.fugseg_range_noise_m,.001f,.5f),input.size());
  cache.cloud.Reset(input.size());
  std::vector<std::size_t> original;original.reserve(input.size());
  for (std::size_t i=0;i<input.size();++i) {
    const auto & point=input[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
    cache.cloud.points().row(static_cast<Eigen::Index>(original.size()))<<
      point.x,point.y,point.z-params.gseg3d_ground_height_m-height,0.f;
    original.push_back(i);
  }
  cache.cloud.points().conservativeResize(static_cast<Eigen::Index>(original.size()),Eigen::NoChange);
  std::vector<unsigned char> ground(input.size(),0);
  if (!original.empty()) {
    if (!cache.model->ExtractGround(cache.cloud,cache.ground,cache.others))
      throw std::runtime_error("Original FugSeg extraction failed");
    for (const auto index:cache.ground) {
      if (index>=original.size()) throw std::runtime_error("FugSeg returned an invalid point index");
      ground[original[index]]=1;
    }
  }
  result.ground_points.reserve(input.size());result.non_ground_points.reserve(input.size());
  for (std::size_t i=0;i<input.size();++i)
    (ground[i] ? result.ground_points : result.non_ground_points).push_back(input[i]);
  return result;
}
} // namespace lidar_viewer::algorithms
