#include "lidar_viewer/algorithms/ground_removal.hpp"
#include <patchwork/patchworkpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace lidar_viewer::algorithms
{
GroundRemovalResult RunGroundRemovalPatchworkPP(const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {result.non_ground_points=input;return result;}
  if (input.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Point cloud exceeds Patchwork++ index capacity");
  patchwork::Params options;
  options.verbose=false;
  // Intensity is normalized per frame in this viewer, so the original raw
  // reflectivity threshold cannot be applied faithfully. Keep geometry-based
  // R-VPF, A-GLE and TGR enabled; preserve every unclassified point as obstacle.
  options.enable_RNR=false;
  options.sensor_height=std::clamp(static_cast<double>(params.sensor_height_m),.05,20.);
  options.min_range=std::max(0.,static_cast<double>(params.patchwork_min_range_m));
  options.max_range=std::max(options.min_range+1.,static_cast<double>(params.max_candidate_radius_m));
  options.num_iter=std::clamp(params.patchwork_refine_iterations,1,20);
  options.num_lpr=std::max(3,params.patchwork_seed_count);
  options.num_min_pts=std::max(3,params.patchwork_min_points_per_patch);
  options.th_seeds=std::max(.01,static_cast<double>(params.patchwork_seed_threshold_m));
  options.th_dist=std::max(.01,static_cast<double>(params.patchwork_distance_threshold_m));
  options.uprightness_thr=std::cos(std::clamp(static_cast<double>(params.max_ground_angle_deg),0.,60.)*kPi/180.);
  const double sectors=std::clamp(params.patchwork_num_sectors,4,128)/32.;
  options.num_sectors_each_zone={std::max(4,static_cast<int>(16*sectors)),std::max(4,static_cast<int>(32*sectors)),
    std::max(4,static_cast<int>(54*sectors)),std::max(4,static_cast<int>(32*sectors))};
  const std::array<double,11> key{{options.sensor_height,options.min_range,options.max_range,
    static_cast<double>(options.num_iter),static_cast<double>(options.num_lpr),static_cast<double>(options.num_min_pts),
    options.th_seeds,options.th_dist,options.uprightness_thr,sectors,params.gseg3d_ground_height_m}};
  struct Cache {
    std::unique_ptr<patchwork::PatchWorkpp> model;
    std::array<double,11> key{};
    std::uint64_t sequence{0};
  };
  static thread_local Cache cache;
  if (!cache.model || cache.key!=key || cache.sequence!=params.sequence_id) {
    cache.model=std::make_unique<patchwork::PatchWorkpp>(options);
    cache.key=key;cache.sequence=params.sequence_id;
  }
  Eigen::MatrixXf cloud(static_cast<Eigen::Index>(input.size()),3);
  std::vector<std::size_t> original;
  original.reserve(input.size());
  for (std::size_t i=0;i<input.size();++i) {
    const auto & point=input[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
    const auto row=static_cast<Eigen::Index>(original.size());
    cloud.row(row)<<point.x,point.y,point.z-params.gseg3d_ground_height_m-options.sensor_height;
    original.push_back(i);
  }
  cloud.conservativeResize(static_cast<Eigen::Index>(original.size()),Eigen::NoChange);
  std::vector<unsigned char> ground(input.size(),0);
  if (!original.empty()) {
    cache.model->estimateGround(std::move(cloud));
    const auto indices=cache.model->getGroundIndices();
    for (Eigen::Index i=0;i<indices.size();++i) {
      const int index=indices[i];
      if (index>=0 && static_cast<std::size_t>(index)<original.size()) ground[original[index]]=1;
    }
  }
  result.ground_points.reserve(input.size());result.non_ground_points.reserve(input.size());
  for (std::size_t i=0;i<input.size();++i)
    (ground[i] ? result.ground_points : result.non_ground_points).push_back(input[i]);
  return result;
}
}  // namespace lidar_viewer::algorithms
