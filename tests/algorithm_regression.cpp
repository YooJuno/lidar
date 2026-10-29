#include "lidar_viewer/algorithms/clustering.hpp"
#include "lidar_viewer/algorithms/ground_removal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
using namespace lidar_viewer::algorithms;
using Membership = std::vector<std::vector<std::size_t>>;

void Check(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

Membership Canonical(const std::vector<Cluster> & clusters)
{
  Membership result;
  for (const auto & cluster : clusters) {
    result.push_back(cluster.indices);
    std::sort(result.back().begin(), result.back().end());
  }
  std::sort(result.begin(), result.end());
  return result;
}

Membership Exhaustive(const std::vector<KittiPoint> & points, const ClusteringParams & params)
{
  Membership result;
  std::vector<bool> visited(points.size(), false);
  for (std::size_t start = 0; start < points.size(); ++start) {
    if (visited[start]) continue;
    std::vector<std::size_t> component{start};
    visited[start] = true;
    for (std::size_t next = 0; next < component.size(); ++next) {
      const auto & a = points[component[next]];
      for (std::size_t i = 0; i < points.size(); ++i) {
        const auto & b = points[i];
        const float range = std::max(std::sqrt(a.x * a.x + a.y * a.y),
          std::sqrt(b.x * b.x + b.y * b.y));
        const float tolerance = std::min(params.max_tolerance_m,
          params.tolerance_m + params.range_tolerance_scale * range);
        const float dx = a.x - b.x, dy = a.y - b.y;
        if (!visited[i] && std::abs(a.z - b.z) <= params.max_vertical_gap_m &&
          dx * dx + dy * dy <= tolerance * tolerance) {
          visited[i] = true;
          component.push_back(i);
        }
      }
    }
    std::sort(component.begin(), component.end());
    result.push_back(std::move(component));
  }
  std::sort(result.begin(), result.end());
  return result;
}

ClusteringParams Unfiltered()
{
  ClusteringParams params;
  params.enabled = true;
  params.min_cluster_size = 1;
  params.max_cluster_size = 100000;
  params.min_height_m = params.min_width_m = params.min_length_m = 0;
  params.max_height_m = params.max_width_m = params.max_length_m = 100000;
  return params;
}

void TestRotatedBoxes()
{
  for (const int degrees : {0, 15, 45, 90, 135, -90}) {
    std::vector<KittiPoint> cloud;
    const float angle = degrees * kPi / 180;
    for (int x = -20; x <= 20; ++x) {
      for (int y = -4; y <= 4; ++y) {
        for (int z = 0; z < 3; ++z) {
          cloud.push_back({-30 + .25f * (x * std::cos(angle) - y * std::sin(angle)),
            -20 + .25f * (x * std::sin(angle) + y * std::cos(angle)), z * 1.5f, 1});
        }
      }
    }
    ClusteringParams params;
    params.enabled = true;
    const auto clusters = RunClusteringEuclidean(cloud, params);
    Check(clusters.size() == 1, "Rotating a long object changed the size-filter result");
    const auto & box = clusters.front().oriented_bounds;
    Check(box.valid && std::abs(box.half_length - 5) < .001f &&
      std::abs(box.half_width - 1) < .001f, "Oriented length/width are not canonical");
    for (const auto & point : cloud) {
      const float dx = point.x - box.center.x, dy = point.y - box.center.y;
      Check(std::abs(dx * box.axis_x.x + dy * box.axis_x.y) <= box.half_length + .001f &&
        std::abs(dx * box.axis_y.x + dy * box.axis_y.y) <= box.half_width + .001f,
        "Canonical axes no longer contain the object's points");
    }
  }
}

void TestConnectivity()
{
  std::mt19937 rng(421);
  std::uniform_real_distribution<float> xy(-3, 3), z(0, 3);
  auto params = Unfiltered();
  params.max_vertical_gap_m = .4f;
  for (int trial = 0; trial < 12; ++trial) {
    params.grid_cell_m = .1f + .05f * (trial % 5);
    params.tolerance_m = .2f + .05f * trial;
    params.max_tolerance_m = 1.5f;
    params.range_tolerance_scale = .006f;
    const float offset = trial % 2 ? 80.0f : 0.0f;
    std::vector<KittiPoint> cloud;
    for (int i = 0; i < 180; ++i) cloud.push_back({xy(rng) - offset, xy(rng) + offset, z(rng), 1});
    const auto expected = Exhaustive(cloud, params);
    for (const bool cuda : {false, true}) {
      params.use_cuda = cuda;
      Check(Canonical(RunClusteringEuclidean(cloud, params)) == expected,
        "Spatial-hash clustering differs from exhaustive point connectivity");
    }
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto invalid = RunClusteringEuclidean({{nan, 0, 0, 1}, {1.0e30f, 0, 0, 1},
    {-2, -2, 0, 1}, {-2, -2, 1, 1}}, params);
  for (const auto & cluster : invalid) {
    for (const auto index : cluster.indices) Check(index >= 2, "Invalid sample reached a cluster");
  }
  std::vector<KittiPoint> dense;
  for (int i = 0; i < 20000; ++i) dense.push_back({-.1f + i % 100 * .001f, -.1f, i % 3 * .01f, 1});
  params.max_vertical_gap_m = 2.5f;
  const auto clusters = RunClusteringEuclidean(dense, params);
  Check(clusters.size() == 1 && clusters.front().indices.size() == dense.size(),
    "Removing visited candidates lost points in a dense component");
}

bool SamePoints(const std::vector<KittiPoint> & a, const std::vector<KittiPoint> & b)
{
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const auto & x, const auto & y) {
    return x.x == y.x && x.y == y.y && x.z == y.z && x.intensity == y.intensity;
  });
}

void TestFastGroundBackends()
{
  for (const float slope : {0.0f, .025f, .1f}) {
    std::vector<KittiPoint> cloud;
    for (int x = -80; x <= 80; ++x) {
      for (int y = -80; y <= 80; ++y) cloud.push_back({x * .25f, y * .25f, slope * x * .25f, 0});
    }
    for (int z = 0; z < 20; ++z) cloud.push_back({8, 3, slope * 8 + .5f + z * .1f, 1});
    GroundRemovalParams params;
    params.enabled = true;
    params.mode = "fast_grid";
    params.max_candidate_radius_m = 40;
    params.max_candidate_z = 3;
    const auto cpu = RunGroundSegmentation(cloud, params);
    Check(cpu.ground_points.size() > 25000 && cpu.non_ground_points.size() >= 20,
      "Fast grid lost the sloped road or classified the raised obstacle as ground");
    params.use_cuda = true;
    const auto gpu = RunGroundSegmentation(cloud, params);
    Check(SamePoints(cpu.ground_points, gpu.ground_points) &&
      SamePoints(cpu.non_ground_points, gpu.non_ground_points), "Fast grid CPU/CUDA classification differs");
    // Exercise workspace reuse after the input size shrinks.
    cloud.resize(8000);
    params.use_cuda = false;
    const auto small_cpu = RunGroundSegmentation(cloud, params);
    params.use_cuda = true;
    const auto small_gpu = RunGroundSegmentation(cloud, params);
    Check(SamePoints(small_cpu.ground_points, small_gpu.ground_points) &&
      SamePoints(small_cpu.non_ground_points, small_gpu.non_ground_points), "Reused CUDA workspace retained stale data");
    // Sliders also resize the grids, independently of the number of points.
    params.gseg3d_cell_size_x_m = .7f;
    params.gseg3d_cuda_cell_size_m = .35f;
    params.use_cuda = false;
    const auto resized_cpu = RunGroundSegmentation(cloud, params);
    params.use_cuda = true;
    const auto resized_gpu = RunGroundSegmentation(cloud, params);
    Check(SamePoints(resized_cpu.ground_points, resized_gpu.ground_points) &&
      SamePoints(resized_cpu.non_ground_points, resized_gpu.non_ground_points), "Resized CPU/CUDA grids differ");
  }
}

void TestGroundQuality()
{
  for (const float slope : {0.f,.12f}) {
    std::vector<KittiPoint> cloud;
    std::size_t truth_ground=0;
    for (int x=-72;x<=72;++x) for (int y=-72;y<=72;++y) {
      const float px=x*.25f,py=y*.25f;
      if (px*px+py*py<4) continue;
      cloud.push_back({px,py,slope*px,0});++truth_ground;
    }
    for (int x=0;x<20;++x) for (int y=0;y<10;++y) for (int z=0;z<10;++z) {
      const float px=7+x*.08f;
      cloud.push_back({px,3+y*.08f,slope*px+.45f+z*.15f,1});
    }
    cloud.push_back(cloud.front());++truth_ground;
    cloud.push_back({std::numeric_limits<float>::quiet_NaN(),0,0,1});
    for (const auto * mode : {"fast_grid","gseg3d","patchworkpp","grid","ransac"}) {
      GroundRemovalParams params;params.enabled=true;params.mode=mode;params.max_candidate_radius_m=40;
      params.max_candidate_z=4;params.patchwork_min_points_per_patch=6;
      params.patchwork_min_range_m=0;params.sequence_id=slope==0 ? 101 : 102;
      const auto result=RunGroundSegmentation(cloud,params);
      std::size_t true_positive=0,false_positive=0;
      for (const auto & point : result.ground_points) (point.intensity==0 ? true_positive : false_positive)++;
      const double precision=double(true_positive)/std::max<std::size_t>(1,true_positive+false_positive);
      const double recall=double(true_positive)/truth_ground;
      std::cout<<"Ground quality "<<mode<<" slope="<<slope<<" precision="<<precision<<" recall="<<recall<<'\n';
      Check(result.ground_points.size()+result.non_ground_points.size()==cloud.size(),"Ground segmentation dropped input points");
      Check(precision>=.98 && recall>=.85,"Ground segmentation failed labeled flat/ramp + raised-obstacle regression");
    }
  }
}
}  // namespace

void RunAlgorithmRegressionTests()
{
  TestRotatedBoxes();
  TestConnectivity();
  TestFastGroundBackends();
  TestGroundQuality();
  bool rejected=false;
  try {GroundRemovalParams invalid;invalid.mode="unknown";RunGroundSegmentation({},invalid);}
  catch (const std::invalid_argument &) {rejected=true;}
  Check(rejected,"Unknown ground method silently selected another algorithm");
}
