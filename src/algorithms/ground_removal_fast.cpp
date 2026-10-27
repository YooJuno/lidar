#include "lidar_viewer/algorithms/ground_removal.hpp"
#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace lidar_viewer::algorithms
{
namespace
{
struct GroundGrid
{
  int width;
  float cell;
  float origin;
  std::vector<float> minimum;
  std::vector<unsigned char> ground;

  GroundGrid(float radius, float cell_size)
  : width(0), cell(cell_size), origin(-radius)
  {
    const double side = std::ceil((2.0f * radius) / cell) + 1;
    if (!std::isfinite(side) || side > 2048) {
      throw std::invalid_argument("Fast ground grid is too large; increase cell size or reduce radius");
    }
    width = static_cast<int>(side);
    minimum.assign(static_cast<std::size_t>(width) * width, 1.0e9f);
    ground.assign(minimum.size(), 0);
  }

  int Coord(float value) const { return static_cast<int>(std::floor((value - origin) / cell)); }
  std::size_t Index(int x, int y) const { return static_cast<std::size_t>(y) * width + x; }
  bool Contains(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < width; }

  float NearbyGround(int x, int y, int & support) const
  {
    float minimum_z = 1.0e9f;
    support = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (!Contains(x + dx, y + dy)) continue;
        const auto index = Index(x + dx, y + dy);
        if (ground[index]) {
          minimum_z = std::min(minimum_z, minimum[index]);
          ++support;
        }
      }
    }
    return minimum_z;
  }
};

bool Candidate(const KittiPoint & point, const GroundRemovalParams & params, float radius_squared)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
    point.z >= params.min_candidate_z && point.z <= params.max_candidate_z &&
    point.x * point.x + point.y * point.y <= radius_squared;
}

std::vector<unsigned char> ClassifyFastGroundCpu(
  const std::vector<KittiPoint> & input, const GroundRemovalParams & params)
{
  const float radius = std::max(params.max_candidate_radius_m, 1.0f);
  const float radius_squared = radius * radius;
  GroundGrid coarse(radius, std::max(params.gseg3d_cell_size_x_m, .25f));
  GroundGrid fine(radius, std::max(params.gseg3d_cuda_cell_size_m, .25f));
  for (const auto & point : input) {
    if (!Candidate(point, params, radius_squared)) continue;
    for (GroundGrid * grid : {&coarse, &fine}) {
      const int x = grid->Coord(point.x), y = grid->Coord(point.y);
      if (grid->Contains(x, y)) {
        auto & minimum = grid->minimum[grid->Index(x, y)];
        minimum = std::min(minimum, point.z);
      }
    }
  }

  const float seed_radius = std::max(params.gseg3d_cuda_seed_radius_m, coarse.cell);
  const float seed_tolerance = std::max(params.gseg3d_ground_inlier_threshold_m * 2, .25f);
  std::vector<std::size_t> frontier;
  for (std::size_t index = 0; index < coarse.minimum.size(); ++index) {
    if (coarse.minimum[index] >= 9.0e8f) continue;
    const float x = coarse.origin + (static_cast<float>(index % coarse.width) + .5f) * coarse.cell;
    const float y = coarse.origin + (static_cast<float>(index / coarse.width) + .5f) * coarse.cell;
    if (std::sqrt(x * x + y * y) <= seed_radius &&
      std::abs(coarse.minimum[index] - params.gseg3d_ground_height_m) <= seed_tolerance) {
      coarse.ground[index] = 1;
      frontier.push_back(index);
    }
  }

  const float angle = std::clamp(params.gseg3d_slope_threshold_deg, 0.0f, 45.0f) * kPi / 180;
  const float maximum_step = std::max(
    params.gseg3d_max_ground_height_deviation_m, std::tan(angle) * coarse.cell * 1.2f);
  const int iterations = std::min(256, static_cast<int>(std::ceil(radius / coarse.cell)) + 1);
  std::size_t begin = 0;
  // Breadth-first expansion visits each accepted cell once. Its layer limit
  // matches the simultaneous CUDA expansion without scanning empty cells.
  for (int layer = 0; layer < iterations && begin < frontier.size(); ++layer) {
    const auto end = frontier.size();
    for (; begin < end; ++begin) {
      const auto index = frontier[begin];
      const int x = static_cast<int>(index % coarse.width);
      const int y = static_cast<int>(index / coarse.width);
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (!coarse.Contains(x + dx, y + dy)) continue;
          const auto next = coarse.Index(x + dx, y + dy);
          if (!coarse.ground[next] && coarse.minimum[next] < 9.0e8f &&
            std::abs(coarse.minimum[next] - coarse.minimum[index]) <= maximum_step) {
            coarse.ground[next] = 1;
            frontier.push_back(next);
          }
        }
      }
    }
  }

  const float deviation = std::max(params.gseg3d_max_ground_height_deviation_m * 1.5f,
    params.gseg3d_ground_inlier_threshold_m);
  for (std::size_t index = 0; index < fine.minimum.size(); ++index) {
    if (fine.minimum[index] >= 9.0e8f) continue;
    const float x = fine.origin + (static_cast<float>(index % fine.width) + .5f) * fine.cell;
    const float y = fine.origin + (static_cast<float>(index / fine.width) + .5f) * fine.cell;
    int support;
    const float ground_z = coarse.NearbyGround(coarse.Coord(x), coarse.Coord(y), support);
    fine.ground[index] = support && std::abs(fine.minimum[index] - ground_z) <= deviation;
  }

  const float threshold = std::max(params.gseg3d_ground_inlier_threshold_m, .05f);
  std::vector<unsigned char> mask(input.size(), 0);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto & point = input[index];
    if (!Candidate(point, params, radius_squared)) continue;
    int support;
    const float ground_z = fine.NearbyGround(fine.Coord(point.x), fine.Coord(point.y), support);
    const int required = std::sqrt(point.x * point.x + point.y * point.y) > 25 ? 1 : 2;
    const float dz = point.z - ground_z;
    mask[index] = support >= required && dz >= -threshold && dz <= threshold;
  }
  return mask;
}
}  // namespace

GroundRemovalResult RunGroundRemovalFastGrid(
  const std::vector<KittiPoint> & input, const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {
    result.non_ground_points = input;
    return result;
  }
  std::vector<unsigned char> mask;
  const bool used_cuda = params.use_cuda && ClassifyGroundPointsGSeg3DCuda(input, params, mask);
  if (!used_cuda) mask = ClassifyFastGroundCpu(input, params);
  result.ground_points.reserve(input.size());
  result.non_ground_points.reserve(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    (mask[index] ? result.ground_points : result.non_ground_points).push_back(input[index]);
  }
  static std::atomic<bool> logged{false};
  if (params.use_cuda && !logged.exchange(true)) {
    std::cout << "Fast grid ground backend: " << (used_cuda ? "CUDA" : "CPU fallback") << '\n';
  }
  return result;
}
}  // namespace lidar_viewer::algorithms
