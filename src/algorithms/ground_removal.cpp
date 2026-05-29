#include "lidar_viewer/algorithms/ground_removal.hpp"
#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <queue>
#include <unordered_map>

namespace lidar_viewer::algorithms
{
namespace
{
struct Plane
{
  float a{0.0f};
  float b{0.0f};
  float c{1.0f};
  float d{0.0f};
};

struct GridCellGround
{
  float min_z{std::numeric_limits<float>::max()};
  float max_z{std::numeric_limits<float>::lowest()};
  float radius{0.0f};
  bool valid{false};
  bool accepted_ground{false};
  float accepted_z{0.0f};
};

Vec3 Subtract(const KittiPoint & lhs, const KittiPoint & rhs)
{
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 CrossProduct(const Vec3 & lhs, const Vec3 & rhs)
{
  return {
    lhs.y * rhs.z - lhs.z * rhs.y,
    lhs.z * rhs.x - lhs.x * rhs.z,
    lhs.x * rhs.y - lhs.y * rhs.x
  };
}

float DotProduct(const Vec3 & lhs, const Vec3 & rhs)
{
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

float Length(const Vec3 & value)
{
  return std::sqrt(DotProduct(value, value));
}

bool TryMakePlane(
  const KittiPoint & p0,
  const KittiPoint & p1,
  const KittiPoint & p2,
  Plane & plane)
{
  Vec3 normal = CrossProduct(Subtract(p1, p0), Subtract(p2, p0));
  const float len = Length(normal);
  if (len < 1e-5f) {
    return false;
  }

  normal = normal * (1.0f / len);
  if (normal.z < 0.0f) {
    normal = normal * -1.0f;
  }

  plane.a = normal.x;
  plane.b = normal.y;
  plane.c = normal.z;
  plane.d = -(normal.x * p0.x + normal.y * p0.y + normal.z * p0.z);
  return true;
}

float DistanceToPlane(const Plane & plane, const KittiPoint & point)
{
  return std::abs(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d);
}

float SignedDistanceToPlane(const Plane & plane, const KittiPoint & point)
{
  return plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d;
}

bool IsCandidate(const KittiPoint & point, const GroundRemovalParams & params)
{
  if (point.z < params.min_candidate_z || point.z > params.max_candidate_z) {
    return false;
  }

  const float radius_sq = point.x * point.x + point.y * point.y;
  const float max_radius = std::max(params.max_candidate_radius_m, 1.0f);
  return radius_sq <= max_radius * max_radius;
}

bool IsWithinRadius(const KittiPoint & point, const GroundRemovalParams & params)
{
  const float radius_sq = point.x * point.x + point.y * point.y;
  const float max_radius = std::max(params.max_candidate_radius_m, 1.0f);
  return radius_sq <= max_radius * max_radius;
}

std::int64_t GridKey(const KittiPoint & point, float cell_size)
{
  const auto ix = static_cast<std::int32_t>(std::floor(point.x / cell_size));
  const auto iy = static_cast<std::int32_t>(std::floor(point.y / cell_size));
  return (static_cast<std::int64_t>(ix) << 32) ^ static_cast<std::uint32_t>(iy);
}

std::vector<std::size_t> BuildLowestPointCandidates(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  std::unordered_map<std::int64_t, std::size_t> lowest_by_cell;
  lowest_by_cell.reserve(input.size() / 8);

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (!IsCandidate(input[i], params)) {
      continue;
    }

    const std::int64_t key = GridKey(input[i], cell_size);
    auto it = lowest_by_cell.find(key);
    if (it == lowest_by_cell.end() || input[i].z < input[it->second].z) {
      lowest_by_cell[key] = i;
    }
  }

  std::vector<std::size_t> candidates;
  candidates.reserve(lowest_by_cell.size());
  for (const auto & [_, index] : lowest_by_cell) {
    candidates.push_back(index);
  }
  return candidates;
}

std::unordered_map<std::int64_t, float> BuildLocalGroundHeights(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  std::unordered_map<std::int64_t, float> lowest_z_by_cell;
  lowest_z_by_cell.reserve(input.size() / 8);

  for (const auto & point : input) {
    if (!IsCandidate(point, params)) {
      continue;
    }

    const std::int64_t key = GridKey(point, cell_size);
    auto it = lowest_z_by_cell.find(key);
    if (it == lowest_z_by_cell.end() || point.z < it->second) {
      lowest_z_by_cell[key] = point.z;
    }
  }

  return lowest_z_by_cell;
}

bool IsNearLocalGround(
  const KittiPoint & point,
  const GroundRemovalParams & params,
  const std::unordered_map<std::int64_t, float> & local_ground_heights)
{
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  const auto ix = static_cast<std::int32_t>(std::floor(point.x / cell_size));
  const auto iy = static_cast<std::int32_t>(std::floor(point.y / cell_size));

  float local_min_z = std::numeric_limits<float>::max();
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const std::int64_t key =
        (static_cast<std::int64_t>(ix + dx) << 32) ^
        static_cast<std::uint32_t>(iy + dy);
      const auto it = local_ground_heights.find(key);
      if (it != local_ground_heights.end()) {
        local_min_z = std::min(local_min_z, it->second);
      }
    }
  }

  if (local_min_z == std::numeric_limits<float>::max()) {
    return false;
  }

  return point.z <= local_min_z + std::max(params.local_ground_margin_m, 0.05f);
}

float RangeAdaptiveThreshold(
  const KittiPoint & point,
  float base_threshold,
  const GroundRemovalParams & params)
{
  const float radius = std::sqrt(point.x * point.x + point.y * point.y);
  const float distance_scaled = base_threshold + std::max(params.adaptive_threshold_scale, 0.0f) * radius;
  return std::clamp(
    std::max(base_threshold, distance_scaled),
    base_threshold,
    std::max(params.max_adaptive_threshold_m, base_threshold));
}

std::pair<int, int> GridCoord(const KittiPoint & point, float cell_size)
{
  return {
    static_cast<int>(std::floor(point.x / cell_size)),
    static_cast<int>(std::floor(point.y / cell_size))
  };
}

float CellCenterDistance(int cx, int cy, float cell_size)
{
  const float x = (static_cast<float>(cx) + 0.5f) * cell_size;
  const float y = (static_cast<float>(cy) + 0.5f) * cell_size;
  return std::sqrt(x * x + y * y);
}

GroundRemovalResult RunGroundRemovalGridInternal(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {
    result.non_ground_points = input;
    return result;
  }

  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  const float max_radius = std::max(params.max_candidate_radius_m, 1.0f);
  const float grid_min_x = -max_radius;
  const float grid_min_y = -max_radius;
  const int grid_width = std::max(1, static_cast<int>(std::ceil((2.0f * max_radius) / cell_size)) + 1);
  const int grid_height = grid_width;
  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * static_cast<std::size_t>(grid_height);

  std::vector<GridCellGround> cells(cell_count);
  std::vector<float> min_z_grid;
  std::vector<unsigned char> valid_grid;
  const bool used_cuda =
    params.use_cuda &&
    BuildMinZGridCuda(
      input,
      params,
      grid_min_x,
      grid_min_y,
      grid_width,
      grid_height,
      cell_size,
      min_z_grid,
      valid_grid);

  auto cell_index = [grid_width](int cx, int cy) {
    return static_cast<std::size_t>(cy) * static_cast<std::size_t>(grid_width) + static_cast<std::size_t>(cx);
  };

  bool has_valid_cell = false;
  for (const auto & point : input) {
    if (!IsCandidate(point, params)) {
      continue;
    }

    const int cx = static_cast<int>(std::floor((point.x - grid_min_x) / cell_size));
    const int cy = static_cast<int>(std::floor((point.y - grid_min_y) / cell_size));
    if (cx < 0 || cx >= grid_width || cy < 0 || cy >= grid_height) {
      continue;
    }
    auto & cell = cells[cell_index(cx, cy)];
    cell.radius = CellCenterDistance(
      cx - grid_width / 2,
      cy - grid_height / 2,
      cell_size);
    cell.max_z = std::max(cell.max_z, point.z);
  }

  if (used_cuda) {
    for (int cy = 0; cy < grid_height; ++cy) {
      for (int cx = 0; cx < grid_width; ++cx) {
        const std::size_t index = cell_index(cx, cy);
        if (valid_grid[index] == 0) {
          continue;
        }
        auto & cell = cells[index];
        cell.valid = true;
        cell.min_z = min_z_grid[index];
        has_valid_cell = true;
      }
    }
  } else {
    for (const auto & point : input) {
      if (!IsCandidate(point, params)) {
        continue;
      }

      const int cx = static_cast<int>(std::floor((point.x - grid_min_x) / cell_size));
      const int cy = static_cast<int>(std::floor((point.y - grid_min_y) / cell_size));
      if (cx < 0 || cx >= grid_width || cy < 0 || cy >= grid_height) {
        continue;
      }
      auto & cell = cells[cell_index(cx, cy)];
      cell.valid = true;
      cell.min_z = std::min(cell.min_z, point.z);
      has_valid_cell = true;
    }
  }

  if (!has_valid_cell) {
    result.non_ground_points = input;
    return result;
  }

  std::vector<std::size_t> ordered_cells;
  ordered_cells.reserve(cell_count);
  for (std::size_t index = 0; index < cells.size(); ++index) {
    if (cells[index].valid) {
      ordered_cells.push_back(index);
    }
  }
  std::sort(
    ordered_cells.begin(),
    ordered_cells.end(),
    [&cells](std::size_t lhs, std::size_t rhs) {
      return cells[lhs].radius < cells[rhs].radius;
    });

  const float seed_radius = std::max(params.grid_seed_radius_m, cell_size * 2.0f);
  const float max_height_step = std::max(params.grid_max_height_step_m, 0.02f);
  const float max_local_slope = std::max(params.grid_max_local_slope_m_per_m, 0.01f);
  const float max_vertical_extent = std::max(params.grid_max_vertical_extent_m, 0.05f);

  for (const std::size_t index : ordered_cells) {
    auto & cell = cells[index];
    const float radius = cell.radius;
    const int cx = static_cast<int>(index % static_cast<std::size_t>(grid_width));
    const int cy = static_cast<int>(index / static_cast<std::size_t>(grid_width));

    if (radius <= seed_radius) {
      if ((cell.max_z - cell.min_z) > max_vertical_extent) {
        continue;
      }
      cell.accepted_ground = true;
      cell.accepted_z = cell.min_z;
      continue;
    }

    if ((cell.max_z - cell.min_z) > max_vertical_extent) {
      continue;
    }

    bool found_neighbor_ground = false;
    int supporting_neighbors = 0;
    float best_neighbor_z = std::numeric_limits<float>::max();
    float highest_support_z = std::numeric_limits<float>::lowest();
    float best_allowed_z = std::numeric_limits<float>::max();

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const int nx = cx + dx;
        const int ny = cy + dy;
        if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
          continue;
        }
        const auto & neighbor = cells[cell_index(nx, ny)];
        if (!neighbor.accepted_ground) {
          continue;
        }
        if (neighbor.radius > radius + cell_size * 0.5f) {
          continue;
        }

        const float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy)) * cell_size;
        const float allowed_z = neighbor.accepted_z + max_height_step + max_local_slope * distance;
        best_neighbor_z = std::min(best_neighbor_z, neighbor.accepted_z);
        highest_support_z = std::max(highest_support_z, neighbor.accepted_z);
        best_allowed_z = std::min(best_allowed_z, allowed_z);
        ++supporting_neighbors;
        found_neighbor_ground = true;
      }
    }

    if (!found_neighbor_ground || supporting_neighbors < 2) {
      continue;
    }

    if ((highest_support_z - best_neighbor_z) > max_height_step * 1.5f) {
      continue;
    }

    if (cell.min_z <= best_allowed_z && cell.min_z >= best_neighbor_z - std::max(params.max_below_plane_m, 0.2f)) {
      cell.accepted_ground = true;
      cell.accepted_z = std::min(cell.min_z, best_allowed_z);
    }
  }

  std::vector<float> accepted_ground_z_grid(cell_count, std::numeric_limits<float>::max());
  std::vector<unsigned char> accepted_ground_valid_grid(cell_count, 0);
  for (std::size_t index = 0; index < cells.size(); ++index) {
    if (!cells[index].accepted_ground) {
      continue;
    }
    accepted_ground_z_grid[index] = cells[index].accepted_z;
    accepted_ground_valid_grid[index] = 1;
  }

  std::vector<unsigned char> is_ground_mask;
  const bool used_cuda_classification =
    params.use_cuda &&
    ClassifyGroundPointsGridCuda(
      input,
      params,
      grid_min_x,
      grid_min_y,
      grid_width,
      grid_height,
      cell_size,
      accepted_ground_z_grid,
      accepted_ground_valid_grid,
      is_ground_mask);

  result.ground_points.reserve(input.size() / 3);
  result.non_ground_points.reserve(input.size());
  if (used_cuda_classification) {
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (is_ground_mask[i] != 0) {
        result.ground_points.push_back(input[i]);
      } else {
        result.non_ground_points.push_back(input[i]);
      }
    }
  } else {
    for (const auto & point : input) {
      if (!IsWithinRadius(point, params)) {
        result.non_ground_points.push_back(point);
        continue;
      }

      const int cx = static_cast<int>(std::floor((point.x - grid_min_x) / cell_size));
      const int cy = static_cast<int>(std::floor((point.y - grid_min_y) / cell_size));
      bool classified_ground = false;
      float local_ground_z = std::numeric_limits<float>::max();
      float highest_ground_z = std::numeric_limits<float>::lowest();
      int support_count = 0;

      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = cx + dx;
          const int ny = cy + dy;
          if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
            continue;
          }
          const auto & cell = cells[cell_index(nx, ny)];
          if (!cell.accepted_ground || (cell.max_z - cell.min_z) > max_vertical_extent) {
            continue;
          }
          local_ground_z = std::min(local_ground_z, cell.accepted_z);
          highest_ground_z = std::max(highest_ground_z, cell.accepted_z);
          ++support_count;
        }
      }

      if (
        local_ground_z != std::numeric_limits<float>::max() &&
        support_count >= 2 &&
        (highest_ground_z - local_ground_z) <= (std::max(params.max_above_plane_m, 0.05f) * 1.5f))
      {
        const float allowed_above =
          RangeAdaptiveThreshold(point, std::max(params.final_distance_threshold_m, 0.05f), params);
        const float dz = point.z - local_ground_z;
        classified_ground =
          dz <= std::min(std::max(params.max_above_plane_m, allowed_above), allowed_above) &&
          dz >= -std::max(params.max_below_plane_m, 0.1f);
      }

      if (classified_ground) {
        result.ground_points.push_back(point);
      } else {
        result.non_ground_points.push_back(point);
      }
    }
  }

  result.non_ground_points.shrink_to_fit();
  result.ground_points.shrink_to_fit();
  return result;
}

std::vector<CudaPlaneModel> BuildPlaneHypotheses(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  int iterations,
  float min_normal_z)
{
  std::mt19937 rng(1337u);
  std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);

  std::vector<CudaPlaneModel> planes;
  planes.reserve(static_cast<std::size_t>(iterations));
  for (int iter = 0; iter < iterations; ++iter) {
    const std::size_t i0 = candidates[pick(rng)];
    const std::size_t i1 = candidates[pick(rng)];
    const std::size_t i2 = candidates[pick(rng)];
    if (i0 == i1 || i0 == i2 || i1 == i2) {
      continue;
    }

    Plane plane;
    if (!TryMakePlane(input[i0], input[i1], input[i2], plane)) {
      continue;
    }
    if (plane.c < min_normal_z) {
      continue;
    }

    planes.push_back({plane.a, plane.b, plane.c, plane.d});
  }

  return planes;
}

std::vector<int> ScorePlanesCpu(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  const std::vector<CudaPlaneModel> & planes,
  float threshold)
{
  std::vector<int> inlier_counts(planes.size(), 0);
  for (std::size_t plane_index = 0; plane_index < planes.size(); ++plane_index) {
    const Plane plane{
      planes[plane_index].a,
      planes[plane_index].b,
      planes[plane_index].c,
      planes[plane_index].d
    };

    int inliers = 0;
    for (const std::size_t candidate_index : candidates) {
      if (DistanceToPlane(plane, input[candidate_index]) <= threshold) {
        ++inliers;
      }
    }
    inlier_counts[plane_index] = inliers;
  }
  return inlier_counts;
}

void LogGroundCudaBackendStatus(bool requested, bool used)
{
  static bool logged = false;
  if (logged || !requested) {
    return;
  }

  if (used) {
    std::cout << "Ground RANSAC CUDA backend enabled\n";
  } else {
    std::cout << "Ground RANSAC CUDA backend unavailable; using CPU fallback\n";
  }
  logged = true;
}
}  // namespace

GroundRemovalResult RunGroundRemovalRansac(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.size() < 3) {
    result.non_ground_points = input;
    return result;
  }

  std::vector<std::size_t> candidates = BuildLowestPointCandidates(input, params);

  if (candidates.size() < 3) {
    result.non_ground_points = input;
    return result;
  }

  const float max_angle_rad = params.max_ground_angle_deg * 3.14159265358979323846f / 180.0f;
  const float min_normal_z = std::cos(max_angle_rad);
  const float threshold = std::max(params.distance_threshold_m, 0.01f);
  const float final_threshold = std::max(params.final_distance_threshold_m, threshold);
  const int iterations = std::max(params.max_iterations, 1);

  const std::vector<CudaPlaneModel> plane_hypotheses =
    BuildPlaneHypotheses(input, candidates, iterations, min_normal_z);
  if (plane_hypotheses.empty()) {
    result.non_ground_points = input;
    return result;
  }

  Plane best_plane{};
  int best_inliers = -1;
  std::vector<int> inlier_counts;
  const bool used_cuda =
    params.use_cuda &&
    ScorePlanesCuda(input, candidates, plane_hypotheses, threshold, inlier_counts);
  LogGroundCudaBackendStatus(params.use_cuda, used_cuda);
  if (!used_cuda) {
    inlier_counts = ScorePlanesCpu(input, candidates, plane_hypotheses, threshold);
  }

  for (std::size_t plane_index = 0; plane_index < plane_hypotheses.size(); ++plane_index) {
    if (inlier_counts[plane_index] > best_inliers) {
      best_inliers = inlier_counts[plane_index];
      best_plane = {
        plane_hypotheses[plane_index].a,
        plane_hypotheses[plane_index].b,
        plane_hypotheses[plane_index].c,
        plane_hypotheses[plane_index].d
      };
    }
  }

  if (best_inliers < params.min_inliers) {
    result.non_ground_points = input;
    return result;
  }

  const auto local_ground_heights = BuildLocalGroundHeights(input, params);

  result.ground_points.reserve(static_cast<std::size_t>(best_inliers));
  result.non_ground_points.reserve(input.size() - static_cast<std::size_t>(best_inliers));
  for (const auto & point : input) {
    const float signed_distance = SignedDistanceToPlane(best_plane, point);
    const float adaptive_threshold = RangeAdaptiveThreshold(point, final_threshold, params);
    const float tight_threshold = std::min(final_threshold, 0.18f);
    const bool near_plane = std::abs(signed_distance) <= adaptive_threshold;
    const bool very_near_plane = std::abs(signed_distance) <= tight_threshold;
    const bool plausible_ground_height =
      signed_distance <= std::max(params.max_above_plane_m, final_threshold) &&
      signed_distance >= -std::max(params.max_below_plane_m, final_threshold);
    const bool near_local_ground = IsNearLocalGround(point, params, local_ground_heights);
    if (IsWithinRadius(point, params) && plausible_ground_height && near_plane &&
      (very_near_plane || near_local_ground))
    {
      result.ground_points.push_back(point);
    } else {
      result.non_ground_points.push_back(point);
    }
  }

  return result;
}

GroundRemovalResult RunGroundRemovalGrid(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  return RunGroundRemovalGridInternal(input, params);
}

GroundRemovalResult RunGroundRemovalPlaceholder(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  if (params.mode == "grid" || params.mode == "GRID") {
    return RunGroundRemovalGrid(input, params);
  }
  return RunGroundRemovalRansac(input, params);
}
}  // namespace lidar_viewer::algorithms
