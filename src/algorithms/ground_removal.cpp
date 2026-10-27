#include "lidar_viewer/algorithms/ground_removal.hpp"
#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"

#include <algorithm>
#include <cctype>
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
  if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
    point.z < params.min_candidate_z || point.z > params.max_candidate_z) {
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

std::uint64_t GridKey(const KittiPoint & point, float cell_size)
{
  const auto ix = static_cast<std::int32_t>(std::floor(point.x / cell_size));
  const auto iy = static_cast<std::int32_t>(std::floor(point.y / cell_size));
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix)) << 32) ^ static_cast<std::uint32_t>(iy);
}

std::vector<std::size_t> BuildLowestPointCandidates(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  std::unordered_map<std::uint64_t, std::size_t> lowest_by_cell;
  lowest_by_cell.reserve(input.size() / 8);

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (!IsCandidate(input[i], params)) {
      continue;
    }

    const std::uint64_t key = GridKey(input[i], cell_size);
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

std::unordered_map<std::uint64_t, float> BuildLocalGroundHeights(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  std::unordered_map<std::uint64_t, float> lowest_z_by_cell;
  lowest_z_by_cell.reserve(input.size() / 8);

  for (const auto & point : input) {
    if (!IsCandidate(point, params)) {
      continue;
    }

    const std::uint64_t key = GridKey(point, cell_size);
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
  const std::unordered_map<std::uint64_t, float> & local_ground_heights)
{
  if (!IsWithinRadius(point, params)) return false;
  const float cell_size = std::max(params.candidate_grid_cell_m, 0.1f);
  const auto ix = static_cast<std::int32_t>(std::floor(point.x / cell_size));
  const auto iy = static_cast<std::int32_t>(std::floor(point.y / cell_size));

  float local_min_z = std::numeric_limits<float>::max();
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix + dx)) << 32) ^
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

std::vector<CudaPlaneModel> BuildPlaneHypotheses(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  int iterations,
  float min_normal_z);

std::vector<int> ScorePlanesCpu(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  const std::vector<CudaPlaneModel> & planes,
  float threshold);

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
  const float max_angle_rad = params.max_ground_angle_deg * 3.14159265358979323846f / 180.0f;
  const float min_normal_z = std::cos(max_angle_rad);
  const float threshold = std::max(params.distance_threshold_m, 0.01f);
  const float final_threshold = std::max(params.final_distance_threshold_m, threshold);
  const float max_radius = std::max(params.max_candidate_radius_m, 1.0f);
  const float grid_min_x = -max_radius;
  const float grid_min_y = -max_radius;
  const int grid_width = std::max(1, static_cast<int>(std::ceil((2.0f * max_radius) / cell_size)) + 1);
  const int grid_height = grid_width;
  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * static_cast<std::size_t>(grid_height);

  Plane best_plane{};
  bool has_plane_prior = false;
  {
    std::vector<std::size_t> candidates = BuildLowestPointCandidates(input, params);
    if (candidates.size() >= 3) {
      const int iterations = std::max(params.max_iterations, 1);
      const std::vector<CudaPlaneModel> plane_hypotheses =
        BuildPlaneHypotheses(input, candidates, iterations, min_normal_z);
      if (!plane_hypotheses.empty()) {
        std::vector<int> inlier_counts;
        const bool used_cuda =
          params.use_cuda &&
          ScorePlanesCuda(input, candidates, plane_hypotheses, threshold, inlier_counts);
        if (!used_cuda) {
          inlier_counts = ScorePlanesCpu(input, candidates, plane_hypotheses, threshold);
        }

        int best_inliers = -1;
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

        has_plane_prior = best_inliers >= std::max(params.min_inliers / 2, 40);
      }
    }
  }

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
      if (has_plane_prior) {
        const float cell_x = grid_min_x + (static_cast<float>(cx) + 0.5f) * cell_size;
        const float cell_y = grid_min_y + (static_cast<float>(cy) + 0.5f) * cell_size;
        const KittiPoint probe{cell_x, cell_y, cell.min_z, 0.0f};
        const float signed_distance =
          best_plane.a * cell_x + best_plane.b * cell_y + best_plane.c * cell.min_z + best_plane.d;
        if (std::abs(signed_distance) > RangeAdaptiveThreshold(probe, final_threshold, params)) {
          continue;
        }
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
      if (has_plane_prior) {
        const float cell_x = grid_min_x + (static_cast<float>(cx) + 0.5f) * cell_size;
        const float cell_y = grid_min_y + (static_cast<float>(cy) + 0.5f) * cell_size;
        const KittiPoint probe{cell_x, cell_y, cell.min_z, 0.0f};
        const float signed_distance =
          best_plane.a * cell_x + best_plane.b * cell_y + best_plane.c * cell.min_z + best_plane.d;
        if (
          signed_distance > std::max(params.max_above_plane_m, final_threshold) ||
          signed_distance < -std::max(params.max_below_plane_m, final_threshold) ||
          std::abs(signed_distance) > RangeAdaptiveThreshold(probe, final_threshold, params))
        {
          continue;
        }
      }
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
      const auto & point = input[i];
      const bool inside_ego_hood_box =
        point.x >= 0.5f && point.x <= 4.0f &&
        std::abs(point.y) <= 1.4f &&
        point.z >= 0.2f;
      const float signed_distance =
        has_plane_prior ? SignedDistanceToPlane(best_plane, point) : 0.0f;
      const bool plane_ok =
        !has_plane_prior ||
        (
          signed_distance <= std::max(params.max_above_plane_m, final_threshold) &&
          signed_distance >= -std::max(params.max_below_plane_m, final_threshold) &&
          std::abs(signed_distance) <= RangeAdaptiveThreshold(point, final_threshold, params)
        );
      if (is_ground_mask[i] != 0 && !inside_ego_hood_box && plane_ok) {
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

      const bool inside_ego_hood_box =
        point.x >= 0.5f && point.x <= 4.0f &&
        std::abs(point.y) <= 1.4f &&
        point.z >= 0.2f;
      if (inside_ego_hood_box) {
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
        support_count >= ((std::sqrt(point.x * point.x + point.y * point.y) > 25.0f) ? 1 : 2) &&
        (highest_ground_z - local_ground_z) <= (std::max(params.max_above_plane_m, 0.05f) * 1.5f))
      {
        const float allowed_above =
          RangeAdaptiveThreshold(point, std::max(params.final_distance_threshold_m, 0.05f), params);
        const float dz = point.z - local_ground_z;
        const float signed_distance =
          has_plane_prior ? SignedDistanceToPlane(best_plane, point) : 0.0f;
        const bool plane_ok =
          !has_plane_prior ||
          (
            signed_distance <= std::max(params.max_above_plane_m, final_threshold) &&
            signed_distance >= -std::max(params.max_below_plane_m, final_threshold) &&
            std::abs(signed_distance) <= allowed_above
          );
        classified_ground =
          plane_ok &&
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

// Jacobi eigensolver for a 3x3 symmetric covariance matrix.  Keeping this
// local avoids adding a heavyweight linear algebra dependency just for the
// small per-patch plane fits used by Patchwork++.
bool FitLocalGroundPlane(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & indices,
  Plane & plane,
  float & flatness)
{
  if (indices.size() < 3) {
    return false;
  }

  Vec3 mean{};
  for (const std::size_t index : indices) {
    mean.x += input[index].x;
    mean.y += input[index].y;
    mean.z += input[index].z;
  }
  const float inverse_count = 1.0f / static_cast<float>(indices.size());
  mean = mean * inverse_count;

  float matrix[3][3]{};
  for (const std::size_t index : indices) {
    const KittiPoint & point = input[index];
    const float d[3]{point.x - mean.x, point.y - mean.y, point.z - mean.z};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        matrix[row][column] += d[row] * d[column];
      }
    }
  }
  for (auto & row : matrix) {
    for (float & value : row) {
      value *= inverse_count;
    }
  }

  float vectors[3][3]{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  for (int iteration = 0; iteration < 12; ++iteration) {
    int p = 0;
    int q = 1;
    if (std::abs(matrix[0][2]) > std::abs(matrix[p][q])) {
      p = 0;
      q = 2;
    }
    if (std::abs(matrix[1][2]) > std::abs(matrix[p][q])) {
      p = 1;
      q = 2;
    }
    if (std::abs(matrix[p][q]) < 1e-6f) {
      break;
    }

    const float angle = 0.5f * std::atan2(2.0f * matrix[p][q], matrix[q][q] - matrix[p][p]);
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const float pp = matrix[p][p];
    const float qq = matrix[q][q];
    const float pq = matrix[p][q];
    matrix[p][p] = cosine * cosine * pp - 2.0f * sine * cosine * pq + sine * sine * qq;
    matrix[q][q] = sine * sine * pp + 2.0f * sine * cosine * pq + cosine * cosine * qq;
    matrix[p][q] = matrix[q][p] = 0.0f;
    for (int k = 0; k < 3; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const float pk = matrix[p][k];
      const float qk = matrix[q][k];
      matrix[p][k] = matrix[k][p] = cosine * pk - sine * qk;
      matrix[q][k] = matrix[k][q] = sine * pk + cosine * qk;
    }
    for (int k = 0; k < 3; ++k) {
      const float vp = vectors[k][p];
      const float vq = vectors[k][q];
      vectors[k][p] = cosine * vp - sine * vq;
      vectors[k][q] = sine * vp + cosine * vq;
    }
  }

  int normal_column = 0;
  if (matrix[1][1] < matrix[normal_column][normal_column]) {
    normal_column = 1;
  }
  if (matrix[2][2] < matrix[normal_column][normal_column]) {
    normal_column = 2;
  }
  Vec3 normal{
    vectors[0][normal_column], vectors[1][normal_column], vectors[2][normal_column]};
  const float normal_length = Length(normal);
  if (!std::isfinite(normal_length) || normal_length < 1e-5f) {
    return false;
  }
  normal = normal * (1.0f / normal_length);
  if (normal.z < 0.0f) {
    normal = normal * -1.0f;
  }
  plane = {normal.x, normal.y, normal.z, -DotProduct(normal, mean)};
  flatness = std::max(matrix[normal_column][normal_column], 0.0f);
  return true;
}

GroundRemovalResult RunGroundRemovalPatchworkPPInternal(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  GroundRemovalResult result;
  if (!params.enabled || input.empty()) {
    result.non_ground_points = input;
    return result;
  }

  const int zone_count = std::clamp(params.patchwork_num_zones, 1, 8);
  const int sector_count = std::clamp(params.patchwork_num_sectors, 4, 128);
  const float min_range = std::max(params.patchwork_min_range_m, 0.0f);
  const float max_range = std::max(params.max_candidate_radius_m, min_range + 1.0f);
  const float range_span = max_range - min_range;
  std::vector<int> zone_offsets(static_cast<std::size_t>(zone_count) + 1, 0);
  std::vector<float> zone_boundaries(static_cast<std::size_t>(zone_count), max_range);
  const float zone_denominator = static_cast<float>((1 << zone_count) - 1);
  for (int zone = 0; zone < zone_count; ++zone) {
    // Small near-field rings preserve curbs and vehicle feet; wider far-field
    // rings keep the number of plane fits bounded.
    const int rings = (zone == 0) ? 2 : 4;
    zone_offsets[zone + 1] = zone_offsets[zone] + rings * sector_count;
    // Exponentially wider zones match LiDAR sampling density: detail is kept
    // close to the ego vehicle while far-range plane fitting stays cheap.
    zone_boundaries[zone] = min_range + range_span *
      (static_cast<float>((1 << (zone + 1)) - 1) / zone_denominator);
  }
  std::vector<std::vector<std::size_t>> patches(static_cast<std::size_t>(zone_offsets.back()));

  for (std::size_t index = 0; index < input.size(); ++index) {
    const KittiPoint & point = input[index];
    if (!IsCandidate(point, params)) {
      continue;
    }
    const float range = std::sqrt(point.x * point.x + point.y * point.y);
    if (range < min_range || range > max_range) {
      continue;
    }
    const int zone = static_cast<int>(std::lower_bound(
      zone_boundaries.begin(), zone_boundaries.end(), range) - zone_boundaries.begin());
    const float zone_start = (zone == 0) ? min_range : zone_boundaries[zone - 1];
    const float zone_end = zone_boundaries[zone];
    const float within_zone = std::clamp(
      (range - zone_start) / std::max(zone_end - zone_start, 1e-3f), 0.0f, 0.999999f);
    const int rings = (zone == 0) ? 2 : 4;
    const int ring = std::min(rings - 1, static_cast<int>(within_zone * rings));
    const float angle = std::atan2(point.y, point.x);
    const float normalized_angle = (angle + 3.14159265358979323846f) / (2.0f * 3.14159265358979323846f);
    const int sector = std::min(sector_count - 1, static_cast<int>(normalized_angle * sector_count));
    patches[static_cast<std::size_t>(zone_offsets[zone] + ring * sector_count + sector)].push_back(index);
  }

  const float max_angle_rad = params.max_ground_angle_deg * 3.14159265358979323846f / 180.0f;
  const float min_normal_z = std::cos(max_angle_rad);
  const int min_patch_points = std::max(params.patchwork_min_points_per_patch, 3);
  const int seed_count = std::max(params.patchwork_seed_count, 3);
  const int refine_iterations = std::max(params.patchwork_refine_iterations, 1);
  const float seed_threshold = std::max(params.patchwork_seed_threshold_m, 0.02f);
  const float base_distance_threshold = std::max(params.patchwork_distance_threshold_m, 0.02f);
  const float max_flatness = std::max(params.patchwork_max_flatness_m2, 0.0f);
  std::vector<unsigned char> is_ground(input.size(), 0);

  for (auto & patch : patches) {
    if (static_cast<int>(patch.size()) < min_patch_points) {
      continue;
    }
    std::sort(patch.begin(), patch.end(), [&input](std::size_t lhs, std::size_t rhs) {
      return input[lhs].z < input[rhs].z;
    });
    const int lpr_count = std::min(seed_count, static_cast<int>(patch.size()));
    float lpr_height = 0.0f;
    for (int i = 0; i < lpr_count; ++i) {
      lpr_height += input[patch[static_cast<std::size_t>(i)]].z;
    }
    lpr_height /= static_cast<float>(lpr_count);

    std::vector<std::size_t> seeds;
    seeds.reserve(patch.size());
    for (const std::size_t index : patch) {
      if (input[index].z <= lpr_height + seed_threshold) {
        seeds.push_back(index);
      }
    }

    Plane plane{};
    float flatness = 0.0f;
    bool valid_plane = false;
    for (int iteration = 0; iteration < refine_iterations; ++iteration) {
      if (!FitLocalGroundPlane(input, seeds, plane, flatness) || plane.c < min_normal_z ||
        flatness > max_flatness)
      {
        valid_plane = false;
        break;
      }
      valid_plane = true;
      std::vector<std::size_t> refined_seeds;
      refined_seeds.reserve(patch.size());
      for (const std::size_t index : patch) {
        const float signed_distance = SignedDistanceToPlane(plane, input[index]);
        if (signed_distance >= -seed_threshold && signed_distance <= seed_threshold) {
          refined_seeds.push_back(index);
        }
      }
      if (refined_seeds.size() < 3) {
        valid_plane = false;
        break;
      }
      seeds.swap(refined_seeds);
    }
    if (!valid_plane) {
      continue;
    }

    for (const std::size_t index : patch) {
      const KittiPoint & point = input[index];
      const float threshold = RangeAdaptiveThreshold(point, base_distance_threshold, params);
      const float signed_distance = SignedDistanceToPlane(plane, point);
      if (signed_distance >= -threshold && signed_distance <= threshold) {
        is_ground[index] = 1;
      }
    }
  }

  result.ground_points.reserve(input.size() / 2);
  result.non_ground_points.reserve(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    const KittiPoint & point = input[index];
    const bool ego_hood =
      point.x >= 0.5f && point.x <= 4.0f && std::abs(point.y) <= 1.4f && point.z >= 0.2f;
    if (is_ground[index] != 0 && !ego_hood) {
      result.ground_points.push_back(point);
    } else {
      result.non_ground_points.push_back(point);
    }
  }
  return result;
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

GroundRemovalResult RunGroundRemovalPatchworkPP(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  return RunGroundRemovalPatchworkPPInternal(input, params);
}

GroundRemovalResult RunGroundRemovalPlaceholder(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params)
{
  std::string mode = params.mode;
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  if (mode == "patchworkpp" || mode == "patchwork++" || mode == "patchwork") {
    return RunGroundRemovalPatchworkPP(input, params);
  }
  if (mode == "grid") {
    return RunGroundRemovalGrid(input, params);
  }
  if (mode == "gseg3d" || mode == "gseg") {
    return RunGroundRemovalGSeg3D(input, params);
  }
  if (mode == "fast_grid") {
    return RunGroundRemovalFastGrid(input, params);
  }
  return RunGroundRemovalRansac(input, params);
}
}  // namespace lidar_viewer::algorithms
