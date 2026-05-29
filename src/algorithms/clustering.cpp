#include "lidar_viewer/algorithms/clustering.hpp"
#include "lidar_viewer/algorithms/clustering_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace lidar_viewer::algorithms
{
namespace
{
struct Vec2
{
  float x{0.0f};
  float y{0.0f};
};

struct Cell
{
  std::vector<std::size_t> point_indices;
};

std::int64_t GridKey(int x, int y)
{
  return (static_cast<std::int64_t>(x) << 32) ^ static_cast<std::uint32_t>(y);
}

std::pair<int, int> CellCoord(const KittiPoint & point, float cell_size)
{
  return {
    static_cast<int>(std::floor(point.x / cell_size)),
    static_cast<int>(std::floor(point.y / cell_size))
  };
}

void ExpandBounds(BoundingBox3D & bounds, const KittiPoint & point)
{
  bounds.min.x = std::min(bounds.min.x, point.x);
  bounds.min.y = std::min(bounds.min.y, point.y);
  bounds.min.z = std::min(bounds.min.z, point.z);
  bounds.max.x = std::max(bounds.max.x, point.x);
  bounds.max.y = std::max(bounds.max.y, point.y);
  bounds.max.z = std::max(bounds.max.z, point.z);
}

float Cross2D(const Vec2 & a, const Vec2 & b, const Vec2 & c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

std::vector<Vec2> BuildConvexHull(std::vector<Vec2> points)
{
  if (points.size() <= 1) {
    return points;
  }

  std::sort(
    points.begin(),
    points.end(),
    [](const Vec2 & lhs, const Vec2 & rhs) {
      if (lhs.x == rhs.x) {
        return lhs.y < rhs.y;
      }
      return lhs.x < rhs.x;
    });

  std::vector<Vec2> hull;
  hull.reserve(points.size() * 2);

  for (const auto & point : points) {
    while (hull.size() >= 2 && Cross2D(hull[hull.size() - 2], hull[hull.size() - 1], point) <= 0.0f) {
      hull.pop_back();
    }
    hull.push_back(point);
  }

  const std::size_t lower_size = hull.size();
  for (std::size_t i = points.size(); i-- > 0;) {
    const auto & point = points[i];
    while (hull.size() > lower_size && Cross2D(hull[hull.size() - 2], hull[hull.size() - 1], point) <= 0.0f) {
      hull.pop_back();
    }
    hull.push_back(point);
  }

  if (!hull.empty()) {
    hull.pop_back();
  }
  return hull;
}

OrientedBox3D ComputeOrientedBounds(const std::vector<KittiPoint> & input, const Cluster & cluster)
{
  constexpr std::size_t kMinStableOrientedPointCount = 24;
  constexpr float kMinStableAspectRatio = 1.35f;
  constexpr float kMinStableFootprintSideM = 0.35f;

  OrientedBox3D oriented;
  if (cluster.indices.size() < 2) {
    return oriented;
  }

  std::vector<Vec2> footprint;
  footprint.reserve(cluster.indices.size());
  for (const std::size_t index : cluster.indices) {
    footprint.push_back({input[index].x, input[index].y});
  }

  const std::vector<Vec2> hull = BuildConvexHull(std::move(footprint));
  if (hull.size() < 2) {
    return oriented;
  }

  float best_area = std::numeric_limits<float>::max();
  Vec2 best_axis_x{1.0f, 0.0f};
  Vec2 best_axis_y{0.0f, 1.0f};
  float best_min_x = 0.0f;
  float best_max_x = 0.0f;
  float best_min_y = 0.0f;
  float best_max_y = 0.0f;

  for (std::size_t i = 0; i < hull.size(); ++i) {
    const Vec2 p0 = hull[i];
    const Vec2 p1 = hull[(i + 1) % hull.size()];
    const float edge_x = p1.x - p0.x;
    const float edge_y = p1.y - p0.y;
    const float edge_len = std::sqrt(edge_x * edge_x + edge_y * edge_y);
    if (edge_len < 1e-5f) {
      continue;
    }

    const Vec2 axis_x{edge_x / edge_len, edge_y / edge_len};
    const Vec2 axis_y{-axis_x.y, axis_x.x};
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto & point : hull) {
      const float proj_x = point.x * axis_x.x + point.y * axis_x.y;
      const float proj_y = point.x * axis_y.x + point.y * axis_y.y;
      min_x = std::min(min_x, proj_x);
      max_x = std::max(max_x, proj_x);
      min_y = std::min(min_y, proj_y);
      max_y = std::max(max_y, proj_y);
    }

    const float area = (max_x - min_x) * (max_y - min_y);
    if (area < best_area) {
      best_area = area;
      best_axis_x = axis_x;
      best_axis_y = axis_y;
      best_min_x = min_x;
      best_max_x = max_x;
      best_min_y = min_y;
      best_max_y = max_y;
    }
  }

  const float center_proj_x = 0.5f * (best_min_x + best_max_x);
  const float center_proj_y = 0.5f * (best_min_y + best_max_y);
  const float length = best_max_x - best_min_x;
  const float width = best_max_y - best_min_y;
  const float longer_side = std::max(length, width);
  const float shorter_side = std::max(std::min(length, width), 1e-4f);
  const float aspect_ratio = longer_side / shorter_side;

  // Small or near-square clusters do not have a stable heading, so falling back
  // to the axis-aligned box avoids jittery, unintuitive rotations.
  if (
    cluster.indices.size() < kMinStableOrientedPointCount ||
    shorter_side < kMinStableFootprintSideM ||
    aspect_ratio < kMinStableAspectRatio)
  {
    return oriented;
  }

  oriented.center = {
    best_axis_x.x * center_proj_x + best_axis_y.x * center_proj_y,
    best_axis_x.y * center_proj_x + best_axis_y.y * center_proj_y,
    0.5f * (cluster.bounds.min.z + cluster.bounds.max.z)
  };
  oriented.axis_x = {best_axis_x.x, best_axis_x.y, 0.0f};
  oriented.axis_y = {best_axis_y.x, best_axis_y.y, 0.0f};
  oriented.half_length = 0.5f * length;
  oriented.half_width = 0.5f * width;
  oriented.min_z = cluster.bounds.min.z;
  oriented.max_z = cluster.bounds.max.z;
  oriented.valid = true;
  return oriented;
}

bool PassesSizeFilter(const Cluster & cluster, const ClusteringParams & params)
{
  const float width = cluster.oriented_bounds.valid
    ? (cluster.oriented_bounds.half_width * 2.0f)
    : (cluster.bounds.max.y - cluster.bounds.min.y);
  const float length = cluster.oriented_bounds.valid
    ? (cluster.oriented_bounds.half_length * 2.0f)
    : (cluster.bounds.max.x - cluster.bounds.min.x);
  const float height = cluster.bounds.max.z - cluster.bounds.min.z;

  return
    width >= params.min_width_m && width <= params.max_width_m &&
    length >= params.min_length_m && length <= params.max_length_m &&
    height >= params.min_height_m && height <= params.max_height_m;
}

void LogCudaBackendStatus(bool requested, bool used)
{
  static bool logged = false;
  if (logged || !requested) {
    return;
  }

  if (used) {
    std::cout << "Clustering CUDA backend enabled\n";
  } else {
    std::cout << "Clustering CUDA backend unavailable; using CPU fallback\n";
  }
  logged = true;
}
}  // namespace

std::vector<Cluster> RunClusteringPlaceholder(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params)
{
  if (!params.enabled || input.empty()) {
    return {};
  }

  const float cell_size = std::max(params.grid_cell_m, 0.05f);
  std::unordered_map<std::int64_t, Cell> cells;
  cells.reserve(input.size() / 4);

  std::vector<CudaCellAssignment> cuda_assignments;
  const bool used_cuda = params.use_cuda && BuildCellAssignmentsCuda(input, params, cuda_assignments);
  LogCudaBackendStatus(params.use_cuda, used_cuda);
  if (used_cuda) {
    for (const auto & assignment : cuda_assignments) {
      cells[assignment.key].point_indices.push_back(assignment.point_index);
    }
  } else {
    for (std::size_t i = 0; i < input.size(); ++i) {
      const auto [cx, cy] = CellCoord(input[i], cell_size);
      cells[GridKey(cx, cy)].point_indices.push_back(i);
    }
  }

  const int neighbor_span = std::max(1, static_cast<int>(std::ceil(params.tolerance_m / cell_size)));
  std::unordered_set<std::int64_t> visited;
  visited.reserve(cells.size());
  std::vector<Cluster> clusters;

  for (const auto & [start_key, _] : cells) {
    if (visited.find(start_key) != visited.end()) {
      continue;
    }

    Cluster cluster;
    cluster.bounds.min = {
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()
    };
    cluster.bounds.max = {
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest()
    };

    std::queue<std::int64_t> pending;
    pending.push(start_key);
    visited.insert(start_key);

    while (!pending.empty()) {
      const std::int64_t key = pending.front();
      pending.pop();

      const auto cell_it = cells.find(key);
      if (cell_it == cells.end()) {
        continue;
      }

      for (const std::size_t point_index : cell_it->second.point_indices) {
        cluster.indices.push_back(point_index);
        ExpandBounds(cluster.bounds, input[point_index]);
      }

      const int cx = static_cast<int>(static_cast<std::int32_t>(key >> 32));
      const int cy = static_cast<int>(static_cast<std::int32_t>(static_cast<std::uint32_t>(key)));
      for (int dy = -neighbor_span; dy <= neighbor_span; ++dy) {
        for (int dx = -neighbor_span; dx <= neighbor_span; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          if (std::sqrt(static_cast<float>(dx * dx + dy * dy)) * cell_size > params.tolerance_m) {
            continue;
          }

          const std::int64_t neighbor_key = GridKey(cx + dx, cy + dy);
          if (cells.find(neighbor_key) == cells.end() || visited.find(neighbor_key) != visited.end()) {
            continue;
          }
          visited.insert(neighbor_key);
          pending.push(neighbor_key);
        }
      }
    }

    const int cluster_size = static_cast<int>(cluster.indices.size());
    cluster.oriented_bounds = ComputeOrientedBounds(input, cluster);
    if (
      cluster_size >= params.min_cluster_size &&
      cluster_size <= params.max_cluster_size &&
      PassesSizeFilter(cluster, params))
    {
      clusters.push_back(std::move(cluster));
    }
  }

  return clusters;
}
}  // namespace lidar_viewer::algorithms
