#include "lidar_viewer/algorithms/clustering.hpp"
#include "lidar_viewer/algorithms/clustering_cuda.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace lidar_viewer::algorithms
{
namespace
{
struct Vec2
{
  float x{0.0f};
  float y{0.0f};
};

std::uint64_t GridKey(int x, int y)
{
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
    static_cast<std::uint32_t>(y);
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
  // The hull's first edge has no preferred heading. Always name the long
  // side "length", including when a vehicle turns across the sensor's axes.
  oriented.axis_x = length >= width
    ? Vec3{best_axis_x.x, best_axis_x.y, 0.0f}
    : Vec3{best_axis_y.x, best_axis_y.y, 0.0f};
  oriented.axis_y = {-oriented.axis_x.y, oriented.axis_x.x, 0.0f};
  oriented.half_length = 0.5f * longer_side;
  oriented.half_width = 0.5f * std::min(length, width);
  oriented.min_z = cluster.bounds.min.z;
  oriented.max_z = cluster.bounds.max.z;
  oriented.valid = true;
  return oriented;
}

bool PassesSizeFilter(const Cluster & cluster, const ClusteringParams & params)
{
  const float width = cluster.oriented_bounds.valid
    ? (cluster.oriented_bounds.half_width * 2.0f)
    : std::min(cluster.bounds.max.x - cluster.bounds.min.x, cluster.bounds.max.y - cluster.bounds.min.y);
  const float length = cluster.oriented_bounds.valid
    ? (cluster.oriented_bounds.half_length * 2.0f)
    : std::max(cluster.bounds.max.x - cluster.bounds.min.x, cluster.bounds.max.y - cluster.bounds.min.y);
  const float height = cluster.bounds.max.z - cluster.bounds.min.z;

  return
    width >= params.min_width_m && width <= params.max_width_m &&
    length >= params.min_length_m && length <= params.max_length_m &&
    height >= params.min_height_m && height <= params.max_height_m;
}

void LogCudaBackendStatus(bool requested, bool used)
{
  static std::atomic<bool> logged{false};
  if (!requested || logged.exchange(true)) {
    return;
  }

  if (used) {
    std::cout << "Clustering CUDA backend enabled\n";
  } else {
    std::cout << "Clustering CUDA backend unavailable; using CPU fallback\n";
  }
}
}  // namespace

std::vector<Cluster> RunClusteringEuclidean(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params)
{
  if (!params.enabled || input.empty()) {
    return {};
  }

  const float base_tolerance = std::max(params.tolerance_m, 0.05f);
  const float max_tolerance = std::max(params.max_tolerance_m, base_tolerance);
  // Small cells avoid repeatedly checking tens of thousands of close-range
  // returns against points on the far side of a maximum-radius bucket.
  // Bound the widest query to eight cells in each direction.
  const float cell_size = std::max({0.05f, max_tolerance / 8.0f,
    std::min(params.grid_cell_m, base_tolerance)});
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> cells;
  cells.reserve(input.size() / 12);

  std::vector<std::pair<int, int>> coordinates(input.size());
  std::vector<unsigned char> visited(input.size(), 0);
  std::vector<float> tolerance_squared(input.size(), 0.0f);
  std::vector<float> query_radius(input.size(), 0.0f);
  const float range_scale = std::max(params.range_tolerance_scale, 0.0f);
  bool all_indexable = true;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto & point = input[i];
    // Leave room for neighbouring cells and reject invalid input before any
    // floating-to-integer conversion (including the optional CUDA kernel).
    constexpr double limit = static_cast<double>(std::numeric_limits<int>::max()) - 16;
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
      std::abs(static_cast<double>(point.x) / cell_size) >= limit ||
      std::abs(static_cast<double>(point.y) / cell_size) >= limit) {
      visited[i] = 1;
      all_indexable = false;
      continue;
    }
    coordinates[i] = CellCoord(point, cell_size);
    const float range = std::hypot(point.x, point.y);
    const float tolerance = std::min(max_tolerance, base_tolerance + range_scale * range);
    tolerance_squared[i] = tolerance * tolerance;
    // A connected neighbour is at most max_tolerance farther from the
    // sensor. This conservative bound also includes a neighbour whose own
    // adaptive tolerance is larger than the current point's tolerance.
    query_radius[i] = std::min(max_tolerance,
      base_tolerance + range_scale * (range + max_tolerance)) + cell_size * 1e-4f;
  }

  std::vector<CudaCellAssignment> cuda_assignments;
  ClusteringParams cuda_params = params;
  cuda_params.grid_cell_m = cell_size;
  const bool used_cuda = params.use_cuda && all_indexable &&
    BuildCellAssignmentsCuda(input, cuda_params, cuda_assignments);
  LogCudaBackendStatus(params.use_cuda, used_cuda);
  if (used_cuda) {
    for (const auto & assignment : cuda_assignments) {
      cells[assignment.key].push_back(assignment.point_index);
    }
  } else {
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (visited[i] != 0) continue;
      const auto [cx, cy] = coordinates[i];
      cells[GridKey(cx, cy)].push_back(i);
    }
  }

  const bool distance_3d=params.mode=="euclidean_3d";
  const float vertical_gap = std::max(params.max_vertical_gap_m, 0.05f);
  std::vector<Cluster> clusters;
  clusters.reserve(cells.size() / 8);

  for (std::size_t start_index = 0; start_index < input.size(); ++start_index) {
    if (visited[start_index] != 0) {
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

    cluster.indices.reserve(64);
    cluster.indices.push_back(start_index);
    visited[start_index] = 1;

    for (std::size_t pending_index = 0; pending_index < cluster.indices.size(); ++pending_index) {
      const std::size_t point_index = cluster.indices[pending_index];
      const KittiPoint & point = input[point_index];
      ExpandBounds(cluster.bounds, point);

      const auto [cx, cy] = coordinates[point_index];
      const float radius = query_radius[point_index];
      const int neighbor_span = static_cast<int>(std::ceil(radius / cell_size));
      for (int dy = -neighbor_span; dy <= neighbor_span; ++dy) {
        const float min_y = static_cast<float>(cy + dy) * cell_size;
        const float distance_y = std::max({min_y - point.y, point.y - (min_y + cell_size), 0.0f});
        if (distance_y > radius) continue;
        for (int dx = -neighbor_span; dx <= neighbor_span; ++dx) {
          const float min_x = static_cast<float>(cx + dx) * cell_size;
          const float distance_x = std::max({min_x - point.x, point.x - (min_x + cell_size), 0.0f});
          if (distance_x * distance_x + distance_y * distance_y > radius * radius) continue;
          const auto neighbor_key = GridKey(cx + dx, cy + dy);
          const auto cell_it = cells.find(neighbor_key);
          if (cell_it == cells.end()) {
            continue;
          }
          auto & candidates = cell_it->second;
          for (std::size_t slot = 0; slot < candidates.size();) {
            const std::size_t neighbor_index = candidates[slot];
            if (visited[neighbor_index] != 0) {
              candidates[slot] = candidates.back();
              candidates.pop_back();
              continue;
            }
            const KittiPoint & neighbor = input[neighbor_index];
            const float dz = std::abs(neighbor.z - point.z);
            if (dz > vertical_gap) {
              ++slot;
              continue;
            }
            const float delta_x = neighbor.x - point.x;
            const float delta_y = neighbor.y - point.y;
            if (delta_x * delta_x + delta_y * delta_y + (distance_3d ? dz*dz : 0.f) >
              std::max(tolerance_squared[point_index], tolerance_squared[neighbor_index])) {
              ++slot;
              continue;
            }
            visited[neighbor_index] = 1;
            cluster.indices.push_back(neighbor_index);
            // A discovered point remains in the BFS queue, but never needs
            // to be searched for again. This avoids quadratic visited scans
            // inside dense cells without changing the connectivity rule.
            candidates[slot] = candidates.back();
            candidates.pop_back();
          }
        }
      }
    }

    const int cluster_size = static_cast<int>(cluster.indices.size());
    if (cluster_size < params.min_cluster_size || cluster_size > params.max_cluster_size) {
      continue;
    }
    const float height = cluster.bounds.max.z - cluster.bounds.min.z;
    if (height < params.min_height_m || height > params.max_height_m) {
      continue;
    }
    cluster.oriented_bounds = ComputeOrientedBounds(input, cluster);
    if (PassesSizeFilter(cluster, params)) {
      clusters.push_back(std::move(cluster));
    }
  }

  return clusters;
}

std::vector<Cluster> RunClustering(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params)
{
  if (params.enabled && params.mode!="euclidean" && params.mode!="euclidean_3d")
    throw std::invalid_argument("Unknown clustering mode: "+params.mode);
  return RunClusteringEuclidean(input, params);
}
}  // namespace lidar_viewer::algorithms
