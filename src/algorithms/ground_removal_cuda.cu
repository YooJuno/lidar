#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"
#include "cuda_workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lidar_viewer::algorithms
{
namespace
{
bool HasCudaDevice()
{
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

__global__ void ScorePlanesKernel(
  const KittiPoint * points,
  const std::size_t * candidate_indices,
  std::size_t candidate_count,
  const CudaPlaneModel * planes,
  std::size_t plane_count,
  float threshold,
  int * counts)
{
  const std::size_t plane_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (plane_index >= plane_count) {
    return;
  }

  const CudaPlaneModel plane = planes[plane_index];
  int inliers = 0;
  for (std::size_t i = 0; i < candidate_count; ++i) {
    const KittiPoint point = points[candidate_indices[i]];
    const float distance = fabsf(
      plane.a * point.x +
      plane.b * point.y +
      plane.c * point.z +
      plane.d);
    if (distance <= threshold) {
      ++inliers;
    }
  }

  counts[plane_index] = inliers;
}

__device__ float AtomicMinFloat(float * address, float value)
{
  int * address_as_int = reinterpret_cast<int *>(address);
  int old = *address_as_int;
  int assumed = 0;

  while (value < __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    if (assumed == old) {
      break;
    }
  }

  return __int_as_float(old);
}

__global__ void BuildMinZGridKernel(
  const KittiPoint * points,
  std::size_t point_count,
  float min_candidate_z,
  float max_candidate_z,
  float max_radius_sq,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  float * min_z_grid,
  unsigned char * valid_grid)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= point_count) {
    return;
  }

  const KittiPoint point = points[index];
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
      point.z < min_candidate_z || point.z > max_candidate_z) {
    return;
  }

  const float radius_sq = point.x * point.x + point.y * point.y;
  if (radius_sq > max_radius_sq) {
    return;
  }

  const int cx = static_cast<int>(floorf((point.x - grid_min_x) / cell_size));
  const int cy = static_cast<int>(floorf((point.y - grid_min_y) / cell_size));
  if (cx < 0 || cx >= grid_width || cy < 0 || cy >= grid_height) {
    return;
  }

  const int cell_index = cy * grid_width + cx;
  AtomicMinFloat(&min_z_grid[cell_index], point.z);
  valid_grid[cell_index] = 1;
}

__global__ void ClassifyGroundPointsGridKernel(
  const KittiPoint * points,
  std::size_t point_count,
  float max_radius_sq,
  float ego_front_min_x,
  float ego_front_max_x,
  float ego_half_width_y,
  float ego_min_non_ground_z,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  const float * accepted_ground_z_grid,
  const unsigned char * accepted_ground_valid_grid,
  float final_distance_threshold_m,
  float max_above_plane_m,
  float max_below_plane_m,
  float local_ground_margin_m,
  float adaptive_threshold_scale,
  float max_adaptive_threshold_m,
  unsigned char * is_ground_mask)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= point_count) {
    return;
  }

  const KittiPoint point = points[index];
  const float radius_sq = point.x * point.x + point.y * point.y;
  if (radius_sq > max_radius_sq) {
    is_ground_mask[index] = 0;
    return;
  }

  if (
    point.x >= ego_front_min_x && point.x <= ego_front_max_x &&
    fabsf(point.y) <= ego_half_width_y &&
    point.z >= ego_min_non_ground_z)
  {
    is_ground_mask[index] = 0;
    return;
  }

  const int cx = static_cast<int>(floorf((point.x - grid_min_x) / cell_size));
  const int cy = static_cast<int>(floorf((point.y - grid_min_y) / cell_size));
  if (cx < 0 || cx >= grid_width || cy < 0 || cy >= grid_height) {
    is_ground_mask[index] = 0;
    return;
  }

  float local_ground_z = 1.0e9f;
  float highest_ground_z = -1.0e9f;
  int support_count = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
        continue;
      }
      const int cell_index = ny * grid_width + nx;
      if (accepted_ground_valid_grid[cell_index] == 0) {
        continue;
      }
      const float ground_z = accepted_ground_z_grid[cell_index];
      local_ground_z = fminf(local_ground_z, ground_z);
      highest_ground_z = fmaxf(highest_ground_z, ground_z);
      ++support_count;
    }
  }

  const float radius = sqrtf(radius_sq);
  const int min_support_count = (radius > 25.0f) ? 1 : 2;
  if (local_ground_z > 9.0e8f || support_count < min_support_count) {
    is_ground_mask[index] = 0;
    return;
  }

  if ((highest_ground_z - local_ground_z) > (max_above_plane_m * 1.5f)) {
    is_ground_mask[index] = 0;
    return;
  }

  const float distance_scaled = final_distance_threshold_m + fmaxf(adaptive_threshold_scale, 0.0f) * radius;
  const float allowed_above = fminf(
    fmaxf(distance_scaled, final_distance_threshold_m),
    fmaxf(max_adaptive_threshold_m, final_distance_threshold_m));
  const float dz = point.z - local_ground_z;
  const float upper_bound = fminf(fmaxf(max_above_plane_m, allowed_above), allowed_above);
  const float lower_bound = -fmaxf(max_below_plane_m, 0.1f);
  is_ground_mask[index] = (dz <= upper_bound && dz >= lower_bound) ? 1 : 0;
}

__global__ void InitializeGroundGridKernel(float * minimum, unsigned char * valid, std::size_t count)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    minimum[index] = 1.0e9f;
    valid[index] = 0;
  }
}

__global__ void SeedGSeg3DGridKernel(
  const float * min_z,
  const unsigned char * valid,
  int grid_width,
  int grid_height,
  float grid_min_x,
  float grid_min_y,
  float cell_size,
  float ground_height,
  float seed_radius,
  float seed_height_tolerance,
  unsigned char * state)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * grid_height;
  if (index >= cell_count) {
    return;
  }
  const int x = static_cast<int>(index % grid_width);
  const int y = static_cast<int>(index / grid_width);
  const float cx = grid_min_x + (static_cast<float>(x) + 0.5f) * cell_size;
  const float cy = grid_min_y + (static_cast<float>(y) + 0.5f) * cell_size;
  const float radius = sqrtf(cx * cx + cy * cy);
  state[index] = (valid[index] != 0 && radius <= seed_radius &&
    fabsf(min_z[index] - ground_height) <= seed_height_tolerance) ? 1 : 0;
}

__global__ void ExpandGSeg3DGridKernel(
  const float * min_z,
  const unsigned char * valid,
  const unsigned char * current_state,
  int grid_width,
  int grid_height,
  float max_height_step,
  unsigned char * next_state)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * grid_height;
  if (index >= cell_count) {
    return;
  }
  if (current_state[index] != 0) {
    next_state[index] = 1;
    return;
  }
  if (valid[index] == 0) {
    next_state[index] = 0;
    return;
  }

  const int x = static_cast<int>(index % grid_width);
  const int y = static_cast<int>(index / grid_width);
  unsigned char accepted = 0;
  for (int dy = -1; dy <= 1 && accepted == 0; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
        continue;
      }
      const std::size_t neighbor = static_cast<std::size_t>(ny) * grid_width + nx;
      if (current_state[neighbor] != 0 &&
          fabsf(min_z[index] - min_z[neighbor]) <= max_height_step) {
        accepted = 1;
        break;
      }
    }
  }
  next_state[index] = accepted;
}

__global__ void RefineGSeg3DGridKernel(
  const float * phase_two_min_z,
  const unsigned char * phase_two_valid,
  int phase_two_width,
  int phase_two_height,
  float grid_min_x,
  float grid_min_y,
  float phase_two_cell_size,
  const float * phase_one_min_z,
  const unsigned char * phase_one_state,
  int phase_one_width,
  int phase_one_height,
  float phase_one_cell_size,
  float max_height_deviation,
  unsigned char * phase_two_state)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t cell_count = static_cast<std::size_t>(phase_two_width) * phase_two_height;
  if (index >= cell_count) {
    return;
  }
  if (phase_two_valid[index] == 0) {
    phase_two_state[index] = 0;
    return;
  }

  const int x2 = static_cast<int>(index % phase_two_width);
  const int y2 = static_cast<int>(index / phase_two_width);
  const float wx = grid_min_x + (static_cast<float>(x2) + 0.5f) * phase_two_cell_size;
  const float wy = grid_min_y + (static_cast<float>(y2) + 0.5f) * phase_two_cell_size;
  const int x1 = static_cast<int>(floorf((wx - grid_min_x) / phase_one_cell_size));
  const int y1 = static_cast<int>(floorf((wy - grid_min_y) / phase_one_cell_size));

  float closest_ground_z = 1.0e9f;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x1 + dx;
      const int ny = y1 + dy;
      if (nx < 0 || nx >= phase_one_width || ny < 0 || ny >= phase_one_height) {
        continue;
      }
      const std::size_t neighbor = static_cast<std::size_t>(ny) * phase_one_width + nx;
      if (phase_one_state[neighbor] != 0) {
        closest_ground_z = fminf(closest_ground_z, phase_one_min_z[neighbor]);
      }
    }
  }
  phase_two_state[index] = (closest_ground_z < 9.0e8f &&
    fabsf(phase_two_min_z[index] - closest_ground_z) <= max_height_deviation) ? 1 : 0;
}

__global__ void ClassifyGSeg3DPointsKernel(
  const KittiPoint * points,
  std::size_t point_count,
  float max_radius_sq,
  float min_candidate_z,
  float max_candidate_z,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  const float * ground_min_z,
  const unsigned char * ground_state,
  float inlier_threshold,
  unsigned char * is_ground_mask)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= point_count) {
    return;
  }
  const KittiPoint point = points[index];
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
      point.z < min_candidate_z || point.z > max_candidate_z ||
      point.x * point.x + point.y * point.y > max_radius_sq) {
    is_ground_mask[index] = 0;
    return;
  }
  const int cx = static_cast<int>(floorf((point.x - grid_min_x) / cell_size));
  const int cy = static_cast<int>(floorf((point.y - grid_min_y) / cell_size));
  if (cx < 0 || cx >= grid_width || cy < 0 || cy >= grid_height) {
    is_ground_mask[index] = 0;
    return;
  }

  float local_ground_z = 1.0e9f;
  int support_count = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
        continue;
      }
      const std::size_t neighbor = static_cast<std::size_t>(ny) * grid_width + nx;
      if (ground_state[neighbor] != 0) {
        local_ground_z = fminf(local_ground_z, ground_min_z[neighbor]);
        ++support_count;
      }
    }
  }
  const float range = sqrtf(point.x * point.x + point.y * point.y);
  const int required_support = range > 25.0f ? 1 : 2;
  const float dz = point.z - local_ground_z;
  is_ground_mask[index] = (support_count >= required_support &&
    dz >= -inlier_threshold && dz <= inlier_threshold) ? 1 : 0;
}
}  // namespace

bool ScorePlanesCuda(
  const std::vector<KittiPoint> & input,
  const std::vector<std::size_t> & candidates,
  const std::vector<CudaPlaneModel> & planes,
  float threshold,
  std::vector<int> & inlier_counts)
{
  inlier_counts.clear();
  if (input.empty() || candidates.empty() || planes.empty() || !HasCudaDevice()) {
    return false;
  }

  KittiPoint * device_points = nullptr;
  std::size_t * device_candidate_indices = nullptr;
  CudaPlaneModel * device_planes = nullptr;
  int * device_counts = nullptr;

  const std::size_t points_bytes = input.size() * sizeof(KittiPoint);
  const std::size_t candidates_bytes = candidates.size() * sizeof(std::size_t);
  const std::size_t planes_bytes = planes.size() * sizeof(CudaPlaneModel);
  const std::size_t counts_bytes = planes.size() * sizeof(int);

  cudaError_t error = cudaMalloc(&device_points, points_bytes);
  if (error != cudaSuccess) {
    return false;
  }
  error = cudaMalloc(&device_candidate_indices, candidates_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_points);
    return false;
  }
  error = cudaMalloc(&device_planes, planes_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_candidate_indices);
    cudaFree(device_points);
    return false;
  }
  error = cudaMalloc(&device_counts, counts_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_planes);
    cudaFree(device_candidate_indices);
    cudaFree(device_points);
    return false;
  }

  error = cudaMemcpy(device_points, input.data(), points_bytes, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_candidate_indices, candidates.data(), candidates_bytes, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_planes, planes.data(), planes_bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(device_counts);
    cudaFree(device_planes);
    cudaFree(device_candidate_indices);
    cudaFree(device_points);
    return false;
  }

  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>((planes.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  ScorePlanesKernel<<<blocks, kThreadsPerBlock>>>(
    device_points,
    device_candidate_indices,
    candidates.size(),
    device_planes,
    planes.size(),
    threshold,
    device_counts);

  error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  if (error != cudaSuccess) {
    cudaFree(device_counts);
    cudaFree(device_planes);
    cudaFree(device_candidate_indices);
    cudaFree(device_points);
    return false;
  }

  inlier_counts.resize(planes.size());
  error = cudaMemcpy(inlier_counts.data(), device_counts, counts_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_counts);
  cudaFree(device_planes);
  cudaFree(device_candidate_indices);
  cudaFree(device_points);

  if (error != cudaSuccess) {
    inlier_counts.clear();
    return false;
  }

  return true;
}

bool BuildMinZGridCuda(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  std::vector<float> & min_z_grid,
  std::vector<unsigned char> & valid_grid)
{
  min_z_grid.clear();
  valid_grid.clear();
  if (input.empty() || grid_width <= 0 || grid_height <= 0 || !HasCudaDevice()) {
    return false;
  }

  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * static_cast<std::size_t>(grid_height);
  const std::size_t points_bytes = input.size() * sizeof(KittiPoint);
  const std::size_t grid_bytes = cell_count * sizeof(float);
  const std::size_t valid_bytes = cell_count * sizeof(unsigned char);

  KittiPoint * device_points = nullptr;
  float * device_min_z = nullptr;
  unsigned char * device_valid = nullptr;

  cudaError_t error = cudaMalloc(&device_points, points_bytes);
  if (error != cudaSuccess) {
    return false;
  }
  error = cudaMalloc(&device_min_z, grid_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_points);
    return false;
  }
  error = cudaMalloc(&device_valid, valid_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_min_z);
    cudaFree(device_points);
    return false;
  }

  const float init_value = 1.0e9f;
  error = cudaMemcpy(device_points, input.data(), points_bytes, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemset(device_valid, 0, valid_bytes);
  }
  if (error == cudaSuccess) {
    std::vector<float> init_grid(cell_count, init_value);
    error = cudaMemcpy(device_min_z, init_grid.data(), grid_bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(device_valid);
    cudaFree(device_min_z);
    cudaFree(device_points);
    return false;
  }

  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>((input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const float max_radius = fmaxf(params.max_candidate_radius_m, 1.0f);
  BuildMinZGridKernel<<<blocks, kThreadsPerBlock>>>(
    device_points,
    input.size(),
    params.min_candidate_z,
    params.max_candidate_z,
    max_radius * max_radius,
    grid_min_x,
    grid_min_y,
    grid_width,
    grid_height,
    cell_size,
    device_min_z,
    device_valid);

  error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  if (error != cudaSuccess) {
    cudaFree(device_valid);
    cudaFree(device_min_z);
    cudaFree(device_points);
    return false;
  }

  min_z_grid.resize(cell_count);
  valid_grid.resize(cell_count);
  error = cudaMemcpy(min_z_grid.data(), device_min_z, grid_bytes, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(valid_grid.data(), device_valid, valid_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(device_valid);
  cudaFree(device_min_z);
  cudaFree(device_points);

  if (error != cudaSuccess) {
    min_z_grid.clear();
    valid_grid.clear();
    return false;
  }

  return true;
}

bool ClassifyGroundPointsGridCuda(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params,
  float grid_min_x,
  float grid_min_y,
  int grid_width,
  int grid_height,
  float cell_size,
  const std::vector<float> & accepted_ground_z_grid,
  const std::vector<unsigned char> & accepted_ground_valid_grid,
  std::vector<unsigned char> & is_ground_mask)
{
  is_ground_mask.clear();
  if (
    input.empty() || grid_width <= 0 || grid_height <= 0 ||
    accepted_ground_z_grid.empty() || accepted_ground_valid_grid.empty() ||
    !HasCudaDevice())
  {
    return false;
  }

  const std::size_t cell_count = static_cast<std::size_t>(grid_width) * static_cast<std::size_t>(grid_height);
  if (accepted_ground_z_grid.size() != cell_count || accepted_ground_valid_grid.size() != cell_count) {
    return false;
  }

  const std::size_t points_bytes = input.size() * sizeof(KittiPoint);
  const std::size_t grid_bytes = cell_count * sizeof(float);
  const std::size_t valid_bytes = cell_count * sizeof(unsigned char);
  const std::size_t mask_bytes = input.size() * sizeof(unsigned char);

  KittiPoint * device_points = nullptr;
  float * device_ground_z = nullptr;
  unsigned char * device_ground_valid = nullptr;
  unsigned char * device_mask = nullptr;

  cudaError_t error = cudaMalloc(&device_points, points_bytes);
  if (error != cudaSuccess) {
    return false;
  }
  error = cudaMalloc(&device_ground_z, grid_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_points);
    return false;
  }
  error = cudaMalloc(&device_ground_valid, valid_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_ground_z);
    cudaFree(device_points);
    return false;
  }
  error = cudaMalloc(&device_mask, mask_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_ground_valid);
    cudaFree(device_ground_z);
    cudaFree(device_points);
    return false;
  }

  error = cudaMemcpy(device_points, input.data(), points_bytes, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_ground_z, accepted_ground_z_grid.data(), grid_bytes, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_ground_valid, accepted_ground_valid_grid.data(), valid_bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(device_mask);
    cudaFree(device_ground_valid);
    cudaFree(device_ground_z);
    cudaFree(device_points);
    return false;
  }

  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>((input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const float max_radius = fmaxf(params.max_candidate_radius_m, 1.0f);
  ClassifyGroundPointsGridKernel<<<blocks, kThreadsPerBlock>>>(
    device_points,
    input.size(),
    max_radius * max_radius,
    0.5f,
    4.0f,
    1.4f,
    0.2f,
    grid_min_x,
    grid_min_y,
    grid_width,
    grid_height,
    cell_size,
    device_ground_z,
    device_ground_valid,
    fmaxf(params.final_distance_threshold_m, 0.05f),
    params.max_above_plane_m,
    params.max_below_plane_m,
    params.local_ground_margin_m,
    params.adaptive_threshold_scale,
    params.max_adaptive_threshold_m,
    device_mask);

  error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  if (error != cudaSuccess) {
    cudaFree(device_mask);
    cudaFree(device_ground_valid);
    cudaFree(device_ground_z);
    cudaFree(device_points);
    return false;
  }

  is_ground_mask.resize(input.size());
  error = cudaMemcpy(is_ground_mask.data(), device_mask, mask_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_mask);
  cudaFree(device_ground_valid);
  cudaFree(device_ground_z);
  cudaFree(device_points);

  if (error != cudaSuccess) {
    is_ground_mask.clear();
    return false;
  }

  return true;
}

bool ClassifyGroundPointsGSeg3DCuda(
  const std::vector<KittiPoint> & input,
  const GroundRemovalParams & params,
  std::vector<unsigned char> & is_ground_mask)
{
  is_ground_mask.clear();
  if (input.empty() || !HasCudaDevice()) {
    return false;
  }

  const float max_radius = fmaxf(params.max_candidate_radius_m, 1.0f);
  const float phase_one_cell_size = fmaxf(params.gseg3d_cell_size_x_m, 0.25f);
  const float phase_two_cell_size = fmaxf(params.gseg3d_cuda_cell_size_m, 0.25f);
  // Match the CPU workspace limit before converting dimensions to integers.
  if (!std::isfinite(max_radius) ||
      ceilf((2.0f * max_radius) / phase_one_cell_size) + 1 > 2048 ||
      ceilf((2.0f * max_radius) / phase_two_cell_size) + 1 > 2048) return false;
  const float grid_min_x = -max_radius;
  const float grid_min_y = -max_radius;
  const int phase_one_width = static_cast<int>(ceilf((2.0f * max_radius) / phase_one_cell_size)) + 1;
  const int phase_two_width = static_cast<int>(ceilf((2.0f * max_radius) / phase_two_cell_size)) + 1;
  const int phase_one_height = phase_one_width;
  const int phase_two_height = phase_two_width;
  const std::size_t phase_one_cells = static_cast<std::size_t>(phase_one_width) * phase_one_height;
  const std::size_t phase_two_cells = static_cast<std::size_t>(phase_two_width) * phase_two_height;

  struct Workspace
  {
    CudaBuffer<KittiPoint> points;
    CudaBuffer<float> coarse_minimum, fine_minimum;
    CudaBuffer<unsigned char> coarse_valid, fine_valid, state_a, state_b, fine_state, mask;
  };
  thread_local Workspace workspace;
  if (!workspace.points.Reserve(input.size()) ||
      !workspace.coarse_minimum.Reserve(phase_one_cells) ||
      !workspace.fine_minimum.Reserve(phase_two_cells) ||
      !workspace.coarse_valid.Reserve(phase_one_cells) ||
      !workspace.fine_valid.Reserve(phase_two_cells) ||
      !workspace.state_a.Reserve(phase_one_cells) ||
      !workspace.state_b.Reserve(phase_one_cells) ||
      !workspace.fine_state.Reserve(phase_two_cells) ||
      !workspace.mask.Reserve(input.size())) return false;
  auto * device_points = workspace.points.Get();
  auto * phase_one_min_z = workspace.coarse_minimum.Get();
  auto * phase_two_min_z = workspace.fine_minimum.Get();
  auto * phase_one_valid = workspace.coarse_valid.Get();
  auto * phase_two_valid = workspace.fine_valid.Get();
  auto * phase_one_state_a = workspace.state_a.Get();
  auto * phase_one_state_b = workspace.state_b.Get();
  auto * phase_two_state = workspace.fine_state.Get();
  auto * device_mask = workspace.mask.Get();

  const std::size_t points_bytes = input.size() * sizeof(KittiPoint);
  const std::size_t mask_bytes = input.size() * sizeof(unsigned char);
  cudaError_t error = cudaMemcpy(device_points, input.data(), points_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return false;

  constexpr int kThreadsPerBlock = 256;
  const int point_blocks = static_cast<int>((input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const int phase_one_blocks = static_cast<int>((phase_one_cells + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const int phase_two_blocks = static_cast<int>((phase_two_cells + kThreadsPerBlock - 1) / kThreadsPerBlock);
  InitializeGroundGridKernel<<<phase_one_blocks, kThreadsPerBlock>>>(
    phase_one_min_z, phase_one_valid, phase_one_cells);
  InitializeGroundGridKernel<<<phase_two_blocks, kThreadsPerBlock>>>(
    phase_two_min_z, phase_two_valid, phase_two_cells);
  BuildMinZGridKernel<<<point_blocks, kThreadsPerBlock>>>(
    device_points, input.size(), params.min_candidate_z, params.max_candidate_z,
    max_radius * max_radius, grid_min_x, grid_min_y, phase_one_width, phase_one_height,
    phase_one_cell_size, phase_one_min_z, phase_one_valid);
  BuildMinZGridKernel<<<point_blocks, kThreadsPerBlock>>>(
    device_points, input.size(), params.min_candidate_z, params.max_candidate_z,
    max_radius * max_radius, grid_min_x, grid_min_y, phase_two_width, phase_two_height,
    phase_two_cell_size, phase_two_min_z, phase_two_valid);

  const float seed_tolerance = fmaxf(params.gseg3d_ground_inlier_threshold_m * 2.0f, 0.25f);
  SeedGSeg3DGridKernel<<<phase_one_blocks, kThreadsPerBlock>>>(
    phase_one_min_z, phase_one_valid, phase_one_width, phase_one_height,
    grid_min_x, grid_min_y, phase_one_cell_size, params.gseg3d_ground_height_m,
    fmaxf(params.gseg3d_cuda_seed_radius_m, phase_one_cell_size), seed_tolerance, phase_one_state_a);

  const float slope_radians = fminf(fmaxf(params.gseg3d_slope_threshold_deg, 0.0f), 45.0f) *
    3.14159265358979323846f / 180.0f;
  const float maximum_step = fmaxf(
    params.gseg3d_max_ground_height_deviation_m,
    tanf(slope_radians) * phase_one_cell_size * 1.2f);
  unsigned char * current_state = phase_one_state_a;
  unsigned char * next_state = phase_one_state_b;
  const int expansion_iterations = std::min(
    256, static_cast<int>(ceilf(max_radius / phase_one_cell_size)) + 1);
  for (int iteration = 0; iteration < expansion_iterations; ++iteration) {
    ExpandGSeg3DGridKernel<<<phase_one_blocks, kThreadsPerBlock>>>(
      phase_one_min_z, phase_one_valid, current_state, phase_one_width, phase_one_height,
      maximum_step, next_state);
    unsigned char * temporary = current_state;
    current_state = next_state;
    next_state = temporary;
  }

  const float refinement_deviation = fmaxf(
    params.gseg3d_max_ground_height_deviation_m * 1.5f,
    params.gseg3d_ground_inlier_threshold_m);
  RefineGSeg3DGridKernel<<<phase_two_blocks, kThreadsPerBlock>>>(
    phase_two_min_z, phase_two_valid, phase_two_width, phase_two_height,
    grid_min_x, grid_min_y, phase_two_cell_size,
    phase_one_min_z, current_state, phase_one_width, phase_one_height, phase_one_cell_size,
    refinement_deviation, phase_two_state);
  ClassifyGSeg3DPointsKernel<<<point_blocks, kThreadsPerBlock>>>(
    device_points, input.size(), max_radius * max_radius,
    params.min_candidate_z, params.max_candidate_z,
    grid_min_x, grid_min_y, phase_two_width, phase_two_height, phase_two_cell_size,
    phase_two_min_z, phase_two_state,
    fmaxf(params.gseg3d_ground_inlier_threshold_m, 0.05f), device_mask);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return false;
  }

  is_ground_mask.resize(input.size());
  error = cudaMemcpy(is_ground_mask.data(), device_mask, mask_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    is_ground_mask.clear();
    return false;
  }
  return true;
}
}  // namespace lidar_viewer::algorithms
