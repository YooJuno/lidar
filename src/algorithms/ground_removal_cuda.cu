#include "lidar_viewer/algorithms/ground_removal_cuda.hpp"

#include <cuda_runtime.h>

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
  if (point.z < min_candidate_z || point.z > max_candidate_z) {
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

  if (local_ground_z > 9.0e8f || support_count < 2) {
    is_ground_mask[index] = 0;
    return;
  }

  if ((highest_ground_z - local_ground_z) > (max_above_plane_m * 1.5f)) {
    is_ground_mask[index] = 0;
    return;
  }

  const float radius = sqrtf(radius_sq);
  const float distance_scaled = final_distance_threshold_m + fmaxf(adaptive_threshold_scale, 0.0f) * radius;
  const float allowed_above = fminf(
    fmaxf(distance_scaled, final_distance_threshold_m),
    fmaxf(max_adaptive_threshold_m, final_distance_threshold_m));
  const float dz = point.z - local_ground_z;
  const float upper_bound = fminf(fmaxf(max_above_plane_m, allowed_above), allowed_above);
  const float lower_bound = -fmaxf(max_below_plane_m, 0.1f);
  is_ground_mask[index] = (dz <= upper_bound && dz >= lower_bound) ? 1 : 0;
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
}  // namespace lidar_viewer::algorithms
