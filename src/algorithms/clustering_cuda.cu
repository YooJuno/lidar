#include "lidar_viewer/algorithms/clustering_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace lidar_viewer::algorithms
{
namespace
{
struct DevicePoint
{
  float x;
  float y;
  float z;
  float intensity;
};

__device__ std::int64_t DeviceGridKey(int x, int y)
{
  return (static_cast<std::int64_t>(x) << 32) ^ static_cast<std::uint32_t>(y);
}

__global__ void BuildAssignmentsKernel(
  const DevicePoint * points,
  CudaCellAssignment * assignments,
  std::size_t count,
  float cell_size)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int cell_x = static_cast<int>(floorf(points[index].x / cell_size));
  const int cell_y = static_cast<int>(floorf(points[index].y / cell_size));
  assignments[index].key = DeviceGridKey(cell_x, cell_y);
  assignments[index].point_index = index;
}

bool HasCudaDevice()
{
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}
}  // namespace

bool BuildCellAssignmentsCuda(
  const std::vector<KittiPoint> & input,
  const ClusteringParams & params,
  std::vector<CudaCellAssignment> & assignments)
{
  assignments.clear();
  if (input.empty() || !HasCudaDevice()) {
    return false;
  }

  const float cell_size = std::max(params.grid_cell_m, 0.05f);
  DevicePoint * device_points = nullptr;
  CudaCellAssignment * device_assignments = nullptr;
  const std::size_t points_bytes = input.size() * sizeof(DevicePoint);
  const std::size_t assignments_bytes = input.size() * sizeof(CudaCellAssignment);

  cudaError_t error = cudaMalloc(&device_points, points_bytes);
  if (error != cudaSuccess) {
    return false;
  }

  error = cudaMalloc(&device_assignments, assignments_bytes);
  if (error != cudaSuccess) {
    cudaFree(device_points);
    return false;
  }

  error = cudaMemcpy(device_points, input.data(), points_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(device_assignments);
    cudaFree(device_points);
    return false;
  }

  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>((input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  BuildAssignmentsKernel<<<blocks, kThreadsPerBlock>>>(
    device_points,
    device_assignments,
    input.size(),
    cell_size);
  error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  if (error != cudaSuccess) {
    cudaFree(device_assignments);
    cudaFree(device_points);
    return false;
  }

  assignments.resize(input.size());
  error = cudaMemcpy(assignments.data(), device_assignments, assignments_bytes, cudaMemcpyDeviceToHost);
  cudaFree(device_assignments);
  cudaFree(device_points);
  if (error != cudaSuccess) {
    assignments.clear();
    return false;
  }

  return true;
}
}  // namespace lidar_viewer::algorithms
