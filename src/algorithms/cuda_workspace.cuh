#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace lidar_viewer::algorithms
{
// Each calling thread owns its workspace. Retain capacity between frames,
// while allowing concurrent offline processing without sharing device buffers.
template<typename T>
class CudaBuffer
{
public:
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer & operator=(const CudaBuffer &) = delete;
  ~CudaBuffer() { if (data_) cudaFree(data_); }

  bool Reserve(std::size_t count)
  {
    if (count <= capacity_) return true;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return false;
    T * replacement = nullptr;
    if (cudaMalloc(&replacement, count * sizeof(T)) != cudaSuccess) return false;
    if (data_) cudaFree(data_);
    data_ = replacement;
    capacity_ = count;
    return true;
  }

  T * Get() const { return data_; }

private:
  T * data_{nullptr};
  std::size_t capacity_{0};
};
}  // namespace lidar_viewer::algorithms
