// SPDX-License-Identifier: Apache-2.0
// Portable C ABI: no Python/NumPy ABI, CUDA, or external library dependencies.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

#ifdef _WIN32
#define LIDAR_NATIVE_EXPORT extern "C" __declspec(dllexport)
#else
#define LIDAR_NATIVE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::size_t kMaxScan=4000000, kMaxVoxels=64000;
}

LIDAR_NATIVE_EXPORT int lidar_pillar_abi() {return 2;}

// Stable counting sort over the model's bounded integer grid. Preserve the
// caller's deterministic shuffle within each pillar, including its first 32.
LIDAR_NATIVE_EXPORT int lidar_pillar_order_v2(const std::int32_t * keys,
  const std::int64_t * permutation,std::size_t n,std::int64_t * output,std::uint32_t grid)
{
  if (!grid || grid>1024 || n>kMaxScan || (n && (!keys || !permutation || !output))) return 1;
  if (!n) return 0;
  const std::size_t cells=grid*grid;
  try {
    thread_local std::vector<std::uint32_t> offsets;
    offsets.resize(cells+1);
    std::fill(offsets.begin(),offsets.end(),0);
    for (std::size_t i=0;i<n;++i) {
      const auto index=permutation[i];
      if (index<0 || static_cast<std::size_t>(index)>=n) return 1;
      const auto key=keys[index];
      if (key<0 || static_cast<std::size_t>(key)>=cells) return 1;
      ++offsets[key+1];
    }
    for (std::size_t i=1;i<offsets.size();++i) offsets[i]+=offsets[i-1];
    for (std::size_t i=0;i<n;++i) {
      const auto index=permutation[i];output[offsets[keys[index]]++]=index;
    }
    return 0;
  } catch (...) {return 2;}
}

LIDAR_NATIVE_EXPORT int lidar_pillar_order(const std::int32_t * keys,
  const std::int64_t * permutation,std::size_t n,std::int64_t * output)
{return lidar_pillar_order_v2(keys,permutation,n,output,480);}

// The caller zeroes padding. Match NumPy's double bincount -> float32 cast ->
// float32 mean division and each float32 coordinate operation exactly.
LIDAR_NATIVE_EXPORT int lidar_pillar_features_v2(const float * points,
  const std::int64_t * order,std::size_t n,const std::int64_t * starts,
  const std::int64_t * counts,const std::int32_t * keys,std::size_t num,float * features,
  std::uint32_t grid,float voxel,float min_x,float min_y,float min_z,float voxel_z,std::uint32_t dimensions)
{
  if (!grid || grid>1024 || (dimensions!=9 && dimensions!=10) ||
      !std::isfinite(voxel) || voxel<=0 || !std::isfinite(voxel_z) || voxel_z<=0 ||
      !std::isfinite(min_x) || !std::isfinite(min_y) || !std::isfinite(min_z) ||
      n>kMaxScan || num>kMaxVoxels ||
      (num && (!points || !order || !starts || !counts || !keys || !features))) return 1;
  for (std::size_t cell=0;cell<num;++cell) {
    const auto start=starts[cell];const auto count=std::min<std::int64_t>(counts[cell],32);
    if (start<0 || count<=0 || static_cast<std::size_t>(start)>n ||
        static_cast<std::size_t>(count)>n-static_cast<std::size_t>(start) ||
        keys[cell]<0 || static_cast<std::size_t>(keys[cell])>=grid*grid) return 1;
    double sum[3]{};
    for (std::int64_t rank=0;rank<count;++rank) {
      const auto index=order[start+rank];
      if (index<0 || static_cast<std::size_t>(index)>=n) return 1;
      for (int axis=0;axis<3;++axis) sum[axis]+=points[index*3+axis];
    }
    const float mean[]={static_cast<float>(sum[0])/static_cast<float>(count),
      static_cast<float>(sum[1])/static_cast<float>(count),static_cast<float>(sum[2])/static_cast<float>(count)};
    const float center_x=(static_cast<float>(keys[cell]%grid)+.5f)*voxel+min_x;
    const float center_y=(static_cast<float>(keys[cell]/grid)+.5f)*voxel+min_y;
    const float center_z=voxel_z*.5f+min_z;
    for (std::int64_t rank=0;rank<count;++rank) {
      const auto * point=points+order[start+rank]*3;
      auto * f=features+(cell*32+rank)*dimensions;
      f[0]=point[0];f[1]=point[1];f[2]=point[2];f[3]=0;
      f[4]=point[0]-mean[0];f[5]=point[1]-mean[1];f[6]=point[2]-mean[2];
      f[7]=point[0]-center_x;f[8]=point[1]-center_y;
      if (dimensions==10) f[9]=point[2]-center_z;
    }
  }
  return 0;
}

LIDAR_NATIVE_EXPORT int lidar_pillar_features(const float * points,
  const std::int64_t * order,std::size_t n,const std::int64_t * starts,
  const std::int64_t * counts,const std::int32_t * keys,std::size_t num,float * features)
{return lidar_pillar_features_v2(points,order,n,starts,counts,keys,num,features,480,.32f,-76.8f,-76.8f,-4.f,10.f,9);}
