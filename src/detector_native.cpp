// SPDX-License-Identifier: Apache-2.0
// Portable C ABI: no Python/NumPy ABI, CUDA, or external library dependencies.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#define LIDAR_NATIVE_EXPORT extern "C" __declspec(dllexport)
#else
#define LIDAR_NATIVE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::size_t kCells=480*480, kMaxScan=4000000, kMaxVoxels=40000;
}

LIDAR_NATIVE_EXPORT int lidar_pillar_abi() {return 1;}

// Stable counting sort over the model's bounded integer grid. Preserve the
// caller's deterministic shuffle within each pillar, including its first 32.
LIDAR_NATIVE_EXPORT int lidar_pillar_order(const std::int32_t * keys,
  const std::int64_t * permutation,std::size_t n,std::int64_t * output)
{
  if (n>kMaxScan || (n && (!keys || !permutation || !output))) return 1;
  if (!n) return 0;
  try {
    thread_local std::vector<std::uint32_t> offsets(kCells+1);
    std::fill(offsets.begin(),offsets.end(),0);
    for (std::size_t i=0;i<n;++i) {
      const auto index=permutation[i];
      if (index<0 || static_cast<std::size_t>(index)>=n) return 1;
      const auto key=keys[index];
      if (key<0 || static_cast<std::size_t>(key)>=kCells) return 1;
      ++offsets[key+1];
    }
    for (std::size_t i=1;i<offsets.size();++i) offsets[i]+=offsets[i-1];
    for (std::size_t i=0;i<n;++i) {
      const auto index=permutation[i];output[offsets[keys[index]]++]=index;
    }
    return 0;
  } catch (...) {return 2;}
}

// The caller zeroes padding. Match NumPy's double bincount -> float32 cast ->
// float32 mean division and each float32 coordinate operation exactly.
LIDAR_NATIVE_EXPORT int lidar_pillar_features(const float * points,
  const std::int64_t * order,std::size_t n,const std::int64_t * starts,
  const std::int64_t * counts,const std::int32_t * keys,std::size_t num,float * features)
{
  if (n>kMaxScan || num>kMaxVoxels ||
      (num && (!points || !order || !starts || !counts || !keys || !features))) return 1;
  for (std::size_t cell=0;cell<num;++cell) {
    const auto start=starts[cell];const auto count=std::min<std::int64_t>(counts[cell],32);
    if (start<0 || count<=0 || static_cast<std::size_t>(start)>n ||
        static_cast<std::size_t>(count)>n-static_cast<std::size_t>(start) ||
        keys[cell]<0 || static_cast<std::size_t>(keys[cell])>=kCells) return 1;
    double sum[3]{};
    for (std::int64_t rank=0;rank<count;++rank) {
      const auto index=order[start+rank];
      if (index<0 || static_cast<std::size_t>(index)>=n) return 1;
      for (int axis=0;axis<3;++axis) sum[axis]+=points[index*3+axis];
    }
    const float mean[]={static_cast<float>(sum[0])/static_cast<float>(count),
      static_cast<float>(sum[1])/static_cast<float>(count),static_cast<float>(sum[2])/static_cast<float>(count)};
    const float center_x=(static_cast<float>(keys[cell]%480)+.5f)*.32f-76.8f;
    const float center_y=(static_cast<float>(keys[cell]/480)+.5f)*.32f-76.8f;
    for (std::int64_t rank=0;rank<count;++rank) {
      const auto * point=points+order[start+rank]*3;
      auto * f=features+(cell*32+rank)*9;
      f[0]=point[0];f[1]=point[1];f[2]=point[2];f[3]=0;
      f[4]=point[0]-mean[0];f[5]=point[1]-mean[1];f[6]=point[2]-mean[2];
      f[7]=point[0]-center_x;f[8]=point[1]-center_y;
    }
  }
  return 0;
}
