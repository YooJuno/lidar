// SPDX-License-Identifier: Apache-2.0
// Portable C ABI: no Python/NumPy ABI, CUDA, or external library dependencies.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <unordered_map>

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

namespace {
struct NmsPoint {double x,y;};
struct NmsBox {
  std::array<NmsPoint,4> corners;
  double x,y,area,min_x,min_y,max_x,max_y;
  int label;
};
std::uint64_t NmsCell(int x,int y) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x))<<32)|static_cast<std::uint32_t>(y);
}
NmsBox PrepareBox(const float * p) {
  NmsBox b{};b.x=p[0];b.y=p[1];b.label=static_cast<int>(p[8]);b.area=double(p[3])*p[4];
  const double c=std::cos(double(p[6])),s=std::sin(double(p[6]));
  b.min_x=b.min_y=1e30;b.max_x=b.max_y=-1e30;
  constexpr double signs[4][2]={{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
  for (std::size_t i=0;i<4;++i) {
    const double x=signs[i][0]*p[3],y=signs[i][1]*p[4];
    b.corners[i]={b.x+c*x-s*y,b.y+s*x+c*y};
    b.min_x=std::min(b.min_x,b.corners[i].x);b.max_x=std::max(b.max_x,b.corners[i].x);
    b.min_y=std::min(b.min_y,b.corners[i].y);b.max_y=std::max(b.max_y,b.corners[i].y);
  }
  return b;
}
double NmsIou(const NmsBox & a,const NmsBox & b) {
  if (a.area<1e-6 || b.area<1e-6 || std::min(a.max_x,b.max_x)<=std::max(a.min_x,b.min_x) ||
      std::min(a.max_y,b.max_y)<=std::max(a.min_y,b.min_y)) return 0;
  std::array<NmsPoint,12> polygon{},clipped{};std::copy(a.corners.begin(),a.corners.end(),polygon.begin());std::size_t count=4;
  for (std::size_t edge=0;edge<4 && count;++edge) {
    const auto u=b.corners[edge],v=b.corners[(edge+1)%4];const double ex=v.x-u.x,ey=v.y-u.y;
    const auto side=[&](const NmsPoint & p){return ex*(p.y-u.y)-ey*(p.x-u.x);};
    auto previous=polygon[count-1];double previous_side=side(previous);std::size_t size=0;
    for (std::size_t i=0;i<count;++i) {
      const auto current=polygon[i];const double current_side=side(current);
      if ((current_side>=0)!=(previous_side>=0)) {
        const double t=previous_side/(previous_side-current_side);
        clipped[size++]={previous.x+(current.x-previous.x)*t,previous.y+(current.y-previous.y)*t};
      }
      if (current_side>=0) clipped[size++]=current;
      previous=current;previous_side=current_side;
    }
    polygon=clipped;count=size;
  }
  if (count<3) return 0;
  double area=0;
  for (std::size_t i=0;i<count;++i) area+=polygon[i].x*polygon[(i+1)%count].y-polygon[i].y*polygon[(i+1)%count].x;
  area=std::abs(area)*.5;const double united=a.area+b.area-area;
  return area<1e-6 || united<.01 ? 0 : std::min(1.,area/united);
}
}  // namespace

// Autoware's two separate stages: greedy circle NMS, then BEV IoU NMS
// against all prior circle survivors (including ones rejected by IoU).
// Pedestrians do not suppress different semantic classes in the IoU stage.
// Input is already in stable descending-score order; output indexes it.
LIDAR_NATIVE_EXPORT int lidar_centerpoint_nms_v1(const float * boxes,std::size_t n,
  std::int64_t * output,std::size_t * count)
{
  if (n>65536 || !count || (n && (!boxes || !output))) return 1;
  *count=0;
  for (std::size_t i=0;i<n;++i) {
    const auto * p=boxes+i*9;
    for (int j=0;j<9;++j) if (!std::isfinite(p[j])) return 1;
    if (std::abs(p[0])>1e6 || std::abs(p[1])>1e6 || p[3]<=0 || p[4]<=0 || p[3]>100 || p[4]>100 ||
        p[8]<0 || p[8]>4 || std::floor(p[8])!=p[8]) return 1;
  }
  try {
    using Cells=std::unordered_map<std::uint64_t,std::vector<std::size_t>>;
    Cells circle,iou;circle.reserve(n);iou.reserve(n);
    std::vector<NmsBox> survivors;survivors.reserve(n);
    for (std::size_t i=0;i<n;++i) {
      const auto * p=boxes+i*9;const int cx=static_cast<int>(std::floor(p[0]*2)),cy=static_cast<int>(std::floor(p[1]*2));
      bool rejected=false;
      for (int y=cy-1;y<=cy+1 && !rejected;++y) for (int x=cx-1;x<=cx+1 && !rejected;++x) {
        const auto found=circle.find(NmsCell(x,y));if (found==circle.end()) continue;
        for (const auto j:found->second) {
          const float dx=p[0]-boxes[j*9],dy=p[1]-boxes[j*9+1];
          if (dx*dx+dy*dy<.25f) {rejected=true;break;}
        }
      }
      if (rejected) continue;
      circle[NmsCell(cx,cy)].push_back(i);
      const auto b=PrepareBox(p);const int bx=static_cast<int>(std::floor(b.x/10)),by=static_cast<int>(std::floor(b.y/10));
      for (int y=by-1;y<=by+1 && !rejected;++y) for (int x=bx-1;x<=bx+1 && !rejected;++x) {
        const auto found=iou.find(NmsCell(x,y));if (found==iou.end()) continue;
        for (const auto j:found->second) {
          const auto & a=survivors[j];const double dx=b.x-a.x,dy=b.y-a.y;
          if ((b.label!=a.label && (b.label==4 || a.label==4)) || dx*dx+dy*dy>100) continue;
          if (NmsIou(a,b)>.1) {rejected=true;break;}
        }
      }
      // IoU-rejected objects must still participate in this second stage.
      iou[NmsCell(bx,by)].push_back(survivors.size());survivors.push_back(b);
      if (!rejected) output[(*count)++]=static_cast<std::int64_t>(i);
    }
    return 0;
  } catch (...) {return 2;}
}
