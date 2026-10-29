#pragma once

#include "lidar_viewer/roi.hpp"
#include "lidar_viewer/camera_2d.hpp"
#include <algorithm>
#include <deque>
#include <optional>
#include <utility>

namespace lidar_viewer
{
inline bool SameRoi(const RoiSettings & a, const RoiSettings & b)
{
  if (a.enabled != b.enabled || a.show_boxes != b.show_boxes || a.boxes.size() != b.boxes.size()) return false;
  for (std::size_t i=0; i<a.boxes.size(); ++i) {
    const auto & x=a.boxes[i]; const auto & y=b.boxes[i];
    if (x.name!=y.name || x.enabled!=y.enabled || x.yaw_deg!=y.yaw_deg ||
        x.center.x!=y.center.x || x.center.y!=y.center.y || x.center.z!=y.center.z ||
        x.size.x!=y.size.x || x.size.y!=y.size.y || x.size.z!=y.size.z) return false;
  }
  return true;
}

class RoiHistory
{
public:
  void Record(const RoiSettings & before) {Push(undo_, before); redo_.clear();}
  bool Undo(RoiSettings & current) {return Restore(undo_, redo_, current);}
  bool Redo(RoiSettings & current) {return Restore(redo_, undo_, current);}
  bool CanUndo() const {return !undo_.empty();}
  bool CanRedo() const {return !redo_.empty();}
  void Clear() {undo_.clear(); redo_.clear();}
private:
  static void Push(std::deque<RoiSettings> & target, const RoiSettings & value)
  {
    if (!target.empty() && SameRoi(target.back(), value)) return;
    if (target.size() == 64) target.pop_front();
    target.push_back(value);
  }
  static bool Restore(std::deque<RoiSettings> & from, std::deque<RoiSettings> & to, RoiSettings & current)
  {
    if (from.empty()) return false;
    Push(to, current); current=std::move(from.back()); from.pop_back(); return true;
  }
  std::deque<RoiSettings> undo_, redo_;
};

inline std::string NextRoiName(const RoiSettings & settings)
{
  for (std::size_t i=1; ; ++i) {
    const auto name="ROI " + std::to_string(i);
    if (std::none_of(settings.boxes.begin(), settings.boxes.end(),
        [&](const auto & box) {return box.name==name;})) return name;
  }
}

inline RoiBox DuplicateRoi(const RoiBox & source, const RoiSettings & settings)
{
  auto result=source; result.name=NextRoiName(settings);
  const float yaw=source.yaw_deg*kPi/180.f;
  result.center=result.center + Vec3{std::cos(yaw), std::sin(yaw), 0}*(source.size.x+.5f);
  ValidateRoiBox(result);
  return result;
}

inline RoiBox RoiFromDrag(Vec3 a, Vec3 b, float center_z, float height, const std::string & name,
  float yaw_deg=0)
{
  const float yaw=yaw_deg*kPi/180.f;
  const auto delta=b-a;
  RoiBox box{name, true, {(a.x+b.x)*.5f, (a.y+b.y)*.5f, center_z},
    {std::max(.05f,std::abs(delta.x*std::cos(yaw)+delta.y*std::sin(yaw))),
     std::max(.05f,std::abs(-delta.x*std::sin(yaw)+delta.y*std::cos(yaw))),height},yaw_deg};
  ValidateRoiBox(box); return box;
}

// Resize in the box's own axes, holding the opposite edge/corner fixed.
inline RoiBox ResizeRoi(const RoiBox & original, Vec3 pointer, int handle_x, int handle_y)
{
  auto box=original;
  const float yaw=box.yaw_deg*kPi/180.f,c=std::cos(yaw),s=std::sin(yaw);
  const auto delta=pointer-original.center;
  const float x=delta.x*c+delta.y*s,y=-delta.x*s+delta.y*c;
  float low_x=-box.size.x*.5f,high_x=-low_x,low_y=-box.size.y*.5f,high_y=-low_y;
  if (handle_x<0) low_x=std::min(x,high_x-.05f);
  if (handle_x>0) high_x=std::max(x,low_x+.05f);
  if (handle_y<0) low_y=std::min(y,high_y-.05f);
  if (handle_y>0) high_y=std::max(y,low_y+.05f);
  const float center_x=(low_x+high_x)*.5f,center_y=(low_y+high_y)*.5f;
  box.center=original.center+Vec3{center_x*c-center_y*s,center_x*s+center_y*c,0};
  box.size.x=high_x-low_x;box.size.y=high_y-low_y;
  ValidateRoiBox(box);return box;
}

inline RoiBox RotateRoi(const RoiBox & original, Vec3 start, Vec3 pointer, bool snap=false)
{
  auto box=original;
  const auto a=start-box.center,b=pointer-box.center;
  if (a.x*a.x+a.y*a.y<1e-8f || b.x*b.x+b.y*b.y<1e-8f) return box;
  box.yaw_deg=std::remainder(original.yaw_deg+(std::atan2(b.y,b.x)-std::atan2(a.y,a.x))*180.f/kPi,360.f);
  if (snap) box.yaw_deg=std::round(box.yaw_deg/15.f)*15.f;
  return box;
}

// Normalized viewport coordinates: left/bottom=-1, right/top=+1. Intersect
// the orthographic display ray with a raw LiDAR Z plane, including IMU tilt.
inline std::optional<Vec3> RoiPointFromTopView(float nx, float ny, float aspect,
  float half_height, Vec3 target, float yaw, const RoiCoordinateFrame & frame, float raw_z)
{
  const Vec3 right{std::cos(yaw),std::sin(yaw),0};
  const auto origin=frame.InverseTransform(target+right*(nx*half_height*aspect)+TopViewUp(yaw)*(ny*half_height));
  const auto direction=frame.InverseTransform(frame.translation+Vec3{0,0,-1});
  if (std::abs(direction.z)<.05f) return {};
  const auto point=origin+direction*((raw_z-origin.z)/direction.z);
  if (!std::isfinite(point.x) || !std::isfinite(point.y) || std::abs(point.x)>1e6f || std::abs(point.y)>1e6f) return {};
  return point;
}
}  // namespace lidar_viewer
