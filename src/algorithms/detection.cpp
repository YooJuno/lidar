#include "lidar_viewer/algorithms/detection.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lidar_viewer::algorithms
{
const char * DetectionClassName(int class_id)
{
  switch (class_id) {
    case DetectionClassCar:return "Car";
    case DetectionClassTruck:return "Truck";
    case DetectionClassBus:return "Bus";
    case DetectionClassPerson:return "Person";
    case DetectionClassCyclist:return "Bicycle";
    default:return "Unknown";
  }
}
Detection DecodeCenterPointBox(const float * box, float display_yaw)
{
  for (int i=0;i<9;++i) if (!std::isfinite(box[i])) throw std::runtime_error("Non-finite detector output");
  const int label=static_cast<int>(box[8]);
  if (label<0 || label>4 || box[8]!=label || box[3]<=0 || box[4]<=0 || box[5]<=0 ||
      box[3]>100 || box[4]>100 || box[5]>30 || box[7]<0 || box[7]>1)
    throw std::runtime_error("Invalid detector box");
  constexpr int classes[]={DetectionClassCar,DetectionClassTruck,DetectionClassBus,DetectionClassCyclist,DetectionClassPerson};
  Detection detection;
  detection.class_id=classes[label];detection.confidence=box[7];
  auto & oriented=detection.oriented_bounds;
  const float c=std::cos(display_yaw),s=std::sin(display_yaw),yaw=box[6]+display_yaw;
  oriented.center={c*box[0]-s*box[1],s*box[0]+c*box[1],box[2]};
  oriented.axis_x={std::cos(yaw),std::sin(yaw),0};
  oriented.axis_y={-std::sin(yaw),std::cos(yaw),0};
  oriented.half_length=box[3]*.5f;oriented.half_width=box[4]*.5f;
  oriented.min_z=box[2]-box[5]*.5f;oriented.max_z=box[2]+box[5]*.5f;oriented.valid=true;
  const float ex=std::abs(oriented.axis_x.x)*oriented.half_length+std::abs(oriented.axis_y.x)*oriented.half_width;
  const float ey=std::abs(oriented.axis_x.y)*oriented.half_length+std::abs(oriented.axis_y.y)*oriented.half_width;
  detection.bounds={{oriented.center.x-ex,oriented.center.y-ey,oriented.min_z},
    {oriented.center.x+ex,oriented.center.y+ey,oriented.max_z}};
  return detection;
}
} // namespace lidar_viewer::algorithms
