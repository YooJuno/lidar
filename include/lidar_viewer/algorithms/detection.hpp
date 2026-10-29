#pragma once

#include "lidar_viewer/algorithms/clustering.hpp"

namespace lidar_viewer::algorithms
{
enum DetectionClassId
{
  DetectionClassCar = 0,
  DetectionClassBus = 1,
  DetectionClassTruck = 2,
  DetectionClassPerson = 3,
  DetectionClassCyclist = 4,
  DetectionClassTree = 5,
  DetectionClassUnknown = 6,
};

struct DetectionParams
{
  bool enabled{false};
  float min_confidence{0.35f};
  int min_points{10};
};

const char * DetectionClassName(int class_id);
// Worker format: x,y,z,length,width,height,ROS yaw,score,model class.
Detection DecodeCenterPointBox(const float * box, float display_yaw_rad);
}  // namespace lidar_viewer::algorithms
