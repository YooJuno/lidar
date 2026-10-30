#pragma once

#include "lidar_viewer/algorithms/clustering.hpp"
#include <cstdint>

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
  static constexpr std::uint32_t AllClasses = (1u << 5) - 1;
  std::uint32_t classes{AllClasses};
  bool Accepts(const Detection & detection) const {
    return detection.class_id>=0 && detection.class_id<5 &&
      (classes & (1u << detection.class_id)) && detection.confidence>=min_confidence;
  }
};

const char * DetectionClassName(int class_id);
// Worker format: x,y,z,length,width,height,ROS yaw,score,model class.
Detection DecodeCenterPointBox(const float * box, float display_yaw_rad);
}  // namespace lidar_viewer::algorithms
