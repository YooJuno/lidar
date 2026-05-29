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
  float min_confidence{0.22f};
  int min_points{10};
};

class DetectionAlgorithm
{
public:
  virtual ~DetectionAlgorithm() = default;
  virtual std::vector<Detection> Run(
    const std::vector<KittiPoint> & input,
    const std::vector<Cluster> & clusters,
    const DetectionParams & params) const = 0;
};

std::vector<Detection> RunDetectionPlaceholder(
  const std::vector<KittiPoint> & input,
  const std::vector<Cluster> & clusters,
  const DetectionParams & params);
}  // namespace lidar_viewer::algorithms
