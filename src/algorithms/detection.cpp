#include "lidar_viewer/algorithms/detection.hpp"

#include <algorithm>
#include <cmath>

namespace lidar_viewer::algorithms
{
namespace
{
float Clamp01(float value)
{
  return std::clamp(value, 0.0f, 1.0f);
}

float ScoreRange(float value, float min_value, float ideal_value, float max_value)
{
  if (value < min_value || value > max_value) {
    return 0.0f;
  }
  if (value <= ideal_value) {
    return Clamp01((value - min_value) / std::max(ideal_value - min_value, 1e-4f));
  }
  return Clamp01((max_value - value) / std::max(max_value - ideal_value, 1e-4f));
}

Detection MakeDetection(
  const Cluster & cluster,
  float confidence,
  int class_id)
{
  Detection detection;
  detection.bounds = cluster.bounds;
  detection.oriented_bounds = cluster.oriented_bounds;
  detection.confidence = confidence;
  detection.class_id = class_id;
  return detection;
}
}  // namespace

std::vector<Detection> RunDetectionPlaceholder(
  const std::vector<KittiPoint> &,
  const std::vector<Cluster> & clusters,
  const DetectionParams & params)
{
  if (!params.enabled) {
    return {};
  }

  std::vector<Detection> detections;
  detections.reserve(clusters.size());

  for (const auto & cluster : clusters) {
    const int point_count = static_cast<int>(cluster.indices.size());
    if (point_count < params.min_points) {
      continue;
    }

    const float length = cluster.oriented_bounds.valid
      ? (cluster.oriented_bounds.half_length * 2.0f)
      : (cluster.bounds.max.x - cluster.bounds.min.x);
    const float width = cluster.oriented_bounds.valid
      ? (cluster.oriented_bounds.half_width * 2.0f)
      : (cluster.bounds.max.y - cluster.bounds.min.y);
    const float height = cluster.bounds.max.z - cluster.bounds.min.z;
    const float footprint = length * width;
    const float aspect = length / std::max(width, 1e-4f);

    const float car_score =
      0.35f * ScoreRange(length, 1.8f, 4.5f, 7.5f) +
      0.25f * ScoreRange(width, 1.2f, 2.0f, 3.2f) +
      0.20f * ScoreRange(height, 1.0f, 1.8f, 3.2f) +
      0.20f * Clamp01(static_cast<float>(point_count) / 400.0f);

    const float bus_score =
      0.38f * ScoreRange(length, 5.5f, 10.5f, 16.0f) +
      0.24f * ScoreRange(width, 2.0f, 2.7f, 3.8f) +
      0.20f * ScoreRange(height, 2.0f, 3.0f, 4.5f) +
      0.18f * Clamp01(static_cast<float>(point_count) / 700.0f);

    const float truck_score =
      0.36f * ScoreRange(length, 4.0f, 7.5f, 14.0f) +
      0.24f * ScoreRange(width, 1.8f, 2.6f, 3.8f) +
      0.18f * ScoreRange(height, 1.8f, 2.8f, 4.2f) +
      0.22f * Clamp01(static_cast<float>(point_count) / 620.0f);

    const float person_score =
      0.38f * ScoreRange(height, 1.0f, 1.7f, 2.3f) +
      0.24f * ScoreRange(width, 0.15f, 0.55f, 1.1f) +
      0.18f * ScoreRange(length, 0.15f, 0.55f, 1.1f) +
      0.20f * Clamp01(static_cast<float>(point_count) / 48.0f);

    const float cyclist_score =
      0.30f * ScoreRange(height, 1.0f, 1.6f, 2.2f) +
      0.18f * ScoreRange(width, 0.25f, 0.7f, 1.4f) +
      0.24f * ScoreRange(length, 0.7f, 1.6f, 3.2f) +
      0.10f * ScoreRange(aspect, 1.0f, 2.1f, 5.5f) +
      0.18f * Clamp01(static_cast<float>(point_count) / 70.0f);

    const float tree_score =
      0.40f * ScoreRange(height, 1.5f, 4.5f, 12.0f) +
      0.20f * ScoreRange(width, 0.3f, 1.2f, 3.0f) +
      0.20f * ScoreRange(aspect, 0.4f, 1.0f, 2.2f) +
      0.20f * Clamp01(static_cast<float>(point_count) / 260.0f);

    const float unknown_score =
      0.40f * Clamp01(footprint / 8.0f) +
      0.30f * Clamp01(height / 3.0f) +
      0.30f * Clamp01(static_cast<float>(point_count) / 250.0f);

    float best_score = car_score;
    int best_class = DetectionClassCar;
    if (bus_score > best_score) {
      best_score = bus_score;
      best_class = DetectionClassBus;
    }
    if (truck_score > best_score) {
      best_score = truck_score;
      best_class = DetectionClassTruck;
    }
    if (person_score > best_score) {
      best_score = person_score;
      best_class = DetectionClassPerson;
    }
    if (cyclist_score > best_score) {
      best_score = cyclist_score;
      best_class = DetectionClassCyclist;
    }
    if (tree_score > best_score) {
      best_score = tree_score;
      best_class = DetectionClassTree;
    }
    if (unknown_score > best_score) {
      best_score = unknown_score;
      best_class = DetectionClassUnknown;
    }

    if (best_score >= params.min_confidence) {
      detections.push_back(MakeDetection(cluster, best_score, best_class));
    }
  }

  return detections;
}
}  // namespace lidar_viewer::algorithms
