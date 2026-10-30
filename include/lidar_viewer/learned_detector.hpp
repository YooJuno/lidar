#pragma once
#include "lidar_viewer/algorithms/detection.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace lidar_viewer
{
struct DetectorResult {
  std::uint64_t revision{0};
  std::vector<algorithms::Detection> detections;
  float latency_ms{0};
  std::string detail, error;
};
struct DetectorStatus {
  bool configured{false}, ready{false}, failed{false};
  std::string message, provider;
};
// One persistent isolated Python/ONNX process and one latest-only request slot.
// Neither model loading nor inference runs on the rendering / ROS threads.
class LearnedDetector {
public:
  LearnedDetector();
  ~LearnedDetector();
  LearnedDetector(const LearnedDetector &)=delete;
  LearnedDetector & operator=(const LearnedDetector &)=delete;
  void Start(const std::filesystem::path & python_root);
  void Stop();
  DetectorStatus Status() const;
  void Submit(std::uint64_t revision, const std::vector<KittiPoint> & full_scan,
    float threshold, float display_yaw_rad);
  // Immutable snapshot shared with the latest-only queue. Packing and yaw
  // conversion happen on the worker, after superseded requests are discarded.
  void Submit(std::uint64_t revision, std::shared_ptr<const std::vector<KittiPoint>> full_scan,
    float threshold, float display_yaw_rad);
  std::optional<DetectorResult> Poll();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace lidar_viewer
