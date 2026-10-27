#pragma once

#include "lidar_viewer/data_io.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace lidar_viewer
{
struct LiveOptions
{
  std::string points_topic{"/b2/points"};
  std::string imu_topic{"/vn100/imu/data_ned"};
};
struct LiveFrame
{
  FrameData frame;
  std::uint64_t sequence{0};
};
struct LiveStats
{
  bool connected{false}, recording{false}, record_busy{false}, record_stopping{false};
  std::uint64_t clouds{0}, imu_samples{0}, decode_errors{0}, replaced_frames{0};
  std::uint64_t recorded_messages{0}, recorded_bytes{0}, record_dropped{0};
  double cloud_hz{0}, cloud_age_ms{-1}, imu_age_ms{-1};
  std::filesystem::path recording_path;
  std::string error, record_error;
};

// Explicitly connected ROS subscriber. The display owns a frozen copy, while
// this receiver retains only one latest frame. Recording never uses ROI output.
class LiveSession
{
public:
  LiveSession();
  ~LiveSession();
  LiveSession(const LiveSession &) = delete;
  LiveSession & operator=(const LiveSession &) = delete;
  static bool Available();
  void Connect(const LiveOptions & options);
  void Disconnect();
  bool Active() const;
  std::unique_ptr<LiveFrame> TakeLatest();
  LiveStats Stats() const;
  void StartRecording(const std::filesystem::path & destination);
  // Asynchronous: drain pending raw messages, then finalize metadata.yaml.
  void StopRecording();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace lidar_viewer
