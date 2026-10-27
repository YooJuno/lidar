#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lidar_viewer
{
struct ImuVector
{
  double x{0}, y{0}, z{0};
};

struct ImuQuaternion
{
  double x{0}, y{0}, z{0}, w{1};
};

ImuVector RotateImuVector(const ImuQuaternion & rotation, const ImuVector & vector);
ImuQuaternion ImuRotationFromRpy(double roll, double pitch, double yaw);
ImuQuaternion MultiplyImuRotation(const ImuQuaternion & a, const ImuQuaternion & b);
std::optional<ImuQuaternion> NormalizeImuRotation(const ImuQuaternion & rotation);

struct ImuObservation
{
  std::int64_t timestamp_ns{0};  // Bag receive clock, shared with frame handles.
  ImuVector acceleration, angular_velocity;
  ImuQuaternion orientation;
  bool acceleration_valid{false}, gyro_valid{false}, orientation_valid{false};
  bool world_ned{false};
};

enum class ImuGravitySource { Orientation, AccelGyro };

struct ImuGravitySample
{
  std::int64_t timestamp_ns{0};
  ImuVector up;
  ImuGravitySource source{ImuGravitySource::AccelGyro};
};

struct ImuStaticRotation
{
  std::string parent, child;
  ImuQuaternion parent_from_child;
};

struct ImuTimeline
{
  std::string topic, frame_id, status;
  std::vector<ImuGravitySample> samples;
  std::vector<std::int64_t> received_timestamps;
  std::vector<ImuStaticRotation> static_rotations;
};

std::vector<ImuGravitySample> EstimateImuGravity(const std::vector<ImuObservation> & observations);
std::optional<ImuQuaternion> FindImuStaticRotation(
  const std::vector<ImuStaticRotation> & transforms,
  const std::string & source, const std::string & target);

struct FrameImuData
{
  std::string topic, imu_frame, lidar_frame, status;
  std::optional<ImuVector> up;
  std::optional<ImuQuaternion> imu_to_lidar;
  ImuGravitySource source{ImuGravitySource::AccelGyro};
  double skew_ms{0};
};

FrameImuData MatchFrameImu(
  const ImuTimeline & timeline, std::int64_t timestamp_ns, const std::string & lidar_frame,
  std::int64_t maximum_skew_ns = 50000000);

struct ImuMountSettings
{
  bool confirmed{false};
  std::string imu_frame, lidar_frame;
  float roll_deg{0}, pitch_deg{0}, yaw_deg{0};
};

struct ImuLevelingResult
{
  bool applied{false};
  float roll_deg{0}, pitch_deg{0};
  std::string status;
};

ImuLevelingResult ComputeImuLeveling(
  const std::optional<FrameImuData> & data, const ImuMountSettings & mount);
}  // namespace lidar_viewer
