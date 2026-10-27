#include "lidar_viewer/imu_alignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace lidar_viewer
{
namespace
{
constexpr double pi = 3.14159265358979323846;
double Norm(const ImuVector & v) { return std::hypot(v.x, v.y, v.z); }
bool Finite(const ImuVector & v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
ImuVector Unit(ImuVector v) { const double n = Norm(v); return {v.x / n, v.y / n, v.z / n}; }
ImuQuaternion Inverse(const ImuQuaternion & q) { return {-q.x, -q.y, -q.z, q.w}; }
}  // namespace

std::optional<ImuQuaternion> NormalizeImuRotation(const ImuQuaternion & q)
{
  const double n = std::hypot(std::hypot(q.x, q.y), std::hypot(q.z, q.w));
  if (!std::isfinite(n) || n < 1e-8) return std::nullopt;
  return ImuQuaternion{q.x / n, q.y / n, q.z / n, q.w / n};
}

ImuQuaternion MultiplyImuRotation(const ImuQuaternion & a, const ImuQuaternion & b)
{
  return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y, a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
    a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w, a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}

ImuQuaternion ImuRotationFromRpy(double roll, double pitch, double yaw)
{
  const double cr=std::cos(roll/2), sr=std::sin(roll/2), cp=std::cos(pitch/2), sp=std::sin(pitch/2);
  const double cy=std::cos(yaw/2), sy=std::sin(yaw/2);
  return {sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy};
}

ImuVector RotateImuVector(const ImuQuaternion & q, const ImuVector & v)
{
  const auto p = MultiplyImuRotation(MultiplyImuRotation(q, {v.x, v.y, v.z, 0}), Inverse(q));
  return {p.x, p.y, p.z};
}

std::vector<ImuGravitySample> EstimateImuGravity(const std::vector<ImuObservation> & observations)
{
  std::vector<ImuGravitySample> result;
  result.reserve(observations.size());
  std::optional<ImuVector> up;
  std::optional<std::int64_t> previous;
  std::int64_t last_reference = 0;
  ImuVector previous_gyro;
  bool previous_gyro_valid = false;
  for (const auto & sample : observations) {
    // Duplicated/reversed timestamps cannot provide an integration interval.
    if (previous && sample.timestamp_ns <= *previous) continue;
    const double dt = previous ? (sample.timestamp_ns - *previous) * 1e-9 : 0;
    if (dt > .1) up.reset();
    const double a_norm = Norm(sample.acceleration), gyro_norm = Norm(sample.angular_velocity);
    const bool gyro_valid = sample.gyro_valid && Finite(sample.angular_velocity) && gyro_norm < 20;
    const bool acceleration_trusted = sample.acceleration_valid && Finite(sample.acceleration) &&
      std::abs(a_norm - 9.80665) < 1.5 && (!gyro_valid || gyro_norm < .35);
    const auto orientation = sample.orientation_valid ? NormalizeImuRotation(sample.orientation) : std::nullopt;
    ImuGravitySource source = ImuGravitySource::AccelGyro;
    if (orientation) {
      up = Unit(RotateImuVector(Inverse(*orientation), {0, 0, sample.world_ned ? -1.0 : 1.0}));
      last_reference = sample.timestamp_ns;
      source = ImuGravitySource::Orientation;
    } else {
      if (up && previous_gyro_valid && dt > 0 && dt <= .1) {
        const double speed = Norm(previous_gyro);
        if (speed > 1e-10) {
          const double angle = -speed * dt / 2;
          const double s = std::sin(angle) / speed;
          up = Unit(RotateImuVector({previous_gyro.x*s, previous_gyro.y*s, previous_gyro.z*s,
            std::cos(angle)}, *up));
        }
      }
      if (acceleration_trusted) {
        const ImuVector measured = Unit(sample.acceleration);
        if (!up) up = measured;
        else {
          const double alpha = dt / (.7 + dt);
          ImuVector blended{(1-alpha)*up->x+alpha*measured.x, (1-alpha)*up->y+alpha*measured.y,
            (1-alpha)*up->z+alpha*measured.z};
          if (Norm(blended) > 1e-8) up = Unit(blended);
        }
        last_reference = sample.timestamp_ns;
      }
      if (!previous_gyro_valid && !acceleration_trusted) up.reset();
      if (up && sample.timestamp_ns - last_reference > 500000000) up.reset();
    }
    if (up) result.push_back({sample.timestamp_ns, *up, source});
    previous = sample.timestamp_ns;
    previous_gyro = sample.angular_velocity;
    previous_gyro_valid = gyro_valid;
  }
  return result;
}

std::optional<ImuQuaternion> FindImuStaticRotation(
  const std::vector<ImuStaticRotation> & transforms, const std::string & source, const std::string & target)
{
  if (source.empty() || target.empty()) return std::nullopt;
  std::map<std::string, std::pair<std::string, ImuQuaternion>> by_child;
  std::set<std::string> ambiguous;
  for (const auto & tf : transforms) {
    const auto q = NormalizeImuRotation(tf.parent_from_child);
    if (!q) { ambiguous.insert(tf.child); continue; }
    const auto inserted = by_child.emplace(tf.child, std::make_pair(tf.parent, *q));
    if (!inserted.second) {
      const auto & previous = inserted.first->second;
      const double dot = previous.second.x*q->x + previous.second.y*q->y +
        previous.second.z*q->z + previous.second.w*q->w;
      if (previous.first != tf.parent || std::abs(dot) < 1-1e-6) ambiguous.insert(tf.child);
    }
  }
  std::vector<std::pair<std::string, ImuQuaternion>> pending{{source, {}}};
  std::set<std::string> seen{source};
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto current = pending[i];
    if (current.first == target) return current.second;
    for (const auto & tf : transforms) {
      if (ambiguous.count(tf.child)) continue;
      const auto q = NormalizeImuRotation(tf.parent_from_child);
      if (!q) continue;
      std::string next;
      ImuQuaternion rotation;
      if (tf.child == current.first) { next = tf.parent; rotation = *q; }
      else if (tf.parent == current.first) { next = tf.child; rotation = Inverse(*q); }
      else continue;
      if (seen.insert(next).second) pending.emplace_back(next, MultiplyImuRotation(rotation, current.second));
    }
  }
  return std::nullopt;
}

FrameImuData MatchFrameImu(const ImuTimeline & timeline, std::int64_t timestamp,
  const std::string & lidar_frame, std::int64_t maximum_skew_ns)
{
  FrameImuData result;
  result.topic = timeline.topic; result.imu_frame = timeline.frame_id; result.lidar_frame = lidar_frame;
  result.imu_to_lidar = FindImuStaticRotation(timeline.static_rotations, timeline.frame_id, lidar_frame);
  if (timeline.samples.empty()) {
    result.status = timeline.status.empty() ? "No usable IMU samples" : timeline.status;
    return result;
  }
  const auto & samples = timeline.samples;
  auto right = std::lower_bound(samples.begin(), samples.end(), timestamp,
    [](const auto & sample, std::int64_t value) { return sample.timestamp_ns < value; });
  auto best = right == samples.end() ? std::prev(right) : right;
  if (right != samples.begin() && std::abs(std::prev(right)->timestamp_ns - timestamp) <
    std::abs(best->timestamp_ns - timestamp)) best = std::prev(right);
  result.skew_ms = std::abs(best->timestamp_ns - timestamp) * 1e-6;
  if (std::abs(best->timestamp_ns - timestamp) > maximum_skew_ns) {
    result.status = "No IMU within 50 ms of this frame";
    const auto & received = timeline.received_timestamps;
    const auto raw_right = std::lower_bound(received.begin(), received.end(), timestamp);
    const bool raw_nearby = (raw_right != received.end() && std::abs(*raw_right-timestamp) <= maximum_skew_ns) ||
      (raw_right != received.begin() && std::abs(*std::prev(raw_right)-timestamp) <= maximum_skew_ns);
    if (raw_nearby) result.status = "IMU gravity estimate is not reliable at this frame";
    return result;
  }
  result.up = best->up;
  result.source = best->source;
  if (right != samples.begin() && right != samples.end()) {
    const auto left = std::prev(right);
    if (timestamp-left->timestamp_ns <= maximum_skew_ns && right->timestamp_ns-timestamp <= maximum_skew_ns &&
      left->source == right->source) {
      const double alpha = double(timestamp-left->timestamp_ns) / double(right->timestamp_ns-left->timestamp_ns);
      ImuVector blended{(1-alpha)*left->up.x+alpha*right->up.x, (1-alpha)*left->up.y+alpha*right->up.y,
        (1-alpha)*left->up.z+alpha*right->up.z};
      if (Norm(blended) > 1e-8) result.up = Unit(blended);
    }
  }
  result.status = result.source == ImuGravitySource::Orientation ? "IMU orientation" : "Accel/gyro gravity estimate";
  return result;
}

ImuLevelingResult ComputeImuLeveling(const std::optional<FrameImuData> & data, const ImuMountSettings & mount)
{
  ImuLevelingResult result;
  if (!data) { result.status = "No IMU recorded for this input"; return result; }
  if (!data->up) { result.status = data->status; return result; }
  auto rotation = data->imu_to_lidar;
  if (!rotation && mount.confirmed && mount.imu_frame == data->imu_frame && mount.lidar_frame == data->lidar_frame) {
    rotation = NormalizeImuRotation(ImuRotationFromRpy(mount.roll_deg*pi/180, mount.pitch_deg*pi/180, mount.yaw_deg*pi/180));
  }
  if (!rotation) { result.status = "IMU to LiDAR mount rotation is missing"; return result; }
  const auto up = RotateImuVector(*rotation, *data->up);
  if (!Finite(up) || Norm(up) < 1e-8) { result.status = "Invalid IMU gravity direction"; return result; }
  result.roll_deg = static_cast<float>(std::atan2(up.y, up.z) * 180 / pi);
  result.pitch_deg = static_cast<float>(std::atan2(-up.x, std::hypot(up.y, up.z)) * 180 / pi);
  result.applied = true;
  result.status = data->status + (data->imu_to_lidar ? " (static TF mount)" : " (profile mount)");
  return result;
}
}  // namespace lidar_viewer
