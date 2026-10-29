#include "lidar_viewer/imu_alignment.hpp"
#include "lidar_viewer/camera_2d.hpp"
#include "lidar_viewer/display_alignment.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
using namespace lidar_viewer;
constexpr double pi = 3.14159265358979323846;
void Check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
bool Near(double a, double b, double epsilon = 1e-6) { return std::abs(a-b) < epsilon; }
ImuQuaternion Inverse(ImuQuaternion q) { return {-q.x, -q.y, -q.z, q.w}; }

void TestLevelingGeometry()
{
  const auto mount = ImuRotationFromRpy(pi, 0, .25);
  for (const double roll : {-.4, .1, .6}) {
    for (const double pitch : {-.3, 0., .4}) {
      const auto body = ImuRotationFromRpy(roll, pitch, 1.2);
      const auto up_lidar = RotateImuVector(Inverse(body), {0, 0, 1});
      FrameImuData data;
      data.imu_frame = "imu_ned"; data.lidar_frame = "lidar";
      data.up = RotateImuVector(Inverse(mount), up_lidar);
      data.imu_to_lidar = mount;
      const auto correction = ComputeImuLeveling(data, {});
      Check(correction.applied && Near(correction.roll_deg*pi/180, roll) &&
        Near(correction.pitch_deg*pi/180, pitch), "Mounting rotation or leveling direction is incorrect");
      const auto level = ImuRotationFromRpy(correction.roll_deg*pi/180, correction.pitch_deg*pi/180, .7);
      const auto up = RotateImuVector(level, up_lidar);
      Check(Near(up.x, 0) && Near(up.y, 0) && Near(up.z, 1), "Gravity did not align to +Z");
      for (const double x : {-3., 2., 5.}) {
        const auto raw = RotateImuVector(Inverse(body), {x, 4, -.57});
        const auto aligned = RotateImuVector(level, raw);
        Check(Near(aligned.z + .57, 0), "A tilted flat floor did not become horizontal");
        const auto sloped = RotateImuVector(level, RotateImuVector(Inverse(body), {x, 4, .2*x-.57}));
        Check(Near(sloped.z+.57, .2*x), "IMU leveling flattened real terrain slope");
      }
      data.imu_to_lidar.reset();
      ImuMountSettings profile;
      profile.confirmed = true; profile.imu_frame = "imu_ned"; profile.lidar_frame = "lidar";
      profile.roll_deg = 180; profile.yaw_deg = static_cast<float>(.25*180/pi);
      Check(ComputeImuLeveling(data, profile).applied, "Configured matching mount was rejected");
      profile.lidar_frame = "another_lidar";
      Check(!ComputeImuLeveling(data, profile).applied, "A mount for another frame was silently applied");
      profile.confirmed = false;
      Check(!ComputeImuLeveling(data, profile).applied, "Unconfirmed identity mount was silently assumed");
    }
  }
}

void TestStaticTransforms()
{
  const auto a = ImuRotationFromRpy(.3, -.1, .4), b = ImuRotationFromRpy(-.2, .4, -.6);
  const std::vector<ImuStaticRotation> tree{{"base", "imu", a}, {"base", "lidar", b},
    {"unrelated", "camera", {}}};
  const auto resolved = FindImuStaticRotation(tree, "imu", "lidar");
  Check(resolved.has_value(), "Connected static mount was not found");
  const auto expected = RotateImuVector(Inverse(b), RotateImuVector(a, {1, 2, 3}));
  const auto actual = RotateImuVector(*resolved, {1, 2, 3});
  Check(Near(actual.x, expected.x) && Near(actual.y, expected.y) && Near(actual.z, expected.z),
    "Static TF composition/inversion is incorrect");
  Check(!FindImuStaticRotation(tree, "imu", "camera"), "Disconnected TF frames were joined");
  Check(FindImuStaticRotation({}, "same", "same").has_value(), "Same-frame identity was rejected");
  auto conflicting = tree;
  conflicting.push_back({"another_parent", "imu", {}});
  Check(!FindImuStaticRotation(conflicting, "imu", "lidar"), "Ambiguous static mount was silently chosen");
}

void TestEstimationAndTime()
{
  ImuObservation sample;
  sample.timestamp_ns = 1000000000;
  sample.orientation_valid = true;
  sample.world_ned = true;
  sample.orientation = {};
  auto samples = EstimateImuGravity({sample});
  Check(samples.size() == 1 && Near(samples.front().up.z, -1), "NED world-up convention was inverted");
  sample.orientation_valid = false;
  sample.acceleration_valid = sample.gyro_valid = true;
  sample.acceleration = {0, 0, -9.80665};
  samples = EstimateImuGravity({sample});
  Check(samples.size() == 1 && samples.front().source == ImuGravitySource::AccelGyro &&
    Near(samples.front().up.z, -1), "Raw acceleration fallback failed");
  sample.acceleration = {20, 0, 9.8};
  Check(EstimateImuGravity({sample}).empty(), "Dynamic acceleration initialized a false gravity direction");
  sample.acceleration = {0, 0, 9.80665};
  sample.angular_velocity = {.2, 0, 0};
  std::vector<ImuObservation> observations{sample};
  for (int i = 1; i <= 20; ++i) {
    sample.timestamp_ns += 10000000;
    sample.acceleration_valid = false;
    observations.push_back(sample);
  }
  samples = EstimateImuGravity(observations);
  Check(samples.size() == 21 && Near(samples.back().up.y, std::sin(.04)) &&
    Near(samples.back().up.z, std::cos(.04)), "Gyro propagation used the wrong sign or interval");
  sample.timestamp_ns += 1000000000;
  observations.push_back(sample);
  Check(EstimateImuGravity(observations).size() == 21, "A stale attitude survived an IMU gap");
  sample.orientation_valid = true;
  sample.orientation = {0, 0, 0, 0};
  Check(EstimateImuGravity({sample}).empty(), "Zero quaternion was accepted");
  sample.orientation.x = std::numeric_limits<double>::quiet_NaN();
  Check(EstimateImuGravity({sample}).empty(), "NaN quaternion was accepted");

  ImuTimeline timeline;
  timeline.topic = "/imu"; timeline.frame_id = "lidar";
  timeline.samples = {{1000000000, {0, 0, 1}, ImuGravitySource::Orientation},
    {1040000000, {0, .1, std::sqrt(.99)}, ImuGravitySource::Orientation}};
  auto frame = MatchFrameImu(timeline, 1020000000, "lidar");
  Check(frame.up && frame.up->y > 0 && frame.up->y < .1 && Near(frame.skew_ms, 20),
    "Frame/IMU interpolation or timestamp matching is incorrect");
  Check(!MatchFrameImu(timeline, 1300000000, "lidar").up, "Stale IMU was paired with a frame");
  timeline.received_timestamps = {1300000000};
  const auto unreliable = MatchFrameImu(timeline, 1300000000, "lidar");
  Check(!unreliable.up && unreliable.status.find("not reliable") != std::string::npos,
    "Unreliable gravity was reported as a missing IMU recording");
  Check(!ComputeImuLeveling(std::nullopt, {}).applied, "Missing IMU was treated as level");
}

void TestTopView()
{
  const auto up = TopViewUp(kPi/2);
  Check(Near(up.x, -1) && Near(up.y, 0), "2D rotation did not rotate the camera basis");
  const auto panned = PanTopView({0, 0, 2}, kPi/2, 10, 20, .1f);
  Check(Near(panned.x, -2) && Near(panned.y, -1) && Near(panned.z, 2),
    "Panning a rotated 2D view did not follow screen coordinates");
  Check(Near(RotateTopView(kPi-.1f, .2f), -kPi+.1f), "2D rotation failed to wrap");
}

void TestDisplayAlignmentFallback()
{
  const DisplayAlignment manual{12,-7,5,.57f};
  const auto missing=ComputeImuLeveling(std::nullopt,{});
  auto resolved=ResolveDisplayAlignment(manual,true,missing);
  Check(!resolved.imu_active && resolved.effective.roll_deg==5 && resolved.effective.pitch_deg==-7,
    "Requested auto leveling locked out the manual fallback without usable IMU");
  const ImuLevelingResult valid{true,2,-3,"fixture"};
  resolved=ResolveDisplayAlignment(manual,true,valid);
  Check(resolved.imu_active && resolved.effective.roll_deg==2 && resolved.effective.pitch_deg==-3 &&
    resolved.effective.yaw_deg==12 && resolved.effective.z_m==manual.z_m,
    "Auto leveling changed yaw/height or displayed stale manual angles");
  resolved=ResolveDisplayAlignment(manual,false,valid);
  Check(!resolved.imu_active && resolved.effective.roll_deg==manual.roll_deg,
    "Disabling automatic leveling lost the saved manual angles");
  auto reset=manual;bool enabled=true;
  ResetDisplayRotation(reset,enabled);
  Check(!enabled && reset.yaw_deg==0 && reset.pitch_deg==0 && reset.roll_deg==0 && reset.z_m==manual.z_m,
    "Rotation reset left IMU enabled or discarded sensor height");
  FrameImuData raw_ned;
  raw_ned.imu_frame="vn100_link_ned";raw_ned.lidar_frame="pandar_xt32";raw_ned.up=ImuVector{0,0,-1};
  ImuMountSettings incorrect{true,raw_ned.imu_frame,raw_ned.lidar_frame,.7f,0,0};
  Check(std::abs(ComputeImuLeveling(raw_ned,incorrect).roll_deg)>170,
    "Raw NED mounting regression fixture no longer reproduces a flip");
  incorrect.roll_deg=180;
  const auto converted=ComputeImuLeveling(raw_ned,incorrect);
  Check(converted.applied && std::abs(converted.roll_deg)<1e-4f && std::abs(converted.pitch_deg)<1e-4f,
    "NED-to-FLU axis conversion left an upright sensor upside down");
  raw_ned.up=ImuVector{0,0,1};  // A prepared FLU vector must not be flipped a second time.
  incorrect.roll_deg=0;
  const auto prepared=ComputeImuLeveling(raw_ned,incorrect);
  Check(prepared.applied && std::abs(prepared.roll_deg)<1e-4f,
    "Prepared FLU vectors did not accept the identity mounting preset");
}
}  // namespace

int main()
{
  try {
    TestLevelingGeometry(); TestStaticTransforms(); TestEstimationAndTime(); TestTopView(); TestDisplayAlignmentFallback();
    std::cout << "IMU leveling and 2D camera checks passed\n";
    return 0;
  } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
