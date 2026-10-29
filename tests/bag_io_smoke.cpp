#include "lidar_viewer/bag_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef LIDAR_VIEWER_HAS_ROSBAG
#include <rclcpp/time.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/imu.hpp>
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
#include <tf2_msgs/msg/tf_message.hpp>
#endif
#endif

namespace
{
void Check(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<typename Function>
void Throws(Function function, const std::string & message)
{
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

#ifdef LIDAR_VIEWER_HAS_ROSBAG
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;

Field MakeField(const char * name, std::uint32_t offset, std::uint8_t type)
{
  Field field;
  field.name = name;
  field.offset = offset;
  field.datatype = type;
  field.count = 1;
  return field;
}

template<typename T>
void Store(std::vector<std::uint8_t> & data, std::size_t offset, T value, bool big_endian = false)
{
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  const std::uint16_t probe = 1;
  std::uint8_t first;
  std::memcpy(&first, &probe, 1);
  if (big_endian != (first == 0)) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::copy(bytes.begin(), bytes.end(), data.begin() + static_cast<std::ptrdiff_t>(offset));
}

Cloud Xt32Cloud(std::uint32_t width = 2, std::uint32_t height = 1, std::uint32_t padding = 0)
{
  Cloud cloud;
  cloud.header.frame_id = "pandar_xt32";
  cloud.width = width;
  cloud.height = height;
  cloud.point_step = 26;
  cloud.row_step = width * cloud.point_step + padding;
  cloud.fields = {
    MakeField("x", 0, Field::FLOAT32), MakeField("y", 4, Field::FLOAT32),
    MakeField("z", 8, Field::FLOAT32), MakeField("intensity", 12, Field::FLOAT32),
    MakeField("ring", 16, Field::UINT16), MakeField("timestamp", 18, Field::FLOAT64)};
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step) * height, 0x5A);
  for (std::uint32_t row = 0; row < height; ++row) {
    for (std::uint32_t column = 0; column < width; ++column) {
      const std::size_t base = static_cast<std::size_t>(row) * cloud.row_step + column * 26;
      const float number = static_cast<float>(row * width + column + 1);
      Store(cloud.data, base, number);
      Store(cloud.data, base + 4, number * 2);
      Store(cloud.data, base + 8, -number);
      Store(cloud.data, base + 12, number * 10);
      Store(cloud.data, base + 16, static_cast<std::uint16_t>(column));
      Store(cloud.data, base + 18, 1792730000.0 + number * .001);
    }
  }
  return cloud;
}

void TestPointLayouts()
{
  const auto cloud = Xt32Cloud(2, 2, 7);
  const auto frame = DecodePointCloud2(cloud);
  Check(frame.points.size() == 4, "Organized rows/padding or unaligned 26-byte stride lost points");
  Check(frame.points[3].x == 4 && frame.points[3].y == 8 && frame.points[3].z == -4 &&
    frame.points[3].intensity == 40, "XYZI values changed during decoding");

  auto big = Xt32Cloud(1);
  big.is_bigendian = true;
  Store(big.data, 0, 1.25f, true);
  Store(big.data, 4, -2.5f, true);
  Store(big.data, 8, 3.75f, true);
  Store(big.data, 12, 44.0f, true);
  auto decoded = DecodePointCloud2(big);
  Check(decoded.points.size() == 1 && decoded.points[0].x == 1.25f &&
    decoded.points[0].y == -2.5f && decoded.points[0].intensity == 44, "Big-endian decoding failed");

  auto nonfinite = Xt32Cloud(3);
  Store(nonfinite.data, 0, std::numeric_limits<float>::quiet_NaN());
  Store(nonfinite.data, 26 + 12, std::numeric_limits<float>::infinity());
  decoded = DecodePointCloud2(nonfinite);
  Check(decoded.points.size() == 2 && decoded.points[0].x == 2 &&
    decoded.points[0].intensity == 0, "Nonfinite XYZ filtering or intensity fallback failed");

  auto no_intensity = Xt32Cloud(1);
  no_intensity.fields.erase(no_intensity.fields.begin() + 3);
  Check(DecodePointCloud2(no_intensity).points[0].intensity == 0, "Optional intensity is not optional");

  auto integer = Xt32Cloud(1);
  integer.fields[3] = MakeField("reflectivity", 12, Field::UINT16);
  Store(integer.data, 12, static_cast<std::uint16_t>(512));
  Check(DecodePointCloud2(integer).points[0].intensity == 512, "Integer reflectivity decoding failed");

  auto doubles = Xt32Cloud(1);
  doubles.point_step = 40;
  doubles.row_step = 40;
  doubles.data.resize(40);
  doubles.fields = {MakeField("x", 1, Field::FLOAT64), MakeField("y", 9, Field::FLOAT64),
    MakeField("z", 17, Field::FLOAT64), MakeField("intensity", 25, Field::FLOAT64)};
  Store(doubles.data, 1, 1.125);
  Store(doubles.data, 9, -2.25);
  Store(doubles.data, 17, 3.5);
  Store(doubles.data, 25, 5.125);
  Check(DecodePointCloud2(doubles).points[0].intensity == 5.125f, "Unaligned float64 scalar decoding failed");

  Cloud empty;
  Check(DecodePointCloud2(empty).points.empty(), "Empty cloud should be allowed");
}

void TestMalformedLayouts()
{
  auto cloud = Xt32Cloud();
  cloud.data.pop_back();
  Throws([&] {DecodePointCloud2(cloud);}, "Truncated data accepted");
  cloud = Xt32Cloud(); cloud.row_step = 1;
  Throws([&] {DecodePointCloud2(cloud);}, "Undersized row accepted");
  cloud = Xt32Cloud(); cloud.point_step = 0;
  Throws([&] {DecodePointCloud2(cloud);}, "Zero point stride accepted");
  cloud = Xt32Cloud(); cloud.fields[0].offset = 25;
  Throws([&] {DecodePointCloud2(cloud);}, "Out-of-point field accepted");
  cloud = Xt32Cloud(); cloud.fields.erase(cloud.fields.begin());
  Throws([&] {DecodePointCloud2(cloud);}, "Missing XYZ accepted");
  cloud = Xt32Cloud(); cloud.fields.push_back(cloud.fields.front());
  Throws([&] {DecodePointCloud2(cloud);}, "Duplicate X accepted");
  cloud = Xt32Cloud(); cloud.fields.front().datatype = 99;
  Throws([&] {DecodePointCloud2(cloud);}, "Unknown datatype accepted");
  cloud = Xt32Cloud(); cloud.fields.front().count = 3;
  Throws([&] {DecodePointCloud2(cloud);}, "Nonscalar X accepted");
}

void TestStorage(const std::string & storage_id)
{
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path folder = fs::temp_directory_path() /
    ("lidar-bag-io-test-" + storage_id + "-" + std::to_string(nonce));
  try {
    {
      rosbag2_cpp::Writer writer;
      rosbag2_storage::StorageOptions options;
      options.uri = folder.string();
      options.storage_id = storage_id;
      writer.open(options);
      auto first = Xt32Cloud(1);
      auto second = first;
      Store(second.data, 0, 9.0f);
      writer.write(first, "/b2/points", rclcpp::Time(1000000));
      writer.write(second, "/b2/points", rclcpp::Time(1000000));
      writer.write(second, "/another/cloud", rclcpp::Time(2000000));
      writer.write(second, "/b2/points", rclcpp::Time(3000000));
      sensor_msgs::msg::Imu imu;
      imu.header.frame_id = "fixture_imu";
      const auto q = lidar_viewer::ImuRotationFromRpy(.1, -.2, .7);
      imu.orientation.x = q.x; imu.orientation.y = q.y; imu.orientation.z = q.z; imu.orientation.w = q.w;
      imu.linear_acceleration_covariance[0] = -1;
      imu.angular_velocity_covariance[0] = -1;
      writer.write(imu, "/fixture/imu", rclcpp::Time(0));
      writer.write(imu, "/fixture/imu", rclcpp::Time(2000000));
      writer.write(imu, "/fixture/imu", rclcpp::Time(4000000));
      imu.header.frame_id = "vn100_link_ned";
      imu.orientation.x = imu.orientation.y = imu.orientation.z = 0; imu.orientation.w = 1;
      imu.linear_acceleration.z = -9.80665;
      imu.linear_acceleration_covariance.fill(0);
      imu.angular_velocity_covariance.fill(0);
      writer.write(imu, "/vn100/vectornav_driver_node/imu/data", rclcpp::Time(0));
      writer.write(imu, "/vn100/vectornav_driver_node/imu/data", rclcpp::Time(4000000));
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
      tf2_msgs::msg::TFMessage transforms;
      geometry_msgs::msg::TransformStamped transform;
      transform.header.frame_id = "fixture_base"; transform.child_frame_id = "fixture_imu";
      transform.transform.rotation.w = 1;
      transforms.transforms.push_back(transform);
      transform.child_frame_id = "pandar_xt32";
      transforms.transforms.push_back(transform);
      writer.write(transforms, "/tf_static", rclcpp::Time(500000));
#endif
      writer.close();
    }
    Check(IsRos2BagInput(folder) && IsRos2BagInput(folder / "metadata.yaml"), "Bag detection failed");
    lidar_viewer::InputLoadState load;
    const auto handles = ResolveBagFrames(folder,"/b2/points","",&load);
    const auto progress=load.Snapshot();
    Check(progress.frames==3 && progress.total_frames==3 && progress.imu_samples==3 &&
      progress.messages==progress.total_messages && progress.messages>=6 &&
      progress.stage==lidar_viewer::InputLoadStage::PreparingImu,
      "Loading progress did not match the filtered bag metadata and actual reads");
    load.Cancel();
    bool canceled=false;
    try {ResolveBagFrames(folder,"/b2/points","",&load);}
    catch (const lidar_viewer::InputLoadCancelled &) {canceled=true;}
    Check(canceled,"Canceled indexing continued or was reported as an input error");
    Check(handles.size() == 3, "Topic filtering lost or included other messages");
    const auto matched = LoadBagFrame(handles[0]);
    Check(matched.imu && matched.imu->up && matched.imu->topic == "/fixture/imu" &&
      matched.imu->source == lidar_viewer::ImuGravitySource::Orientation, "Recorded IMU was not paired with LiDAR");
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
    Check(matched.imu->imu_to_lidar.has_value(), "Static TF mount rotation was not read from the bag");
    Check(lidar_viewer::ComputeImuLeveling(matched.imu, {}).applied, "Valid recorded orientation/TF could not level LiDAR");
#endif
    Throws([&] { ResolveBagFrames(folder, "/b2/points", "/missing_imu"); }, "Explicit missing IMU topic was accepted");
    const auto raw_handles = ResolveBagFrames(folder, "/b2/points", "/vn100/vectornav_driver_node/imu/data");
    const auto raw = LoadBagFrame(raw_handles.front());
    Check(raw.imu && raw.imu->up && raw.imu->source == lidar_viewer::ImuGravitySource::AccelGyro &&
      raw.imu->up->z < -.99, "VN100 placeholder quaternion was mistaken for a fused attitude");
    Check(handles[0].bag_duplicate_index == 0 && handles[1].bag_duplicate_index == 1,
      "Duplicate receipt timestamps lost exact frame identity");
    Check(LoadBagFrame(handles[2]).points[0].x == 9 && LoadBagFrame(handles[0]).points[0].x == 1 &&
      LoadBagFrame(handles[1]).points[0].x == 9, "Backward seeking/duplicate frame access failed");
    const auto read_sequence = [&handles](bool reverse) {
        for (std::size_t iteration = 0; iteration < 48; ++iteration) {
          const auto slot = iteration % handles.size();
          const auto index = reverse ? handles.size() - 1 - slot : slot;
          const auto frame = LoadBagFrame(handles[index]);
          Check(frame.points.size() == 1 && frame.points[0].x == (index == 0 ? 1 : 9),
            "Concurrent reader seeks mixed point-cloud frames");
        }
      };
    auto forward = std::async(std::launch::async, read_sequence, false);
    auto backward = std::async(std::launch::async, read_sequence, true);
    forward.get();
    backward.get();
    Check(ResolveBagFrames(folder / "metadata.yaml", "/another/cloud").size() == 1,
      "Explicit topic/metadata.yaml selection failed");
    Check(ResolveBagFrames(folder, "").size() == 3, "Automatic preferred topic selection failed");
    Throws([&] {ResolveBagFrames(folder, "/missing");}, "Missing topic should fail clearly");
    for (const auto & entry : fs::directory_iterator(folder)) {
      if (entry.path().extension() == ".mcap" || entry.path().extension() == ".db3") {
        Check(ResolveBagFrames(entry.path()).size() == 3, "Direct storage-file input failed");
      }
    }
  } catch (...) {
    fs::remove_all(folder);
    throw;
  }
  fs::remove_all(folder);
}
#endif
}  // namespace

int main(int argc, char ** argv)
{
  try {
#ifdef LIDAR_VIEWER_HAS_ROSBAG
    Check(RosbagInputAvailable(), "Bag-enabled build reports unavailable");
    TestPointLayouts();
    TestMalformedLayouts();
    TestStorage("mcap");
    TestStorage("sqlite3");
    if (argc == 2) {
      const auto frames = ResolveBagFrames(argv[1]);
      const auto first = LoadBagFrame(frames.front());
      const auto last = LoadBagFrame(frames.back());
      Check(!first.points.empty() && !last.points.empty(), "Real bag first/last frame empty");
      std::cout << "Real bag: " << frames.size() << " frames, first/last points "
                << first.points.size() << '/' << last.points.size() << '\n';
      const auto middle = LoadBagFrame(frames[frames.size()/2]);
      if (middle.imu) {
        std::size_t matched = 0;
        for (const auto & handle : frames) {
          if (handle.imu_timeline && lidar_viewer::MatchFrameImu(*handle.imu_timeline,
            handle.bag_timestamp_ns, first.frame_id).up) ++matched;
        }
        std::cout << "IMU coverage within 50 ms: " << matched << '/' << frames.size() << " frames\n";
        lidar_viewer::ImuMountSettings mount;
        mount.confirmed = true; mount.imu_frame = "vn100_link_ned"; mount.lidar_frame = "pandar_xt32"; mount.roll_deg = 180;
        const auto level = lidar_viewer::ComputeImuLeveling(middle.imu, mount);
        std::cout << "Real IMU: " << middle.imu->topic << " | " << level.status
                  << " | roll " << level.roll_deg << " pitch " << level.pitch_deg << '\n';
        if (middle.imu->up && middle.imu->imu_frame == mount.imu_frame && middle.imu->lidar_frame == mount.lidar_frame)
          Check(level.applied, "Recorded VN100 sample failed to level the configured sensor mount");
      }
    }
#else
    (void)argc;
    (void)argv;
    Check(!RosbagInputAvailable(), "No-ROS build reports available");
    Throws([] {ResolveBagFrames("missing");}, "Disabled backend did not explain unsupported bag input");
#endif
    std::cout << "bag_io_smoke: all checks passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "bag_io_smoke: " << error.what() << '\n';
    return 1;
  }
}
