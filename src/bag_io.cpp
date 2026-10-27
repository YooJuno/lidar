#include "lidar_viewer/bag_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef LIDAR_VIEWER_HAS_ROSBAG
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/msg/imu.hpp>
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
#include <tf2_msgs/msg/tf_message.hpp>
#endif
#endif

bool IsRos2BagInput(const fs::path & input)
{
  if (fs::is_directory(input)) {
    return fs::is_regular_file(input / "metadata.yaml");
  }
  return fs::is_regular_file(input) &&
    (input.extension() == ".mcap" || input.extension() == ".db3" ||
    input.filename() == "metadata.yaml");
}

bool RosbagInputAvailable()
{
#ifdef LIDAR_VIEWER_HAS_ROSBAG
  return true;
#else
  return false;
#endif
}

#ifdef LIDAR_VIEWER_HAS_ROSBAG
namespace
{
using PointCloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;

fs::path BagPath(const fs::path & input)
{
  const fs::path path = input.filename() == "metadata.yaml" ? input.parent_path() : input;
  if (!IsRos2BagInput(path)) {
    throw std::runtime_error("Expected a ROS 2 bag directory, .mcap, or .db3: " + input.string());
  }
  return fs::canonical(path);
}

void OpenReader(rosbag2_cpp::Reader & reader, const fs::path & path)
{
  rosbag2_storage::StorageOptions storage;
  storage.uri = path.string();
  if (path.extension() == ".mcap") {
    storage.storage_id = "mcap";
  } else if (path.extension() == ".db3") {
    storage.storage_id = "sqlite3";
  }
  reader.open(storage);
  if (!reader.set_read_order({rosbag2_storage::ReadOrder::ReceivedTimestamp, false})) {
    throw std::runtime_error("Bag storage does not support timestamp-ordered reading: " + path.string());
  }
}

std::string ChooseTopic(rosbag2_cpp::Reader & reader, const std::string & requested)
{
  const auto topics = reader.get_all_topics_and_types();
  std::vector<std::string> point_topics;
  for (const auto & entry : topics) {
    if (entry.type == "sensor_msgs/msg/PointCloud2") {
      point_topics.push_back(entry.name);
      if (entry.name == requested && !requested.empty()) {
        if (entry.serialization_format != "cdr") {
          throw std::runtime_error("PointCloud2 topic must use ROS CDR serialization: " + requested);
        }
        return requested;
      }
    }
  }
  if (requested.empty()) {
    const auto preferred = std::find(point_topics.begin(), point_topics.end(), "/b2/points");
    if (preferred != point_topics.end()) {
      return ChooseTopic(reader, *preferred);
    }
    if (point_topics.size() == 1) {
      return ChooseTopic(reader, point_topics.front());
    }
  }
  std::ostringstream error;
  error << (requested.empty() ? "Choose a PointCloud2 topic using --topic." :
    "No compatible PointCloud2 topic '" + requested + "'. Use --topic to select one.");
  error << " Available PointCloud2 topics:";
  for (const auto & name : point_topics) {
    error << ' ' << name;
  }
  if (point_topics.empty()) {
    error << " (none)";
  }
  throw std::runtime_error(error.str());
}

void FilterTopic(rosbag2_cpp::Reader & reader, const std::string & topic)
{
  rosbag2_storage::StorageFilter filter;
  filter.topics = {topic};
  reader.set_filter(filter);
}

std::size_t ScalarSize(std::uint8_t type)
{
  switch (type) {
    case Field::INT8:
    case Field::UINT8: return 1;
    case Field::INT16:
    case Field::UINT16: return 2;
    case Field::INT32:
    case Field::UINT32:
    case Field::FLOAT32: return 4;
    case Field::FLOAT64: return 8;
    default: throw std::runtime_error("Unsupported PointField datatype: " + std::to_string(type));
  }
}

const Field * FindField(const PointCloud & cloud, const std::string & name, bool required)
{
  const Field * result = nullptr;
  for (const auto & field : cloud.fields) {
    if (field.name != name) {
      continue;
    }
    if (result) {
      throw std::runtime_error("Duplicate PointCloud2 field: " + name);
    }
    if (field.count != 1 ||
      static_cast<std::uint64_t>(field.offset) + ScalarSize(field.datatype) > cloud.point_step)
    {
      throw std::runtime_error("PointCloud2 field exceeds point_step or is not scalar: " + name);
    }
    result = &field;
  }
  if (required && !result) {
    throw std::runtime_error("PointCloud2 is missing field: " + name);
  }
  return result;
}

template<typename T>
double ReadNumber(const std::uint8_t * bytes, bool swap)
{
  std::array<std::uint8_t, sizeof(T)> data{};
  std::memcpy(data.data(), bytes, sizeof(T));
  if (swap) {
    std::reverse(data.begin(), data.end());
  }
  T value;
  std::memcpy(&value, data.data(), sizeof(T));
  return static_cast<double>(value);
}

double ReadScalar(const std::uint8_t * point, const Field & field, bool swap)
{
  const auto * bytes = point + field.offset;
  switch (field.datatype) {
    case Field::INT8: return ReadNumber<std::int8_t>(bytes, false);
    case Field::UINT8: return ReadNumber<std::uint8_t>(bytes, false);
    case Field::INT16: return ReadNumber<std::int16_t>(bytes, swap);
    case Field::UINT16: return ReadNumber<std::uint16_t>(bytes, swap);
    case Field::INT32: return ReadNumber<std::int32_t>(bytes, swap);
    case Field::UINT32: return ReadNumber<std::uint32_t>(bytes, swap);
    case Field::FLOAT32: return ReadNumber<float>(bytes, swap);
    case Field::FLOAT64: return ReadNumber<double>(bytes, swap);
    default: throw std::runtime_error("Unsupported PointField scalar datatype");
  }
}

bool HostBigEndian()
{
  const std::uint16_t value = 1;
  std::uint8_t byte;
  std::memcpy(&byte, &value, 1);
  return byte == 0;
}

struct ReaderCache
{
  fs::path path;
  std::string topic;
  std::unique_ptr<rosbag2_cpp::Reader> reader;
};
}  // namespace

FrameData DecodePointCloud2(const PointCloud & cloud)
{
  FrameData frame;
  frame.frame_id = cloud.header.frame_id;
  const std::uint64_t count = static_cast<std::uint64_t>(cloud.width) * cloud.height;
  if (count == 0) {
    return frame;
  }
  const std::uint64_t packed_row = static_cast<std::uint64_t>(cloud.width) * cloud.point_step;
  const std::uint64_t required_bytes = static_cast<std::uint64_t>(cloud.row_step) * cloud.height;
  if (cloud.point_step == 0 || packed_row > cloud.row_step || required_bytes > cloud.data.size()) {
    throw std::runtime_error("Truncated or inconsistent PointCloud2 point_step/row_step/data layout");
  }
  const Field * x = FindField(cloud, "x", true);
  const Field * y = FindField(cloud, "y", true);
  const Field * z = FindField(cloud, "z", true);
  const Field * intensity = FindField(cloud, "intensity", false);
  if (!intensity) {
    intensity = FindField(cloud, "reflectivity", false);
  }
  if (count > frame.points.max_size()) {
    throw std::runtime_error("PointCloud2 point count exceeds addressable memory");
  }
  frame.points.reserve(static_cast<std::size_t>(count));
  const bool swap = cloud.is_bigendian != HostBigEndian();
  for (std::uint32_t row = 0; row < cloud.height; ++row) {
    const auto * row_data = cloud.data.data() + static_cast<std::size_t>(row) * cloud.row_step;
    for (std::uint32_t column = 0; column < cloud.width; ++column) {
      const auto * point = row_data + static_cast<std::size_t>(column) * cloud.point_step;
      KittiPoint decoded{
        static_cast<float>(ReadScalar(point, *x, swap)),
        static_cast<float>(ReadScalar(point, *y, swap)),
        static_cast<float>(ReadScalar(point, *z, swap)),
        intensity ? static_cast<float>(ReadScalar(point, *intensity, swap)) : 0.0f
      };
      if (!std::isfinite(decoded.x) || !std::isfinite(decoded.y) || !std::isfinite(decoded.z)) {
        continue;
      }
      if (!std::isfinite(decoded.intensity)) {
        decoded.intensity = 0.0f;
      }
      frame.points.push_back(decoded);
    }
  }
  return frame;
}

std::vector<FrameHandle> ResolveBagFrames(
  const fs::path & input, const std::string & requested_topic, const std::string & requested_imu)
{
  const fs::path path = BagPath(input);
  rosbag2_cpp::Reader reader;
  OpenReader(reader, path);
  const std::string topic = ChooseTopic(reader, requested_topic);
  auto imu = std::make_shared<lidar_viewer::ImuTimeline>();
  std::unordered_map<std::string, std::uint64_t> counts;
  for (const auto & entry : reader.get_metadata().topics_with_message_count)
    counts[entry.topic_metadata.name] = entry.message_count;
  std::vector<std::string> imu_topics;
  std::string static_tf_topic;
  for (const auto & entry : reader.get_all_topics_and_types()) {
    if (entry.serialization_format != "cdr" || counts[entry.name] == 0) continue;
    if (entry.type == "sensor_msgs/msg/Imu") imu_topics.push_back(entry.name);
    if (entry.name == "/tf_static" && entry.type == "tf2_msgs/msg/TFMessage") static_tf_topic = entry.name;
  }
  std::sort(imu_topics.begin(), imu_topics.end());
  if (!requested_imu.empty()) {
    if (std::find(imu_topics.begin(), imu_topics.end(), requested_imu) == imu_topics.end())
      throw std::runtime_error("No nonempty CDR IMU topic: " + requested_imu);
    imu->topic = requested_imu;
  } else if (!imu_topics.empty()) {
    // This adapter explicitly flags the VN100's unavailable attitude, unlike
    // the raw driver which publishes a placeholder identity quaternion.
    const auto vn100 = std::find(imu_topics.begin(), imu_topics.end(), "/vn100/imu/data_ned");
    imu->topic = vn100 == imu_topics.end() ? imu_topics.front() : *vn100;
  }
  rosbag2_storage::StorageFilter filter;
  filter.topics = {topic};
  if (!imu->topic.empty()) filter.topics.push_back(imu->topic);
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
  if (!static_tf_topic.empty()) filter.topics.push_back(static_tf_topic);
#endif
  reader.set_filter(filter);
  std::vector<lidar_viewer::ImuObservation> observations;
  bool inconsistent_imu_frame = false;
  std::size_t malformed_imu_records = 0;
  std::vector<FrameHandle> frames;
  std::unordered_map<std::int64_t, std::size_t> duplicates;
  std::cout << "Indexing ROS 2 bag " << path << " topic " << topic
            << " (messages remain on disk)..." << std::endl;
  while (reader.has_next()) {
    const auto message = reader.read_next();
    if (message->topic_name == imu->topic && !imu->topic.empty()) {
      try {
        sensor_msgs::msg::Imu sample;
        rclcpp::SerializedMessage serialized(*message->serialized_data);
        rclcpp::Serialization<sensor_msgs::msg::Imu>().deserialize_message(&serialized, &sample);
        if (imu->frame_id.empty()) imu->frame_id = sample.header.frame_id;
        if (imu->frame_id != sample.header.frame_id || sample.header.frame_id.empty()) inconsistent_imu_frame = true;
        lidar_viewer::ImuObservation observation;
        observation.timestamp_ns = message->recv_timestamp;
        observation.acceleration = {sample.linear_acceleration.x, sample.linear_acceleration.y, sample.linear_acceleration.z};
        observation.angular_velocity = {sample.angular_velocity.x, sample.angular_velocity.y, sample.angular_velocity.z};
        observation.orientation = {sample.orientation.x, sample.orientation.y, sample.orientation.z, sample.orientation.w};
        observation.acceleration_valid = sample.linear_acceleration_covariance[0] >= 0;
        observation.gyro_valid = sample.angular_velocity_covariance[0] >= 0;
        observation.orientation_valid = sample.orientation_covariance[0] >= 0;
        observation.world_ned = sample.header.frame_id.find("_ned") != std::string::npos;
        observations.push_back(observation);
      } catch (const std::exception &) { ++malformed_imu_records; }
      continue;
    }
#ifdef LIDAR_VIEWER_HAS_STATIC_TF
    if (message->topic_name == static_tf_topic && !static_tf_topic.empty()) {
      try {
        tf2_msgs::msg::TFMessage transforms;
        rclcpp::SerializedMessage serialized(*message->serialized_data);
        rclcpp::Serialization<tf2_msgs::msg::TFMessage>().deserialize_message(&serialized, &transforms);
        for (const auto & transform : transforms.transforms) {
          const auto & q = transform.transform.rotation;
          imu->static_rotations.push_back({transform.header.frame_id, transform.child_frame_id, {q.x, q.y, q.z, q.w}});
        }
      } catch (const std::exception &) { /* An invalid optional TF never prevents reading LiDAR. */ }
      continue;
    }
#endif
    if (message->topic_name != topic) {
      continue;
    }
    FrameHandle handle;
    handle.path = path;
    handle.format = InputFormat::Ros2Bag;
    handle.bag_topic = topic;
    handle.bag_timestamp_ns = message->recv_timestamp;
    handle.bag_duplicate_index = duplicates[handle.bag_timestamp_ns]++;
    std::ostringstream label;
    label << path.filename().string() << " #" << std::setw(6) << std::setfill('0') << frames.size();
    handle.label = label.str();
    frames.push_back(std::move(handle));
    if (frames.size() % 1000 == 0) {
      std::cout << "  indexed " << frames.size() << " point-cloud frames" << std::endl;
    }
  }
  if (frames.empty()) {
    throw std::runtime_error("Bag topic has no PointCloud2 messages: " + topic);
  }
  std::stable_sort(frames.begin(), frames.end(), [](const FrameHandle & a, const FrameHandle & b) {
    return a.bag_timestamp_ns < b.bag_timestamp_ns;
  });
  if (inconsistent_imu_frame) imu->status = "IMU frame_id is empty or changes within this bag";
  else {
    imu->received_timestamps.reserve(observations.size());
    for (const auto & sample : observations) imu->received_timestamps.push_back(sample.timestamp_ns);
    // These sensor-family-1 recordings contain a placeholder attitude with
    // zero covariance. Preserve genuine VN100 attitude when it is present.
    if (imu->topic == "/vn100/vectornav_driver_node/imu/data" && imu->frame_id == "vn100_link_ned" &&
      !observations.empty() && std::all_of(observations.begin(), observations.end(), [](const auto & sample) {
        return std::abs(sample.orientation.x) < 1e-8 && std::abs(sample.orientation.y) < 1e-8 &&
          std::abs(sample.orientation.z) < 1e-8 && std::abs(std::abs(sample.orientation.w) - 1) < 1e-8;
      })) {
      for (auto & sample : observations) sample.orientation_valid = false;
    }
    imu->samples = lidar_viewer::EstimateImuGravity(observations);
    imu->status = imu->topic.empty() ? "No IMU topic recorded" : "No reliable IMU gravity estimate";
  }
  for (auto & handle : frames) handle.imu_timeline = imu;
  if (!imu->topic.empty()) std::cout << "IMU: " << imu->topic << " | " << imu->samples.size()
    << " usable gravity samples | " << malformed_imu_records << " malformed records skipped\n";
  std::cout << "Indexed " << frames.size() << " frames; no coordinate, clock, or intensity conversion applied."
            << std::endl;
  return frames;
}

FrameData LoadBagFrame(const FrameHandle & handle)
{
  const fs::path path = BagPath(handle.path);
  // std::async prefetch uses a fresh thread for each frame. A thread-local
  // reader would reopen the complete bag (and its index) on every prefetch.
  // Retain only one storage reader process-wide; serialize the complete cursor
  // operation, including decode, so concurrent UI/prefetch requests cannot
  // seek one another's reader between selecting and loading a frame.
  static ReaderCache cache;
  static std::mutex reader_mutex;
  const std::lock_guard<std::mutex> guard(reader_mutex);
  if (!cache.reader || cache.path != path || cache.topic != handle.bag_topic) {
    cache.reader = std::make_unique<rosbag2_cpp::Reader>();
    OpenReader(*cache.reader, path);
    ChooseTopic(*cache.reader, handle.bag_topic);
    FilterTopic(*cache.reader, handle.bag_topic);
    cache.path = path;
    cache.topic = handle.bag_topic;
  }
  cache.reader->seek(handle.bag_timestamp_ns);
  std::size_t duplicate = 0;
  while (cache.reader->has_next()) {
    const auto message = cache.reader->read_next();
    if (message->topic_name != handle.bag_topic || message->recv_timestamp < handle.bag_timestamp_ns) {
      continue;
    }
    if (message->recv_timestamp != handle.bag_timestamp_ns) {
      break;
    }
    if (duplicate++ != handle.bag_duplicate_index) {
      continue;
    }
    PointCloud cloud;
    rclcpp::SerializedMessage serialized(*message->serialized_data);
    rclcpp::Serialization<PointCloud> serialization;
    serialization.deserialize_message(&serialized, &cloud);
    auto frame = DecodePointCloud2(cloud);
    if (handle.imu_timeline)
      frame.imu = lidar_viewer::MatchFrameImu(*handle.imu_timeline, handle.bag_timestamp_ns, cloud.header.frame_id);
    return frame;
  }
  throw std::runtime_error("Indexed bag frame is missing; input bag may have changed: " + handle.label);
}
#else
std::vector<FrameHandle> ResolveBagFrames(const fs::path &, const std::string &, const std::string &)
{
  throw std::runtime_error(
    "ROS 2 bag support is not compiled. Source /opt/ros/jazzy/setup.bash and reconfigure CMake "
    "with ROS bag support enabled; .bin loading remains available without ROS.");
}

FrameData LoadBagFrame(const FrameHandle &)
{
  throw std::runtime_error("This viewer was built without ROS 2 bag support");
}
#endif
