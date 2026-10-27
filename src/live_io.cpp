#include "lidar_viewer/live_io.hpp"
#include "lidar_viewer/bag_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#endif

namespace lidar_viewer
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
namespace
{
using Steady = std::chrono::steady_clock;
std::int64_t NowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
template<class Message> std::shared_ptr<rclcpp::SerializedMessage> Serialize(const Message & message)
{
  auto output = std::make_shared<rclcpp::SerializedMessage>();
  rclcpp::Serialization<Message>().serialize_message(&message, output.get());
  return output;
}
void NormalizeIntensity(FrameData & frame)
{
  if (frame.points.empty()) return;
  const auto range = std::minmax_element(frame.points.begin(), frame.points.end(),
    [](const auto & a, const auto & b) {return a.intensity < b.intensity;});
  const float low = range.first->intensity, extent = range.second->intensity - low;
  for (auto & point : frame.points) point.intensity = extent > 1e-6f ? (point.intensity-low)/extent : .5f;
}
}

struct LiveSession::Impl
{
  LiveOptions options;
  mutable std::mutex mutex;
  LiveStats stats;
  std::unique_ptr<LiveFrame> latest;
  std::deque<Steady::time_point> cloud_times;
  Steady::time_point last_cloud{}, last_imu{};
  std::vector<ImuObservation> observations;
  std::string imu_frame;
  std::map<std::string, geometry_msgs::msg::TransformStamped> static_transforms;
  rclcpp::Context::SharedPtr context;
  rclcpp::Node::SharedPtr node;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::thread receive_thread, record_thread;

  struct RecordMessage {
    std::shared_ptr<rclcpp::SerializedMessage> data;
    std::string topic, type;
    std::int64_t received, sent;
  };
  std::mutex record_mutex;
  std::condition_variable record_ready;
  std::deque<RecordMessage> record_queue;
  std::size_t queued_bytes{0};
  static constexpr std::size_t max_queued_bytes = 256u * 1024u * 1024u;
  bool record_requested{false}, stop_requested{false}, shutting_down{false};
  std::filesystem::path record_destination;
  std::shared_ptr<rclcpp::SerializedMessage> cached_static_tf;

  template<class Message>
  void Record(const Message & message, const std::string & topic, std::int64_t received)
  {
    std::lock_guard<std::mutex> lock(record_mutex);
    if (!record_requested || stop_requested) return;
    auto serialized = Serialize(message);
    const auto size = serialized->size();
    if (queued_bytes + size > max_queued_bytes) {
      std::lock_guard<std::mutex> state_lock(mutex);
      ++stats.record_dropped;
      stats.record_error = "Recording queue full: messages were dropped. Check disk throughput.";
      return;
    }
    const auto stamp = message.header.stamp;
    record_queue.push_back({std::move(serialized), topic, rosidl_generator_traits::name<Message>(),
      received, static_cast<std::int64_t>(stamp.sec)*1000000000 + stamp.nanosec});
    queued_bytes += size;
    record_ready.notify_one();
  }

  // TFMessage has no common header. Preserve all transforms and their stamps.
  void RecordTf(const tf2_msgs::msg::TFMessage & message, const std::string & topic, std::int64_t received)
  {
    std::lock_guard<std::mutex> lock(record_mutex);
    if (topic == "/tf_static") {
      for (const auto & transform : message.transforms) {
        if (static_transforms.size() < 4096 || static_transforms.count(transform.child_frame_id))
          static_transforms[transform.child_frame_id] = transform;
      }
      tf2_msgs::msg::TFMessage combined;
      for (const auto & transform : static_transforms) combined.transforms.push_back(transform.second);
      cached_static_tf = Serialize(combined);
    }
    if (!record_requested || stop_requested) return;
    auto data = Serialize(message);
    if (queued_bytes + data->size() > max_queued_bytes) {
      std::lock_guard<std::mutex> state_lock(mutex);
      ++stats.record_dropped;
      stats.record_error = "Recording queue full: messages were dropped. Check disk throughput.";
      return;
    }
    queued_bytes += data->size();
    record_queue.push_back({std::move(data), topic, "tf2_msgs/msg/TFMessage", received, received});
    record_ready.notify_one();
  }

  void RecordLoop()
  {
    std::unique_ptr<rosbag2_cpp::Writer> writer;
    for (;;) {
      std::unique_lock<std::mutex> lock(record_mutex);
      record_ready.wait(lock, [&] {return shutting_down || record_requested || writer;});
      if (!writer && record_requested) {
        const auto destination = record_destination;
        const auto static_tf = cached_static_tf;
        lock.unlock();
        try {
          writer = std::make_unique<rosbag2_cpp::Writer>();
          rosbag2_storage::StorageOptions storage;
          storage.uri = destination.string();
          storage.storage_id = "mcap";
          writer->open(storage);
          const auto add_topic = [&](const std::string & name, const std::string & type, bool latched) {
            if (name.empty()) return;
            rosbag2_storage::TopicMetadata topic;
            topic.name=name; topic.type=type; topic.serialization_format="cdr";
            topic.offered_qos_profiles.push_back(latched ? rclcpp::QoS(100).transient_local().reliable() :
              rclcpp::QoS(rclcpp::SensorDataQoS()));
            writer->create_topic(topic);
          };
          add_topic(options.points_topic, "sensor_msgs/msg/PointCloud2", false);
          add_topic(options.imu_topic, "sensor_msgs/msg/Imu", false);
          add_topic("/tf", "tf2_msgs/msg/TFMessage", false);
          add_topic("/tf_static", "tf2_msgs/msg/TFMessage", true);
          if (static_tf) {
            // Writer takes ownership of a serialized buffer; duplicate our cache.
            auto copy = std::make_shared<rclcpp::SerializedMessage>(*static_tf);
            writer->write(copy, "/tf_static", "tf2_msgs/msg/TFMessage", rclcpp::Time(NowNs()));
          }
          std::lock_guard<std::mutex> state_lock(mutex);
          stats.recording=true;
        } catch (const std::exception & error) {
          writer.reset();
          lock.lock();
          record_requested=false; stop_requested=false; record_queue.clear(); queued_bytes=0;
          std::lock_guard<std::mutex> state_lock(mutex);
          stats.record_busy=false; stats.recording=false; stats.record_stopping=false;
          stats.record_error=error.what();
          continue;
        }
        lock.lock();
      }
      if (writer && !record_queue.empty()) {
        auto message = std::move(record_queue.front()); record_queue.pop_front();
        const auto bytes = message.data->size();
        queued_bytes -= bytes;
        lock.unlock();
        try {
          writer->write(message.data, message.topic, message.type, message.received, message.sent);
          std::lock_guard<std::mutex> state_lock(mutex);
          ++stats.recorded_messages; stats.recorded_bytes += bytes;
        } catch (const std::exception & error) {
          {std::lock_guard<std::mutex> state_lock(mutex); stats.record_error=error.what();}
          lock.lock(); stop_requested=true; record_queue.clear(); queued_bytes=0; lock.unlock();
        }
        continue;
      }
      if (writer && (stop_requested || shutting_down)) {
        record_requested=false; stop_requested=false;
        lock.unlock();
        try {writer->close();}
        catch (const std::exception & error) {
          std::lock_guard<std::mutex> state_lock(mutex); stats.record_error=error.what();
        }
        writer.reset();
        std::lock_guard<std::mutex> state_lock(mutex);
        stats.recording=false; stats.record_busy=false; stats.record_stopping=false;
        continue;
      }
      if (shutting_down && !writer) break;
      // A recording with no pending data must sleep until data or stop arrives.
      record_ready.wait(lock, [&] {return shutting_down || stop_requested || !record_queue.empty();});
    }
  }

  void OnImu(const sensor_msgs::msg::Imu & message)
  {
    const auto received = NowNs();
    Record(message, options.imu_topic, received);
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++stats.imu_samples; last_imu=Steady::now();
    }
    if (imu_frame != message.header.frame_id || (!observations.empty() && received<=observations.back().timestamp_ns)) {
      observations.clear(); imu_frame=message.header.frame_id;
    }
    ImuObservation observation;
    observation.timestamp_ns=received;
    observation.acceleration={message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z};
    observation.angular_velocity={message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z};
    observation.orientation={message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w};
    observation.acceleration_valid=message.linear_acceleration_covariance[0]>=0;
    observation.gyro_valid=message.angular_velocity_covariance[0]>=0;
    observation.orientation_valid=message.orientation_covariance[0]>=0;
    observation.world_ned=imu_frame.find("_ned")!=std::string::npos;
    if (options.imu_topic=="/vn100/vectornav_driver_node/imu/data" && imu_frame=="vn100_link_ned" &&
        std::abs(observation.orientation.x)+std::abs(observation.orientation.y)+std::abs(observation.orientation.z)<1e-8 &&
        std::abs(std::abs(observation.orientation.w)-1)<1e-8) observation.orientation_valid=false;
    observations.push_back(observation);
    const auto begin=std::lower_bound(observations.begin(), observations.end(), received-2000000000LL,
      [](const auto & sample, auto stamp) {return sample.timestamp_ns<stamp;});
    observations.erase(observations.begin(), begin);
    if (observations.size()>2000) observations.erase(observations.begin(), observations.end()-2000);
  }

  void OnCloud(const sensor_msgs::msg::PointCloud2 & cloud)
  {
    const auto received=NowNs();
    std::uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(mutex);
      sequence=++stats.clouds; last_cloud=Steady::now(); cloud_times.push_back(last_cloud);
      while (!cloud_times.empty() && last_cloud-cloud_times.front()>std::chrono::seconds(2)) cloud_times.pop_front();
    }
    try {
      Record(cloud, options.points_topic, received);
      auto next=std::make_unique<LiveFrame>();
      next->frame=DecodePointCloud2(cloud); next->sequence=sequence;
      NormalizeIntensity(next->frame);
      ImuTimeline timeline;
      timeline.topic=options.imu_topic; timeline.frame_id=imu_frame;
      timeline.status=options.imu_topic.empty() ? "IMU subscription disabled" : "Waiting for reliable live IMU data";
      timeline.samples=EstimateImuGravity(observations);
      for (const auto & sample : observations) timeline.received_timestamps.push_back(sample.timestamp_ns);
      // Single ROS executor owns observations and transform updates.
      for (const auto & entry : static_transforms) {
        const auto & tf=entry.second;
        timeline.static_rotations.push_back({tf.header.frame_id, tf.child_frame_id,
          {tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w}});
      }
      next->frame.imu=MatchFrameImu(timeline, received, cloud.header.frame_id);
      std::lock_guard<std::mutex> lock(mutex);
      if (latest) ++stats.replaced_frames;
      latest=std::move(next);
    } catch (const std::exception & error) {
      std::lock_guard<std::mutex> lock(mutex);
      ++stats.decode_errors; stats.error=error.what();
    }
  }

  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
  void Start(const LiveOptions & selected)
  {
    options=selected;
    if (options.points_topic.empty()) throw std::runtime_error("LiDAR topic is required");
    if (options.points_topic==options.imu_topic || options.points_topic=="/tf" || options.points_topic=="/tf_static" ||
        options.imu_topic=="/tf" || options.imu_topic=="/tf_static") throw std::runtime_error("Choose distinct sensor topic names");
    context=std::make_shared<rclcpp::Context>(); context->init(0, nullptr);
    auto node_options=rclcpp::NodeOptions().context(context).use_global_arguments(false)
      .start_parameter_services(false).start_parameter_event_publisher(false);
    node=std::make_shared<rclcpp::Node>("lidar_viewer_live", node_options);
    subscriptions.push_back(node->create_subscription<sensor_msgs::msg::PointCloud2>(options.points_topic,
      rclcpp::SensorDataQoS().keep_last(2), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {OnCloud(*cloud);}));
    if (!options.imu_topic.empty()) subscriptions.push_back(node->create_subscription<sensor_msgs::msg::Imu>(options.imu_topic,
      rclcpp::SensorDataQoS().keep_last(200), [this](sensor_msgs::msg::Imu::ConstSharedPtr sample) {
        try {OnImu(*sample);} catch (const std::exception & error) {
          std::lock_guard<std::mutex> lock(mutex); stats.error=error.what();
        }
      }));
    for (const bool latched : {false, true}) {
      const std::string topic=latched ? "/tf_static" : "/tf";
      auto qos=latched ? rclcpp::QoS(100).transient_local().reliable() : rclcpp::QoS(rclcpp::SensorDataQoS());
      subscriptions.push_back(node->create_subscription<tf2_msgs::msg::TFMessage>(topic, qos,
        [this, topic](tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
          try {RecordTf(*message, topic, NowNs());}
          catch (const std::exception & error) {std::lock_guard<std::mutex> lock(mutex); stats.error=error.what();}
        }));
    }
    rclcpp::ExecutorOptions executor_options; executor_options.context=context;
    executor=std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executor_options);
    executor->add_node(node);
    stats.connected=true;
    record_thread=std::thread([this] {RecordLoop();});
    receive_thread=std::thread([this] {
      try {executor->spin();}
      catch (const std::exception & error) {std::lock_guard<std::mutex> lock(mutex); stats.error=error.what(); stats.connected=false;}
    });
  }
  void Stop()
  {
    // Shut down this private context before joining. This also handles an
    // immediate disconnect before the executor thread has entered spin().
    if (context && context->is_valid()) context->shutdown("Viewer disconnected");
    if (executor) executor->cancel();
    if (receive_thread.joinable()) receive_thread.join();
    {std::lock_guard<std::mutex> lock(record_mutex); shutting_down=true; stop_requested=true;}
    record_ready.notify_all();
    if (record_thread.joinable()) record_thread.join();
    subscriptions.clear(); executor.reset(); node.reset();
    context.reset();
    std::lock_guard<std::mutex> lock(mutex); stats.connected=false;
  }
  ~Impl() {Stop();}
};
#else
struct LiveSession::Impl {};
#endif

LiveSession::LiveSession() = default;
LiveSession::~LiveSession() = default;
bool LiveSession::Available()
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  return true;
#else
  return false;
#endif
}
void LiveSession::Connect(const LiveOptions & options)
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (Active()) throw std::runtime_error("Disconnect the current live session first");
  auto replacement=std::make_unique<Impl>(); replacement->Start(options); impl_=std::move(replacement);
#else
  (void)options; throw std::runtime_error("Live ROS 2 input was not built");
#endif
}
void LiveSession::Disconnect()
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (impl_) impl_->Stop();
#endif
}
bool LiveSession::Active() const {return Stats().connected;}
std::unique_ptr<LiveFrame> LiveSession::TakeLatest()
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (impl_) {std::lock_guard<std::mutex> lock(impl_->mutex); return std::move(impl_->latest);}
#endif
  return {};
}
LiveStats LiveSession::Stats() const
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (impl_) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto result=impl_->stats;
    const auto now=Steady::now();
    if (result.clouds) result.cloud_age_ms=std::chrono::duration<double, std::milli>(now-impl_->last_cloud).count();
    if (result.imu_samples) result.imu_age_ms=std::chrono::duration<double, std::milli>(now-impl_->last_imu).count();
    if (impl_->cloud_times.size()>1 && now-impl_->last_cloud<std::chrono::seconds(1))
      result.cloud_hz=(impl_->cloud_times.size()-1)/std::chrono::duration<double>(impl_->cloud_times.back()-impl_->cloud_times.front()).count();
    return result;
  }
#endif
  return {};
}
void LiveSession::StartRecording(const std::filesystem::path & destination)
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (!Active()) throw std::runtime_error("Connect to live sensors before recording");
  if (destination.empty() || std::filesystem::exists(destination)) throw std::runtime_error("Recording needs a new, nonexisting output folder");
  const auto absolute=std::filesystem::absolute(destination);
  std::filesystem::create_directories(absolute.parent_path());
  std::lock_guard<std::mutex> lock(impl_->record_mutex);
  {std::lock_guard<std::mutex> state_lock(impl_->mutex);
    if (impl_->stats.record_busy) throw std::runtime_error("A recording is already active or being finalized");
    impl_->stats.record_busy=true; impl_->stats.record_stopping=false; impl_->stats.recording=false;
    impl_->stats.recorded_messages=0; impl_->stats.recorded_bytes=0; impl_->stats.record_dropped=0;
    impl_->stats.record_error.clear(); impl_->stats.recording_path=absolute;}
  impl_->record_destination=absolute; impl_->record_requested=true; impl_->stop_requested=false;
  impl_->record_ready.notify_one();
#else
  (void)destination; throw std::runtime_error("Live ROS 2 input was not built");
#endif
}
void LiveSession::StopRecording()
{
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
  if (!impl_) return;
  std::lock_guard<std::mutex> lock(impl_->record_mutex);
  if (!impl_->record_requested) return;
  impl_->stop_requested=true;
  {std::lock_guard<std::mutex> state_lock(impl_->mutex); impl_->stats.record_stopping=true;}
  impl_->record_ready.notify_all();
#endif
}
}  // namespace lidar_viewer
