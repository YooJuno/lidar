#include "lidar_viewer/live_io.hpp"
#include "lidar_viewer/bag_io.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "lidar_viewer/platform.hpp"

#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#endif

namespace
{
void Check(bool value, const char * message) {if (!value) throw std::runtime_error(message);}
template<class F> void Wait(F condition)
{
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(8);
  while (!condition()) {
    if (std::chrono::steady_clock::now()>deadline) throw std::runtime_error("Live test timed out");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
sensor_msgs::msg::PointCloud2 Cloud(float x)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id="test_lidar";
  cloud.header.stamp=rclcpp::Clock(RCL_SYSTEM_TIME).now();
  cloud.width=2; cloud.height=1; cloud.point_step=16; cloud.row_step=32;
  for (const auto & name : {"x", "y", "z", "intensity"}) {
    sensor_msgs::msg::PointField field;
    field.name=name; field.offset=static_cast<std::uint32_t>(cloud.fields.size()*4);
    field.datatype=sensor_msgs::msg::PointField::FLOAT32; field.count=1;
    cloud.fields.push_back(field);
  }
  const float points[]{x,0,-1,7, x+1,1,1,19};
  cloud.data.resize(sizeof(points)); std::memcpy(cloud.data.data(), points, sizeof(points));
  return cloud;
}
#endif
}

int main()
{
  const auto root=std::filesystem::temp_directory_path()/("lidar-live-test-"+std::to_string(lidar_viewer::ProcessId()));
  try {
    lidar_viewer::LiveSession live;
    Check(!live.Active() && !live.TakeLatest(), "Disconnected receiver returned data");
#ifdef LIDAR_VIEWER_HAS_LIVE_ROS
    auto context=std::make_shared<rclcpp::Context>(); context->init(0,nullptr);
    auto node=std::make_shared<rclcpp::Node>("lidar_live_test_publisher", rclcpp::NodeOptions().context(context));
    const lidar_viewer::LiveOptions options{"/lidar_test/points", "/lidar_test/imu"};
    auto points=node->create_publisher<sensor_msgs::msg::PointCloud2>(options.points_topic,rclcpp::SensorDataQoS());
    auto imu=node->create_publisher<sensor_msgs::msg::Imu>(options.imu_topic,rclcpp::SensorDataQoS());
    auto tf=node->create_publisher<tf2_msgs::msg::TFMessage>("/tf_static",rclcpp::QoS(10).reliable().transient_local());
    live.Connect(options);
    Wait([&] {return points->get_subscription_count()>0 && imu->get_subscription_count()>0 && tf->get_subscription_count()>0;});
    tf2_msgs::msg::TFMessage transform;
    geometry_msgs::msg::TransformStamped mount;
    mount.header.frame_id="test_lidar"; mount.child_frame_id="test_imu"; mount.transform.rotation.w=1;
    transform.transforms.push_back(mount); tf->publish(transform);
    sensor_msgs::msg::Imu sample;
    sample.header.frame_id="test_imu"; sample.orientation_covariance[0]=-1; sample.linear_acceleration.z=9.80665;
    auto publish_imu=[&] {sample.header.stamp=node->now(); imu->publish(sample);};
    for (int i=0;i<15;++i) {publish_imu(); std::this_thread::sleep_for(std::chrono::milliseconds(5));}
    points->publish(Cloud(1));
    std::unique_ptr<lidar_viewer::LiveFrame> frozen;
    Wait([&] {if (auto next=live.TakeLatest()) frozen=std::move(next); return frozen!=nullptr;});
    Check(frozen->frame.points.size()==2 && frozen->frame.points[0].intensity==0 && frozen->frame.points[1].intensity==1,
      "Live PointCloud2 decode/normalization failed");
    Check(lidar_viewer::ComputeImuLeveling(frozen->frame.imu, {}).applied, "Live IMU/static TF alignment missing");
    const auto frozen_sequence=frozen->sequence;
    const float frozen_x=frozen->frame.points[0].x;
    live.StartRecording(root/"first");
    Wait([&] {return live.Stats().recording || !live.Stats().record_error.empty();});
    Check(live.Stats().recording, live.Stats().record_error.c_str());
    // While a viewer retains its frozen frame, receiving and recording continue.
    for (int i=0;i<80;++i) {
      publish_imu();
      if (i%4==0) points->publish(Cloud(static_cast<float>(i+10)));
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Wait([&] {return live.Stats().clouds>=15 && live.Stats().recorded_messages>=40;});
    Check(frozen->sequence==frozen_sequence && frozen->frame.points[0].x==frozen_x,
      "Receiver mutated a displayed/frozen frame");
    auto newest=live.TakeLatest();
    Check(newest && newest->sequence>frozen_sequence && newest->frame.points[0].x>frozen_x,
      "Resume failed to obtain the newest frame");
    Check(live.Stats().replaced_frames>0 && live.Stats().record_dropped==0, "Latest-frame replacement or recording failed");
    live.StopRecording();
    Wait([&] {return !live.Stats().record_busy;});
    Check(std::filesystem::is_regular_file(root/"first/metadata.yaml"), "Stop did not finalize bag metadata");
    auto frames=ResolveBagFrames(root/"first", options.points_topic, options.imu_topic);
    Check(frames.size()>=10, "Point clouds were not recorded during display pause");
    const auto recorded=LoadBagFrame(frames.back());
    Check(recorded.points[0].intensity==7 && recorded.points[1].intensity==19,
      "Recording saved display-normalized points instead of original messages");
    Check(recorded.imu && recorded.imu->imu_to_lidar, "Cached static TF was not included in recording");
    live.StartRecording(root/"second");
    Wait([&] {return live.Stats().recording;});
    for (int i=0;i<10;++i) {publish_imu(); points->publish(Cloud(100+i)); std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    live.Disconnect();
    Check(!live.Active() && std::filesystem::is_regular_file(root/"second/metadata.yaml"), "Disconnect failed to finalize recording");
    Check(context->is_valid(), "Viewer disconnected another ROS context");
    // Same object can connect again; subscription state does not leak across sessions.
    live.Connect(options); Check(live.Active(), "Reconnect failed"); live.Disconnect();
    context->shutdown("Live test complete");
#else
    Check(!lidar_viewer::LiveSession::Available(), "Unavailable backend reported ready");
    bool rejected=false; try {live.Connect({});} catch (const std::exception &) {rejected=true;}
    Check(rejected, "Unavailable backend accepted a connection");
#endif
    std::filesystem::remove_all(root);
    std::cout << "Live receive, frozen-frame ownership, newest-frame resume and raw recording passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(root);
    return 1;
  }
}
