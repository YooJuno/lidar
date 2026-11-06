#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace lidar_viewer
{
enum class InputLoadStage { Opening, Indexing, PreparingImu, ReadingFirstFrame, PreparingDisplay };

struct InputLoadProgress
{
  InputLoadStage stage{InputLoadStage::Opening};
  std::uint64_t messages{0}, total_messages{0};
  std::uint64_t frames{0}, total_frames{0}, imu_samples{0};
  std::optional<double> indexing_eta_seconds;
};

class InputLoadCancelled : public std::runtime_error
{
public:
  InputLoadCancelled() : std::runtime_error("Opening canceled") {}
};

// The worker owns indexing; the UI only reads snapshots and requests a stop.
class InputLoadState
{
public:
  void Cancel() {cancelled_.store(true, std::memory_order_relaxed);}
  bool Cancelled() const {return cancelled_.load(std::memory_order_relaxed);}
  void CheckCancelled() const {if (Cancelled()) throw InputLoadCancelled();}
  void Publish(const InputLoadProgress & progress)
  {
    CheckCancelled();
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto now=std::chrono::steady_clock::now();
    auto eta=progress_.indexing_eta_seconds;
    if(progress.stage!=InputLoadStage::Indexing) {indexing_started_.reset();eta.reset();}
    else {
      if(!indexing_started_ || progress.messages<progress_.messages || progress.total_messages!=progress_.total_messages) {
        indexing_started_=now;eta_sampled_=now;eta.reset();
      }
      const auto elapsed=std::chrono::duration<double>(now-*indexing_started_).count();
      // Ignore startup noise, then update at most once a second. Count every
      // selected message (including IMU), matching the progress bar's units.
      if(progress.total_messages && progress.messages>=progress.total_messages) eta=0;
      else if(elapsed>=1 && progress.messages>0 && progress.total_messages>progress.messages &&
          (!eta || now-eta_sampled_>=std::chrono::seconds(1))) {
        eta=elapsed*(static_cast<double>(progress.total_messages-progress.messages)/progress.messages);
        eta_sampled_=now;
      }
    }
    progress_=progress;
    progress_.indexing_eta_seconds=eta;
  }
  void Stage(InputLoadStage stage)
  {
    CheckCancelled();
    const std::lock_guard<std::mutex> lock(mutex_);
    progress_.stage=stage;
    indexing_started_.reset();progress_.indexing_eta_seconds.reset();
  }
  InputLoadProgress Snapshot() const
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
  }
private:
  std::atomic<bool> cancelled_{false};
  mutable std::mutex mutex_;
  InputLoadProgress progress_;
  std::optional<std::chrono::steady_clock::time_point> indexing_started_;
  std::chrono::steady_clock::time_point eta_sampled_;
};

inline const char * InputLoadStageName(InputLoadStage stage)
{
  switch (stage) {
    case InputLoadStage::Opening: return "Opening recording";
    case InputLoadStage::Indexing: return "Indexing ROS bag";
    case InputLoadStage::PreparingImu: return "Preparing IMU alignment";
    case InputLoadStage::ReadingFirstFrame: return "Reading first point cloud";
    case InputLoadStage::PreparingDisplay: return "Preparing display";
  }
  return "Opening recording";
}
}  // namespace lidar_viewer
