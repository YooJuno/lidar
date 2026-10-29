#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace lidar_viewer
{
enum class InputLoadStage { Opening, Indexing, PreparingImu, ReadingFirstFrame, PreparingDisplay };

struct InputLoadProgress
{
  InputLoadStage stage{InputLoadStage::Opening};
  std::uint64_t messages{0}, total_messages{0};
  std::uint64_t frames{0}, total_frames{0}, imu_samples{0};
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
    progress_=progress;
  }
  void Stage(InputLoadStage stage)
  {
    CheckCancelled();
    const std::lock_guard<std::mutex> lock(mutex_);
    progress_.stage=stage;
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
