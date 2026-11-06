#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lidar_viewer
{
// One owned process group/job object. Polling never waits for training or GPU I/O.
class TrainingJob
{
public:
  TrainingJob();~TrainingJob();
  TrainingJob(const TrainingJob &)=delete;TrainingJob & operator=(const TrainingJob &)=delete;
  void Start(const std::filesystem::path & python,const std::filesystem::path & script,
    const std::filesystem::path & request,const std::filesystem::path & log,bool validate_only);
  void Poll();void Cancel();
  bool Running() const;bool Stopping() const;
  std::optional<int> ExitCode() const;
  std::string LogTail(std::size_t limit=32768) const;
private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}  // namespace lidar_viewer
