#pragma once
#include <filesystem>
#include <memory>
namespace lidar_viewer
{
class TrainingWorkspace
{
public:
  TrainingWorkspace();~TrainingWorkspace();
  void Draw(bool & open,const std::filesystem::path & dataset,float scale);
  bool Running() const;
private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
