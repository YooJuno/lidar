#pragma once
#include <filesystem>
#include <memory>

namespace lidar_viewer {
// One independent review window per studio instance. Closing the main viewer
// leaves the review window available; exited children are reaped on reuse.
class ReviewWindowProcess {
public:
  ReviewWindowProcess();
  ~ReviewWindowProcess();
  bool Open(const std::filesystem::path & executable,const std::filesystem::path & log);
  void Poll();
private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
