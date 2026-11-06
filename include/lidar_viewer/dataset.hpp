#pragma once
#include "lidar_viewer/data_io.hpp"

namespace lidar_viewer
{
struct DatasetSample
{
  fs::path root, metadata, labels;
  std::string name, source, coordinate_space, annotation_status, category;
  std::size_t source_frame{0}, point_count{0};
};
bool IsDatasetFile(const fs::path & path);
std::vector<FrameHandle> ResolveDatasetFrames(const fs::path & path,InputLoadState * load=nullptr);
FrameData LoadDatasetFrame(const FrameHandle & handle);
}  // namespace lidar_viewer
