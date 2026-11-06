#pragma once
#include "lidar_viewer/data_io.hpp"
#include <utility>

namespace lidar_viewer
{
struct DatasetBox {std::string class_name;Vec3 center,size;float yaw_deg{0};};
struct DatasetAnnotations {
  std::vector<std::int32_t> point_labels;
  std::vector<DatasetBox> boxes;
  std::vector<std::pair<int,std::size_t>> label_counts;
  std::vector<std::pair<int,std::string>> class_names;
  std::string source,error;
  bool draft{false};
  bool box_annotations{false};
};
struct DatasetSample
{
  fs::path root, metadata, labels;
  std::string name, source, coordinate_space, annotation_status, category;
  std::vector<std::string> class_hints;
  std::size_t source_frame{0}, point_count{0};
};
bool IsDatasetFile(const fs::path & path);
std::vector<FrameHandle> ResolveDatasetFrames(const fs::path & path,InputLoadState * load=nullptr,bool include_bin=false);
FrameData LoadDatasetFrame(const FrameHandle & handle);
}  // namespace lidar_viewer
