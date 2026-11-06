#pragma once
#include "lidar_viewer/dataset.hpp"

namespace lidar_viewer {
struct DatasetTrashEntry {
  fs::path directory,root,manifest;
  std::vector<std::pair<fs::path,fs::path>> files; // original -> recovery copy
};
// Move a sample and its own annotations into the dataset's recovery folder.
// Dataset manifests are edited atomically; unrelated JSON values stay intact.
DatasetTrashEntry TrashDatasetSample(const FrameHandle & frame);
void RestoreDatasetSample(const DatasetTrashEntry & entry);
} // namespace lidar_viewer
