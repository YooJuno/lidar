#pragma once

#include "lidar_viewer/data_io.hpp"
#include "lidar_viewer/display_alignment.hpp"
#include "lidar_viewer/roi.hpp"

#include <atomic>
#include <memory>
#include <optional>

namespace lidar_viewer
{
struct RoiExportSource
{
  FrameHandle handle;
  std::size_t frame_index{0}; // Zero-based index in the viewer's source list.
};

struct RoiExportRequest
{
  // Must not already exist; failed/canceled jobs remove only their new folder.
  fs::path destination;
  RoiSettings regions;
  bool aligned_coordinates{false};
  DisplayAlignment calibration;
  bool imu_level_enabled{false};
  ImuMountSettings imu_mount;
  float global_ground_offset{0};
  std::string profile;
  // Class hints for annotation, separate from point/box ground truth.
  std::string class_name;
  bool roi_names_as_labels{true};
  std::vector<RoiExportSource> sources;
  // Live input is already decoded, before calibration or ground processing.
  std::optional<FrameData> live_snapshot;
  std::uint64_t live_sequence{0};
  std::string live_topic;
};

struct RoiExportProgress
{
  std::atomic<bool> cancel{false};
  std::atomic<std::size_t> completed{0}, exported{0}, empty{0};
  std::atomic<std::uint64_t> points{0};
};

struct RoiExportResult
{
  fs::path destination;
  std::size_t frames{0}, empty_frames{0};
  std::uint64_t points{0};
  bool canceled{false};
};

// Short source/frame/region names, safe on all supported platforms.
// Repeated exports receive a numeric suffix; the worker still creates its
// destination exclusively to protect against races and accidental overwrites.
std::string RoiExportDatasetName(const RoiExportRequest & request);
fs::path AvailableRoiExportDestination(const fs::path & parent,const RoiExportRequest & request);

// CPU/file I/O only, intended for a reusable BackgroundWorker. Original point
// order/intensity are preserved; display colors, exclusions and predictions
// never become labels. Every successful sample has identical NPY/PCD data.
RoiExportResult ExportRoiDataset(const RoiExportRequest & request,
  const std::shared_ptr<RoiExportProgress> & progress);
}  // namespace lidar_viewer
