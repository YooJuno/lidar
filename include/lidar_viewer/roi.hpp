#pragma once

#include "lidar_viewer/data_io.hpp"

#include <array>
#include <string>
#include <vector>

namespace lidar_viewer
{
constexpr std::size_t kMaxRoiBoxes = 128;
enum class RoiBoxMode {Include, Exclude};

// Coordinates in the original LiDAR frame; size is the full extent in metres.
struct RoiBox
{
  std::string name{"ROI"};
  bool enabled{true};
  Vec3 center{};
  Vec3 size{20, 20, 6};
  float yaw_deg{0};
  RoiBoxMode mode{RoiBoxMode::Include};
};

struct RoiSettings
{
  bool enabled{false};
  bool show_boxes{true};
  std::vector<RoiBox> boxes;
};

struct RoiCoordinateFrame
{
  std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  Vec3 translation{};
  Vec3 Rotate(Vec3 point) const;
  Vec3 Transform(Vec3 point) const;
  Vec3 InverseTransform(Vec3 point) const;
};

void ValidateRoiBox(const RoiBox & box);
std::array<Vec3, 8> RoiCorners(const RoiBox & box);
std::string SerializeRoiSettings(const RoiSettings & settings, const std::string & prefix = "roi_");
// A malformed ROI section throws without changing the caller's settings.
// Files without an ROI section leave the supplied fallback intact.
RoiSettings ParseRoiSettings(const std::string & text, const RoiSettings & fallback = {},
  const std::string & prefix = "roi_");
// Ignore names, outlines and include regions when checking analysis invalidation.
bool SameRoiExclusions(const RoiSettings & a, const RoiSettings & b);
// Normalize only the ground-exclusion collection; display ROIs are separate.
// Older display/filter flags and box types do not gate ground exclusion.
RoiSettings GroundExclusionRoi(RoiSettings settings);
// Include regions preview immediately; exclusions use the last applied snapshot.
RoiSettings ComposeRoiFilters(const RoiSettings & preview, const RoiSettings & applied_exclusions);

enum class RoiFilterStage {Display, Preprocessing};

class PreparedRoi
{
public:
  explicit PreparedRoi(const RoiSettings & settings, const RoiCoordinateFrame & frame = {},
    RoiFilterStage stage = RoiFilterStage::Display);
  bool Contains(Vec3 point) const;
  void Filter(std::vector<KittiPoint> & points) const;
  std::size_t CountRejected(const std::vector<KittiPoint> & points) const;
  std::vector<KittiPoint> ExtractRejected(std::vector<KittiPoint> & points) const;
  bool RejectsAll() const {return enabled_ && includes_required_ && includes_.empty();}

private:
  struct Box {Vec3 center, half_size, extent; std::array<Vec3, 3> axes;};
  static bool Inside(Vec3 point, const Box & box);
  bool enabled_;
  bool includes_required_{false};
  std::vector<Box> includes_, excludes_;
};
}  // namespace lidar_viewer
