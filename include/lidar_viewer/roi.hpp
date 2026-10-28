#pragma once

#include "lidar_viewer/data_io.hpp"

#include <array>
#include <string>
#include <vector>

namespace lidar_viewer
{
constexpr std::size_t kMaxRoiBoxes = 128;

// Coordinates in the original LiDAR frame; size is the full extent in metres.
struct RoiBox
{
  std::string name{"ROI"};
  bool enabled{true};
  Vec3 center{};
  Vec3 size{20, 20, 6};
  float yaw_deg{0};
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
std::string SerializeRoiSettings(const RoiSettings & settings);
// A malformed ROI section throws without changing the caller's settings.
// Files without an ROI section leave the supplied fallback intact.
RoiSettings ParseRoiSettings(const std::string & text, const RoiSettings & fallback = {});

class PreparedRoi
{
public:
  explicit PreparedRoi(const RoiSettings & settings, const RoiCoordinateFrame & frame = {});
  bool Contains(Vec3 point) const;
  void Filter(std::vector<KittiPoint> & points) const;
  bool RejectsAll() const {return enabled_ && boxes_.empty();}

private:
  struct Box {Vec3 center, half_size, extent; std::array<Vec3, 3> axes;};
  bool enabled_;
  std::vector<Box> boxes_;
};
}  // namespace lidar_viewer
