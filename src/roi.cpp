#include "lidar_viewer/roi.hpp"
#include "lidar_viewer/settings_io.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lidar_viewer
{
namespace
{
constexpr float kRadians = 0.017453292519943295f;
std::array<Vec3, 3> Axes(float yaw_deg)
{
  const float yaw = std::remainder(yaw_deg, 360.f) * kRadians;
  const float c = std::cos(yaw), s = std::sin(yaw);
  return {{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}};
}
float Dot(Vec3 a, Vec3 b) {return a.x * b.x + a.y * b.y + a.z * b.z;}
bool ParseBool(const std::string & value)
{
  const int result = settings::ParseStrictInt(value);
  if (result != 0 && result != 1) throw std::runtime_error("expected 0 or 1");
  return result != 0;
}
void ValidateSettings(const RoiSettings & value)
{
  if (value.boxes.size() > kMaxRoiBoxes) throw std::runtime_error("Too many ROI boxes (maximum 128)");
  for (const auto & box : value.boxes) ValidateRoiBox(box);
}
}  // namespace

Vec3 RoiCoordinateFrame::Rotate(Vec3 p) const
{
  return {rotation[0]*p.x + rotation[1]*p.y + rotation[2]*p.z,
    rotation[3]*p.x + rotation[4]*p.y + rotation[5]*p.z,
    rotation[6]*p.x + rotation[7]*p.y + rotation[8]*p.z};
}
Vec3 RoiCoordinateFrame::Transform(Vec3 p) const {return Rotate(p) + translation;}
Vec3 RoiCoordinateFrame::InverseTransform(Vec3 p) const
{
  p = p - translation;
  return {rotation[0]*p.x + rotation[3]*p.y + rotation[6]*p.z,
    rotation[1]*p.x + rotation[4]*p.y + rotation[7]*p.z,
    rotation[2]*p.x + rotation[5]*p.y + rotation[8]*p.z};
}

void ValidateRoiBox(const RoiBox & box)
{
  if (box.name.size() > 127 || box.name.find_first_of("\r\n") != std::string::npos ||
      box.name.find('\0') != std::string::npos)
    throw std::runtime_error("ROI name must be at most 127 bytes without line breaks");
  for (float v : {box.center.x, box.center.y, box.center.z, box.size.x, box.size.y,
      box.size.z, box.yaw_deg}) {
    if (!std::isfinite(v) || std::abs(v) > 1e6f)
      throw std::runtime_error("ROI values must be finite and within +/-1000000");
  }
  if (std::min({box.size.x, box.size.y, box.size.z}) < .001f)
    throw std::runtime_error("Each ROI size must be at least 0.001 m");
}

std::array<Vec3, 8> RoiCorners(const RoiBox & box)
{
  ValidateRoiBox(box);
  const auto axes = Axes(box.yaw_deg);
  std::array<Vec3, 8> result;
  for (unsigned i = 0; i < result.size(); ++i) {
    result[i] = box.center + axes[0] * (box.size.x * (i & 1 ? .5f : -.5f)) +
      axes[1] * (box.size.y * (i & 2 ? .5f : -.5f)) +
      axes[2] * (box.size.z * (i & 4 ? .5f : -.5f));
  }
  return result;
}

PreparedRoi::PreparedRoi(const RoiSettings & settings, const RoiCoordinateFrame & frame)
: enabled_(settings.enabled)
{
  ValidateSettings(settings);
  if (!enabled_) return;
  boxes_.reserve(settings.boxes.size());
  for (const auto & box : settings.boxes) {
    if (!box.enabled) continue;
    auto axes = Axes(box.yaw_deg);
    for (auto & axis : axes) axis = frame.Rotate(axis);
    const auto half = box.size * .5f;
    const Vec3 extent{
      std::abs(axes[0].x)*half.x + std::abs(axes[1].x)*half.y + std::abs(axes[2].x)*half.z,
      std::abs(axes[0].y)*half.x + std::abs(axes[1].y)*half.y + std::abs(axes[2].y)*half.z,
      std::abs(axes[0].z)*half.x + std::abs(axes[1].z)*half.y + std::abs(axes[2].z)*half.z};
    boxes_.push_back({frame.Transform(box.center), half, extent, axes});
  }
}

bool PreparedRoi::Contains(Vec3 p) const
{
  if (!enabled_) return true;
  for (const auto & box : boxes_) {
    const Vec3 d = p - box.center;
    // Cheap broad phase before the three oriented-axis projections.
    if (std::abs(d.x) > box.extent.x + 2e-5f || std::abs(d.y) > box.extent.y + 2e-5f ||
        std::abs(d.z) > box.extent.z + 2e-5f) continue;
    if (std::abs(Dot(d, box.axes[0])) <= box.half_size.x + 1e-5f &&
        std::abs(Dot(d, box.axes[1])) <= box.half_size.y + 1e-5f &&
        std::abs(Dot(d, box.axes[2])) <= box.half_size.z + 1e-5f) return true;
  }
  return false;
}

void PreparedRoi::Filter(std::vector<KittiPoint> & points) const
{
  if (!enabled_) return;
  points.erase(std::remove_if(points.begin(), points.end(), [&](const auto & p) {
    return !Contains({p.x, p.y, p.z});
  }), points.end());
}

std::string SerializeRoiSettings(const RoiSettings & value)
{
  ValidateSettings(value);
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<float>::max_digits10)
      << "roi_enabled=" << value.enabled << '\n' << "roi_show_boxes=" << value.show_boxes
      << '\n' << "roi_count=" << value.boxes.size() << '\n';
  for (const auto & b : value.boxes) {
    out << "roi_box=" << b.enabled << ' ' << std::quoted(b.name) << ' '
        << b.center.x << ' ' << b.center.y << ' ' << b.center.z << ' '
        << b.size.x << ' ' << b.size.y << ' ' << b.size.z << ' '
        << std::remainder(b.yaw_deg, 360.f) << '\n';
  }
  return out.str();
}

RoiSettings ParseRoiSettings(const std::string & text, const RoiSettings & fallback)
{
  RoiSettings result;
  bool found = false;
  int count = -1;
  std::istringstream lines(text);
  std::string line;
  std::size_t number = 0;
  while (std::getline(lines, line)) {
    ++number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    const auto key = line.substr(0, eq), value = line.substr(eq + 1);
    if (key != "roi_enabled" && key != "roi_show_boxes" && key != "roi_count" && key != "roi_box") continue;
    found = true;
    try {
      if (key == "roi_enabled") result.enabled = ParseBool(value);
      else if (key == "roi_show_boxes") result.show_boxes = ParseBool(value);
      else if (key == "roi_count") {
        count = settings::ParseStrictInt(value);
        if (count < 0 || count > static_cast<int>(kMaxRoiBoxes)) throw std::runtime_error("invalid box count");
      } else {
        RoiBox box;
        std::istringstream fields(value);
        std::string enabled, token;
        if (!(fields >> enabled >> std::ws) || fields.peek() != '"' || !(fields >> std::quoted(box.name)))
          throw std::runtime_error("expected enabled flag and quoted name");
        box.enabled = ParseBool(enabled);
        for (float * target : {&box.center.x, &box.center.y, &box.center.z,
            &box.size.x, &box.size.y, &box.size.z, &box.yaw_deg}) {
          if (!(fields >> token)) throw std::runtime_error("missing box coordinate");
          *target = settings::ParseFiniteFloat(token);
        }
        if (fields >> token) throw std::runtime_error("unexpected box field");
        ValidateRoiBox(box);
        box.yaw_deg = std::remainder(box.yaw_deg, 360.f);
        result.boxes.push_back(std::move(box));
        if (result.boxes.size() > kMaxRoiBoxes) throw std::runtime_error("too many boxes");
      }
    } catch (const std::exception & error) {
      throw std::runtime_error("ROI line " + std::to_string(number) + ": " + error.what());
    }
  }
  if (!found) return fallback;
  if (count < 0 || static_cast<std::size_t>(count) != result.boxes.size())
    throw std::runtime_error("ROI count does not match box entries");
  return result;
}
}  // namespace lidar_viewer
