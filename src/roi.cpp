#include "lidar_viewer/roi.hpp"
#include "lidar_viewer/settings_io.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
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
void ValidateSettings(const RoiSettings & value, std::size_t limit=kMaxRoiBoxes)
{
  if (value.boxes.size() > limit) throw std::runtime_error("Too many ROI boxes (maximum "+std::to_string(limit)+")");
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
  if(box.color_index < -1 || box.color_index>=static_cast<int>(kRoiColors.size())) throw std::runtime_error("ROI color must be one of the eight palette entries");
  if (box.mode != RoiBoxMode::Include && box.mode != RoiBoxMode::Exclude)
    throw std::runtime_error("Invalid ROI box mode");
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

PreparedRoi::PreparedRoi(const RoiSettings & settings, const RoiCoordinateFrame & frame, RoiFilterStage stage)
: enabled_(settings.enabled)
{
  // A preview may temporarily combine 128 new keep regions and 128 previously
  // applied exclusions. Persisted/editor settings still have a 128-box limit.
  ValidateSettings(settings,2*kMaxRoiBoxes);
  if (!enabled_) return;
  for (const auto & box : settings.boxes) {
    if (box.mode == RoiBoxMode::Include) {
      if (stage == RoiFilterStage::Preprocessing) continue;
      includes_required_ = true;
    }
    if (!box.enabled) continue;
    auto axes = Axes(box.yaw_deg);
    for (auto & axis : axes) axis = frame.Rotate(axis);
    const auto half = box.size * .5f;
    const Vec3 extent{
      std::abs(axes[0].x)*half.x + std::abs(axes[1].x)*half.y + std::abs(axes[2].x)*half.z,
      std::abs(axes[0].y)*half.x + std::abs(axes[1].y)*half.y + std::abs(axes[2].y)*half.z,
      std::abs(axes[0].z)*half.x + std::abs(axes[1].z)*half.y + std::abs(axes[2].z)*half.z};
    auto & target = box.mode == RoiBoxMode::Exclude ? excludes_ : includes_;
    target.push_back({frame.Transform(box.center), half, extent, axes});
  }
}

bool PreparedRoi::Inside(Vec3 p, const Box & box)
{
  const Vec3 d = p - box.center;
  // Cheap broad phase before the three oriented-axis projections.
  if (std::abs(d.x) > box.extent.x + 2e-5f || std::abs(d.y) > box.extent.y + 2e-5f ||
      std::abs(d.z) > box.extent.z + 2e-5f) return false;
  return std::abs(Dot(d, box.axes[0])) <= box.half_size.x + 1e-5f &&
    std::abs(Dot(d, box.axes[1])) <= box.half_size.y + 1e-5f &&
    std::abs(Dot(d, box.axes[2])) <= box.half_size.z + 1e-5f;
}

bool PreparedRoi::Contains(Vec3 p) const
{
  if (!enabled_) return true;
  if (includes_required_ && std::none_of(includes_.begin(), includes_.end(),
      [&](const Box & box) {return Inside(p, box);})) return false;
  return std::none_of(excludes_.begin(), excludes_.end(),
    [&](const Box & box) {return Inside(p, box);});
}

void PreparedRoi::Filter(std::vector<KittiPoint> & points) const
{
  if (!enabled_ || (!includes_required_ && excludes_.empty())) return;
  points.erase(std::remove_if(points.begin(), points.end(), [&](const auto & p) {
    return !Contains({p.x, p.y, p.z});
  }), points.end());
}

std::size_t PreparedRoi::CountRejected(const std::vector<KittiPoint> & points) const
{
  if (!enabled_ || (!includes_required_ && excludes_.empty())) return 0;
  return static_cast<std::size_t>(std::count_if(points.begin(),points.end(),[&](const auto & p) {
    return !Contains({p.x,p.y,p.z});
  }));
}

std::vector<KittiPoint> PreparedRoi::ExtractRejected(std::vector<KittiPoint> & points) const
{
  std::vector<KittiPoint> skipped;
  if (!enabled_ || (!includes_required_ && excludes_.empty())) return skipped;
  points.erase(std::remove_if(points.begin(),points.end(),[&](const auto & p) {
    if (Contains({p.x,p.y,p.z})) return false;
    skipped.push_back(p);return true;
  }),points.end());
  return skipped;
}

std::string SerializeRoiSettings(const RoiSettings & value, const std::string & prefix)
{
  ValidateSettings(value);
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<float>::max_digits10)
      << prefix << "enabled=" << value.enabled << '\n' << prefix << "show_boxes=" << value.show_boxes
      << '\n' << prefix << "count=" << value.boxes.size() << '\n';
  for (const auto & b : value.boxes) {
    out << prefix << (b.mode == RoiBoxMode::Exclude ? "exclude_box=" : "box=")
        << b.enabled << ' ' << std::quoted(b.name) << ' '
        << b.center.x << ' ' << b.center.y << ' ' << b.center.z << ' '
        << b.size.x << ' ' << b.size.y << ' ' << b.size.z << ' '
        << std::remainder(b.yaw_deg, 360.f) << '\n';
  }
  for(std::size_t i=0;i<value.boxes.size();++i)
    if(value.boxes[i].color_index>=0) out<<prefix<<"color="<<i<<' '<<value.boxes[i].color_index<<'\n';
  return out.str();
}

RoiSettings ParseRoiSettings(const std::string & text, const RoiSettings & fallback, const std::string & prefix)
{
  RoiSettings result;
  bool found = false;
  int count = -1;
  std::map<int,int> colors;
  std::istringstream lines(text);
  std::string line;
  std::size_t number = 0;
  while (std::getline(lines, line)) {
    ++number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    if (line.compare(0, prefix.size(), prefix) != 0) continue;
    const auto key = line.substr(prefix.size(), eq-prefix.size()), value = line.substr(eq + 1);
    if (key != "enabled" && key != "show_boxes" && key != "count" &&
        key != "box" && key != "exclude_box" && key != "color") continue;
    found = true;
    try {
      if(key=="color") {
        std::istringstream fields(value);std::string index,color,extra;
        if(!(fields>>index>>color) || (fields>>extra)) throw std::runtime_error("expected box index and palette color");
        const int i=settings::ParseStrictInt(index),c=settings::ParseStrictInt(color);
        if(i<0 || c<0 || c>=8 || !colors.emplace(i,c).second) throw std::runtime_error("invalid or duplicate ROI color");
      }
      else if (key == "enabled") result.enabled = ParseBool(value);
      else if (key == "show_boxes") result.show_boxes = ParseBool(value);
      else if (key == "count") {
        count = settings::ParseStrictInt(value);
        if (count < 0 || count > static_cast<int>(kMaxRoiBoxes)) throw std::runtime_error("invalid box count");
      } else {
        RoiBox box;
        box.mode = key == "exclude_box" ? RoiBoxMode::Exclude : RoiBoxMode::Include;
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
  for(const auto & entry:colors) {
    if(static_cast<std::size_t>(entry.first)>=result.boxes.size()) throw std::runtime_error("ROI color references a missing box");
    result.boxes[static_cast<std::size_t>(entry.first)].color_index=entry.second;
  }
  return result;
}

bool SameRoiExclusions(const RoiSettings & a, const RoiSettings & b)
{
  auto active = [](const RoiSettings & settings, auto it) {
    return std::find_if(it, settings.boxes.end(), [&](const RoiBox & box) {
      return settings.enabled && box.enabled && box.mode == RoiBoxMode::Exclude;
    });
  };
  auto x = active(a, a.boxes.begin()), y = active(b, b.boxes.begin());
  while (x != a.boxes.end() && y != b.boxes.end()) {
    if (x->center.x != y->center.x || x->center.y != y->center.y || x->center.z != y->center.z ||
        x->size.x != y->size.x || x->size.y != y->size.y || x->size.z != y->size.z || x->yaw_deg != y->yaw_deg)
      return false;
    x = active(a, std::next(x)); y = active(b, std::next(y));
  }
  return x == a.boxes.end() && y == b.boxes.end();
}

RoiSettings ComposeRoiFilters(const RoiSettings & preview, const RoiSettings & applied_exclusions)
{
  RoiSettings result;
  result.show_boxes=preview.show_boxes;
  if (preview.enabled) for (const auto & box:preview.boxes)
    if (box.mode==RoiBoxMode::Include) result.boxes.push_back(box);
  if (applied_exclusions.enabled) for (const auto & box:applied_exclusions.boxes)
    if (box.mode==RoiBoxMode::Exclude && box.enabled) result.boxes.push_back(box);
  result.enabled=!result.boxes.empty();
  return result;
}

RoiSettings GroundExclusionRoi(RoiSettings settings)
{
  settings.enabled=true;
  for (auto & box:settings.boxes) {
    box.mode=RoiBoxMode::Exclude;
    // Migrate only the automatic names used by the previous two-type editor.
    if (box.name.compare(0,8,"Exclude ")==0 && box.name.size()>8 &&
        std::all_of(box.name.begin()+8,box.name.end(),[](char c) {return c>='0' && c<='9';})) {
      for (std::size_t i=1;;++i) {
        const auto name="ROI "+std::to_string(i);
        if (std::none_of(settings.boxes.begin(),settings.boxes.end(),[&](const auto & b) {return b.name==name;})) {
          box.name=name;break;
        }
      }
    }
  }
  return settings;
}
}  // namespace lidar_viewer
