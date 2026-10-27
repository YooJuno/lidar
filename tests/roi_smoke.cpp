#include "lidar_viewer/roi.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void Check(bool value, const char * message)
{
  if (!value) throw std::runtime_error(message);
}
template<class F> void Reject(F operation)
{
  bool rejected = false;
  try {operation();} catch (const std::exception &) {rejected = true;}
  Check(rejected, "invalid ROI was accepted");
}
}
int main()
{
  using namespace lidar_viewer;
  try {
    RoiBox a{"first \"box\" \\ A", true, {3, -2, 1}, {4, 2, 6}, 90};
    RoiSettings settings{true, true, {a}};
    PreparedRoi filter(settings);
    Check(filter.Contains(a.center), "box center rejected");
    Check(filter.Contains({3, 0, 4}), "boundary rejected");
    Check(!filter.Contains({4.01f, -2, 1}), "yaw ignored");
    Check(!filter.Contains({3, -2, 4.01f}), "Z extent ignored");
    for (auto corner : RoiCorners(a)) Check(filter.Contains(corner), "corner rejected");
    RoiBox b{"second", true, {-8, 0, 0}, {2, 2, 2}, -45};
    settings.boxes.push_back(b);
    settings.boxes.push_back(a);  // An overlap never duplicates points.
    std::vector<KittiPoint> points{{3,-2,1,7}, {-8,0,0,9}, {50,50,50,11}};
    PreparedRoi(settings).Filter(points);
    Check(points.size() == 2 && points[0].intensity == 7 && points[1].intensity == 9,
      "union, order or point attributes changed");
    settings.boxes[1].enabled = false;
    Check(!PreparedRoi(settings).Contains(b.center), "disabled box accepted point");
    settings.boxes.clear();
    Check(PreparedRoi(settings).RejectsAll() && !PreparedRoi(settings).Contains({}), "empty active ROI passed points");
    settings.enabled = false;
    Check(PreparedRoi(settings).Contains({50,50,50}), "master off rejected point");

    // Same invariant under the viewer's roll/pitch/yaw + KITTI alignment + Z shift.
    settings = {true, false, {a, b}};
    RoiCoordinateFrame transform;
    const float roll=.31f, pitch=-.23f, yaw=.7f;
    const float cr=std::cos(roll), sr=std::sin(roll), cp=std::cos(pitch), sp=std::sin(pitch),
      cy=std::cos(yaw), sy=std::sin(yaw);
    transform.rotation = {cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy,
      cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy, -sp, sr*cp, cr*cp};
    transform.translation = transform.Rotate({0,0,2.1f}) + Vec3{0,0,.57f};
    const PreparedRoi raw(settings), leveled(settings, transform);
    for (int x=-20; x<=20; ++x) for (int y=-10; y<=10; ++y) for (int z=-8; z<=8; ++z) {
      const Vec3 p{x*.71f, y*.63f, z*.77f};
      Check(raw.Contains(p) == leveled.Contains(transform.Transform(p)), "ROI changed with display/IMU transform");
    }
    for (const auto & box : settings.boxes)
      for (auto p : RoiCorners(box)) Check(leveled.Contains(transform.Transform(p)), "leveled boundary rejected");

    const auto serialized = SerializeRoiSettings(settings);
    const auto parsed = ParseRoiSettings("profile=default\n" + serialized);
    Check(SerializeRoiSettings(parsed) == serialized, "ROI serialization roundtrip failed");
    Check(SerializeRoiSettings(ParseRoiSettings("grid_enabled=1\n", settings)) == serialized, "absent ROI replaced fallback");
    Check(ParseRoiSettings("roi_enabled=1\nroi_count=0\n", settings).boxes.empty(), "empty section did not clear fallback");
    for (const auto & invalid : {
        "roi_count=1\n", "roi_count=-1\n", "roi_count=129\n", "roi_enabled=2\nroi_count=0\n",
        "roi_count=1\nroi_box=1 unquoted 0 0 0 1 1 1 0\n",
        "roi_count=1\nroi_box=1 \"a\" nan 0 0 1 1 1 0\n",
        "roi_count=1\nroi_box=1 \"a\" 0 0 0 1 0 1 0\n",
        "roi_count=1\nroi_box=1 \"a\" 0 0 0 1 1 1 0 extra\n"}) {
      Reject([&] {ParseRoiSettings(invalid, settings);});
      Check(SerializeRoiSettings(settings) == serialized, "malformed section mutated fallback");
    }
    auto invalid=a;
    invalid.size.x=std::numeric_limits<float>::infinity();
    Reject([&] {ValidateRoiBox(invalid);});
    invalid=a; invalid.name="line\nbreak";
    Reject([&] {SerializeRoiSettings({true,true,{invalid}});});
    std::cout << "ROI geometry, union, rigid transforms and persistence passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
