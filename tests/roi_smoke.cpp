#include "lidar_viewer/roi.hpp"
#include "lidar_viewer/roi_editor.hpp"
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
    const Vec3 original{3,-4,2};
    const auto recovered=transform.InverseTransform(transform.Transform(original));
    Check(std::abs(original.x-recovered.x)<1e-4f && std::abs(original.y-recovered.y)<1e-4f &&
      std::abs(original.z-recovered.z)<1e-4f,"inverse ROI frame changed coordinates");
    const auto from_view=RoiPointFromTopView(.2f,-.3f,1.5f,20,{2,3,4},.7f,transform,-2);
    Check(from_view && std::abs(from_view->z+2)<1e-4f,"top-view ray missed raw height plane");
    const auto displayed=transform.Transform(*from_view);
    const auto expected=Vec3{2,3,4}+Vec3{std::cos(.7f),std::sin(.7f),0}*6+TopViewUp(.7f)*-6;
    Check(std::abs(displayed.x-expected.x)<1e-4f && std::abs(displayed.y-expected.y)<1e-4f,
      "drawing ignored rotated top view or IMU tilt");
    const auto drag=RoiFromDrag({5,7,0},{-3,1,0},-2,4,"drawn");
    Check(drag.center.x==1 && drag.center.y==4 && drag.center.z==-2 && drag.size.x==8 && drag.size.y==6,
      "reversed drag changed ROI dimensions");
    RoiSettings edited{true,true,{drag}};
    const auto duplicate=DuplicateRoi(drag,edited);
    Check(!PreparedRoi(edited).Contains(duplicate.center),"duplicate overlaps source center");
    RoiHistory history; history.Record(edited); edited.boxes.push_back(duplicate);
    const auto after=edited;
    Check(history.Undo(edited) && edited.boxes.size()==1 && history.Redo(edited) && SameRoi(edited,after),
      "ROI undo/redo changed box data");
    history.Undo(edited);history.Record(edited);edited.boxes.clear();
    Check(!history.CanRedo(),"new ROI edit retained stale redo history");

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
