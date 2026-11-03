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
    Check(!PreparedRoi(settings).RejectsAll() && PreparedRoi(settings).Contains({}), "deleting last box hid the full cloud");
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
    const float root=std::sqrt(.5f);
    const auto rotated_drag=RoiFromDrag({0,0,0},{2*root,6*root,0},-2,4,"rotated",45);
    Check(std::abs(rotated_drag.size.x-4)<1e-4f && std::abs(rotated_drag.size.y-2)<1e-4f &&
      rotated_drag.yaw_deg==45,"drawn ROI ignored view orientation");
    RoiSettings edited{true,true,{drag}};
    const auto resized=ResizeRoi(a,{1,1,a.center.z},1,1);
    Check(std::abs(resized.size.x-5)<1e-5f && std::abs(resized.size.y-3)<1e-5f &&
      std::abs(resized.center.x-2.5f)<1e-5f && std::abs(resized.center.y+1.5f)<1e-5f &&
      resized.center.z==a.center.z && resized.size.z==a.size.z && resized.yaw_deg==a.yaw_deg,
      "Rotated corner resize moved the opposite corner or changed the vertical extent");
    const auto edge=ResizeRoi(a,{3,1,a.center.z},1,0);
    Check(std::abs(edge.size.x-5)<1e-5f && edge.size.y==a.size.y,"Edge resize changed the other axis");
    Check(ResizeRoi(a,{3,-100,a.center.z},1,0).size.x>=.049f,"Crossed resize handle inverted a box");
    const auto turned=RotateRoi(a,a.center+Vec3{0,2,0},a.center+Vec3{2,0,0});
    Check(std::abs(turned.yaw_deg)<1e-5f && turned.center.x==a.center.x && turned.size.x==a.size.x,
      "Rotation handle moved or resized the box");
    const auto snapped=RotateRoi(a,a.center+Vec3{0,2,0},a.center+Vec3{2,.1f,0},true);
    Check(std::abs(snapped.yaw_deg)<1e-5f,"Shift rotation did not snap to 15 degrees");
    const auto duplicate=DuplicateRoi(drag,edited);
    Check(!PreparedRoi(edited).Contains(duplicate.center),"duplicate overlaps source center");
    RoiHistory history; history.Record(edited); edited.boxes.push_back(duplicate);
    const auto after=edited;
    Check(history.Undo(edited) && edited.boxes.size()==1 && history.Redo(edited) && SameRoi(edited,after),
      "ROI undo/redo changed box data");
    history.Undo(edited);history.Record(edited);edited.boxes.clear();
    Check(!history.CanRedo(),"new ROI edit retained stale redo history");

    RoiBox mount{"mount",true,{0,0,0},{4,2,2},45,RoiBoxMode::Exclude};
    RoiSettings exclude{true,true,{mount}};
    Check(!PreparedRoi(exclude).Contains({0,0,0}) && PreparedRoi(exclude).Contains({5,0,0}),
      "standalone exclusion did not retain the surrounding cloud");
    for (auto p:RoiCorners(mount)) Check(!PreparedRoi(exclude).Contains(p),"exclusion boundary leaked");
    const auto exclusion_text=SerializeRoiSettings(exclude);
    Check(exclusion_text.find("roi_exclude_box=")!=std::string::npos &&
      SameRoi(ParseRoiSettings(exclusion_text),exclude),"exclusion type did not survive persistence");
    const auto exclusion_copy=DuplicateRoi(mount,exclude);
    Check(exclusion_copy.mode==RoiBoxMode::Exclude && exclusion_copy.name=="ROI 1", "duplicate lost ground exclusion type");
    Check(RoiFromDrag({0,0,0},{2,3,0},0,1,"draw",0,RoiBoxMode::Exclude).mode==RoiBoxMode::Exclude,
      "drawing lost exclusion type");
    auto renamed=exclude;renamed.boxes[0].name="renamed";renamed.show_boxes=false;
    Check(SameRoiExclusions(exclude,renamed),"cosmetic edit invalidated analysis");
    auto moved=exclude;moved.boxes[0].center.x+=.1f;
    Check(!SameRoiExclusions(exclude,moved),"moving an exclusion reused stale analysis");
    auto disabled=exclude;disabled.boxes[0].enabled=false;
    Check(!SameRoiExclusions(exclude,disabled) && PreparedRoi(disabled).Contains({}) &&
      SameRoiExclusions(disabled,RoiSettings{}),"disabled exclusion still removed points");
    disabled=exclude;disabled.enabled=false;
    Check(!SameRoiExclusions(exclude,disabled) && PreparedRoi(disabled).Contains({}),"master toggle ignored exclusion");
    auto mixed=exclude;mixed.boxes.insert(mixed.boxes.begin(),{"view",true,{0,0,0},{20,20,10},0});
    mixed.boxes.push_back({"overlap",true,{1,0,0},{3,3,2},0,RoiBoxMode::Exclude});
    const PreparedRoi visible(mixed), analysis(mixed,{},RoiFilterStage::Preprocessing);
    Check(!visible.Contains({0,0,0}) && visible.Contains({5,0,0}) && !visible.Contains({50,0,0}),
      "exclusions did not take precedence over include union");
    Check(!analysis.Contains({0,0,0}) && analysis.Contains({50,0,0}),"view crop limited the ground/detector input");
    auto keep_edit=mixed;keep_edit.boxes[0].size.x=1;
    Check(SameRoiExclusions(mixed,keep_edit),"include edit invalidated analysis");
    points={{0,0,0,4},{1,0,0,5},{5,0,0,6},{50,0,0,7}};
    analysis.Filter(points);
    Check(points.size()==2 && points[0].intensity==6 && points[1].intensity==7,"exclusion filter changed order or attributes");
    const PreparedRoi transformed_exclusion(mixed,transform);
    for (int x=-20;x<=20;++x) for (int y=-10;y<=10;++y) for (int z=-8;z<=8;++z) {
      const Vec3 p{x*.71f,y*.63f,z*.77f};
      Check(visible.Contains(p)==transformed_exclusion.Contains(transform.Transform(p)),"IMU moved sensor exclusion volume");
    }
    RoiHistory mask_history;mask_history.Record(exclude);exclude.boxes[0].mode=RoiBoxMode::Include;
    Check(mask_history.Undo(exclude) && exclude.boxes[0].mode==RoiBoxMode::Exclude && mask_history.Redo(exclude) &&
      exclude.boxes[0].mode==RoiBoxMode::Include,"undo/redo lost Keep/Exclude type");

    const RoiSettings original_mask{true,true,{mount}};
    auto draft=original_mask;draft.boxes[0].center={5,0,0};
    const PreparedRoi preview(ComposeRoiFilters(draft,original_mask));
    Check(!preview.Contains({0,0,0}) && preview.Contains({5,0,0}),"draft exclusion moved the active mask before Apply");
    const PreparedRoi committed(ComposeRoiFilters(draft,draft));
    Check(committed.Contains({0,0,0}) && !committed.Contains({5,0,0}),"Apply did not move the exclusion mask");
    draft.enabled=false;
    Check(!PreparedRoi(ComposeRoiFilters(draft,original_mask)).Contains({0,0,0}),"master edit bypassed Apply");
    Check(PreparedRoi(ComposeRoiFilters(draft,draft)).Contains({0,0,0}),"applied master-off retained exclusions");
    draft={true,true,{{"view",true,{0,0,0},{20,20,10},0}}};
    Check(!PreparedRoi(ComposeRoiFilters(draft,original_mask)).Contains({50,0,0}) &&
      !PreparedRoi(ComposeRoiFilters(draft,original_mask)).Contains({0,0,0}),"keep preview lost applied exclusions");
    draft.boxes[0].enabled=false;
    Check(PreparedRoi(ComposeRoiFilters(draft,original_mask)).RejectsAll(),"disabled keep preview changed legacy cropping");
    RoiSettings maximum_keep{true,true,{}}, maximum_masks{true,true,{}};
    for (std::size_t i=0;i<kMaxRoiBoxes;++i) {
      maximum_keep.boxes.push_back({"keep",true,{0,0,0},{20,20,20},0});
      maximum_masks.boxes.push_back(mount);
    }
    Check(PreparedRoi(ComposeRoiFilters(maximum_keep,maximum_masks)).Contains({5,0,0}),
      "maximum-size draft could not preview previously applied masks");
    Reject([&] {SerializeRoiSettings(ComposeRoiFilters(maximum_keep,maximum_masks));});

    auto ground_only=GroundExclusionRoi({false,true,{mount,{"legacy",false,{5,0,0},{2,2,2},0}}});
    Check(ground_only.enabled && ground_only.boxes[0].mode==RoiBoxMode::Exclude &&
      ground_only.boxes[1].mode==RoiBoxMode::Exclude && !ground_only.boxes[1].enabled,
      "legacy display switch or box kind changed ground ROI activation");
    const auto names=GroundExclusionRoi({false,true,{{"ROI 1"},{"Exclude 1"},{"custom mount"}}});
    Check(names.boxes[1].name=="ROI 2" && names.boxes[2].name=="custom mount","ground ROI migration renamed a custom box or duplicated a name");
    const PreparedRoi ground_mask(ground_only,{},RoiFilterStage::Preprocessing);
    points={{0,0,0,.2f},{5,0,0,.3f},{0,0,.5f,.4f},{50,0,0,.5f}};
    Check(ground_mask.CountRejected(points)==2,"wrong count of points skipped by ground ROI");
    auto skipped=ground_mask.ExtractRejected(points);
    Check(points.size()==2 && points[0].intensity==.3f && points[1].intensity==.5f &&
      skipped.size()==2 && skipped[0].intensity==.2f && skipped[1].intensity==.4f,
      "ground ROI partition lost point attributes or changed relative order");

    const auto serialized = SerializeRoiSettings(settings);
    const auto parsed = ParseRoiSettings("profile=default\n" + serialized);
    Check(SerializeRoiSettings(parsed) == serialized, "ROI serialization roundtrip failed");
    Check(SerializeRoiSettings(ParseRoiSettings("grid_enabled=1\n", settings)) == serialized, "absent ROI replaced fallback");
    Check(ParseRoiSettings("roi_enabled=1\nroi_count=0\n", settings).boxes.empty(), "empty section did not clear fallback");
    const RoiSettings display_settings{true,false,{a}};
    const auto display_text=SerializeRoiSettings(display_settings,"display_roi_");
    const auto combined=serialized+display_text;
    Check(SameRoi(ParseRoiSettings(combined),settings),"display section changed ground boxes");
    Check(SameRoi(ParseRoiSettings(combined,{},"display_roi_"),display_settings),
      "ground section changed display boxes");
    Check(ParseRoiSettings(serialized,{},"display_roi_").boxes.empty(),
      "old ground-only profile created display boxes");
    Check(SameRoi(ParseRoiSettings("display_roi_count=1\ndisplay_roi_box=bad\n"+serialized),settings),
      "bad display section corrupted ground settings");
    Reject([&] {ParseRoiSettings("display_roi_count=1\ndisplay_roi_box=bad\n"+serialized,{},"display_roi_");});
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
    Reject([&] {ParseRoiSettings("roi_count=1\nroi_exclude_box=1 \"mount\" 0 0 0 1 -1 1 0\n");});
    std::cout << "ROI geometry, union, rigid transforms and persistence passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
