#include "lidar_viewer/object_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lidar_viewer;
namespace {
void Check(bool value,const char * message) {if (!value) throw std::runtime_error(message);}
algorithms::Detection D(float x,float y=0,int cls=3,float yaw=0,float score=.8f) {
  const int model_class=cls==3 ? 4 : cls==4 ? 3 : cls==2 ? 1 : cls==1 ? 2 : 0;
  const float box[]{x,y,.9f,cls==3 ? .7f : 4.f,cls==3 ? .6f : 2.f,1.8f,yaw,score,static_cast<float>(model_class)};
  return algorithms::DecodeCenterPointBox(box,0);
}
TrackingFrame F(std::uint64_t sequence,double time=-1,std::uint64_t session=1,bool live=false) {
  return {session,sequence,static_cast<std::int64_t>((time<0 ? 1+sequence*.1 : time)*1e9),live};
}
const ObjectTrack & Id(const std::vector<ObjectTrack> & tracks,std::uint64_t id) {
  const auto found=std::find_if(tracks.begin(),tracks.end(),[&](const auto & t){return t.id==id;});
  if (found==tracks.end()) throw std::runtime_error("Expected track ID is missing");return *found;
}
}
int main() {
  try {
    TrackingParams params;ObjectTracker tracker;
    const auto initial=tracker.Update({D(0)},F(0),params,.5f).front().id;
    Check(!tracker.Tracks().front().confirmed,"One observation was declared confirmed");
    for (std::uint64_t i=1;i<20;++i) tracker.Update({D(static_cast<float>(i)*.1f)},F(i),params,.5f);
    const auto moving=Id(tracker.Tracks(),initial);
    Check(moving.confirmed && std::abs(moving.velocity.x-1)<.15f && std::abs(moving.velocity.y)<.01f,
      "Constant-velocity track changed ID or used a wrong time interval");
    const auto history=moving.trail.size();
    for (int i=0;i<50;++i) tracker.Update({D(100)},F(19),params,.5f);
    Check(Id(tracker.Tracks(),initial).trail.size()==history && Id(tracker.Tracks(),initial).detection.oriented_bounds.center.x<2,
      "Repeated rendering aged, moved or duplicated a track");
    tracker.Update({},F(20),params,.5f);const auto predicted=Id(tracker.Tracks(),initial);
    Check(predicted.predicted && predicted.detection.oriented_bounds.center.x>moving.detection.oriented_bounds.center.x,
      "Missing detection did not coast with the same ID");
    tracker.Update({},F(21),params,.5f);tracker.Update({D(2.2f)},F(22),params,.5f);
    Check(Id(tracker.Tracks(),initial).confirmed && !Id(tracker.Tracks(),initial).predicted,
      "Short occlusion created a new object ID");
    tracker.Update({D(3.1f)},F(23,4.1),params,.5f);
    Check(tracker.Tracks().size()==1 && tracker.Tracks().front().id!=initial,"Expired track was revived");

    ObjectTracker crossing;auto objects=crossing.Update({D(-3),D(3)},F(0),params,.5f);
    const auto rightward=objects[0].id,leftward=objects[1].id;
    for (std::uint64_t i=1;i<=40;++i) {
      const float position=-3+static_cast<float>(i)*.15f;
      std::vector<algorithms::Detection> detections{D(position),D(-position)};
      if (i%2) std::reverse(detections.begin(),detections.end());
      crossing.Update(detections,F(i),params,.5f);
    }
    Check(Id(crossing.Tracks(),rightward).velocity.x>1.2f && Id(crossing.Tracks(),rightward).detection.oriented_bounds.center.x>2.5f &&
      Id(crossing.Tracks(),leftward).velocity.x< -1.2f,"Crossing or reordered detections swapped IDs");
    Check(Id(crossing.Tracks(),rightward).trail.size()<=ObjectTracker::kTrailLength,"Trajectory history grew without bound");

    ObjectTracker typed;const auto person=typed.Update({D(0,0,3)},F(0),params,.5f).front().id;
    typed.Update({D(0,0,0)},F(1),params,.5f);
    Check(typed.Tracks().size()==1 && typed.Tracks().front().id!=person && typed.Tracks().front().detection.class_id==0,
      "Different semantic classes shared an ID");
    ObjectTracker brief;brief.Update({D(0)},F(0),params,.5f);brief.Update({},F(1),params,.5f);
    Check(brief.Tracks().empty(),"A single-frame false detection was displayed as a prediction");
    ObjectTracker turn;const auto car=turn.Update({D(0,0,0,3.13f)},F(0),params,.5f).front().id;
    for (std::uint64_t i=1;i<20;++i) turn.Update({D(.1f*static_cast<float>(i),0,0,i%2 ? -3.13f : .01f)},F(i),params,.5f);
    const auto & oriented=Id(turn.Tracks(),car).detection.oriented_bounds;
    Check(std::abs(oriented.axis_x.y)<.05f && std::isfinite(oriented.center.x),"Yaw wrap or pi ambiguity rotated a stable box");

    ObjectTracker discontinuity;auto id=discontinuity.Update({D(0)},F(10),params,.5f).front().id;
    auto next=discontinuity.Update({D(0)},F(5),params,.5f).front().id;
    Check(next>id && discontinuity.Stats().reset_reason=="seek or loop","Backward seek reused old IDs");id=next;
    next=discontinuity.Update({D(0)},F(20),params,.5f).front().id;
    Check(next>id,"Forward seek reused old IDs");id=next;
    next=discontinuity.Update({D(0)},F(0,-1,2),params,.5f).front().id;
    Check(next>id,"New source reused old IDs");id=next;
    discontinuity.Reset();next=discontinuity.Update({D(0)},F(0),params,.5f).front().id;
    Check(next>id,"Explicit reset recycled an ID");
    ObjectTracker live;id=live.Update({D(0)},F(1,1,1,true),params,.5f).front().id;
    live.Update({D(.2f)},F(4,1.2,1,true),params,.5f);
    Check(live.Tracks().front().id==id,"Latest-frame live subscription reset on a skipped sequence");
    live.Update({D(.4f)},F(7,1.1,1,true),params,.5f);
    Check(live.Tracks().front().id!=id,"Reversed sensor time retained old velocity");

    ObjectTracker noise;double raw_error=0,filtered_error=0;
    for (std::uint64_t i=0;i<150;++i) {
      const float truth=static_cast<float>(i)*.05f,measurement=truth+.22f*std::sin(static_cast<float>(i)*2.1f);
      noise.Update({D(measurement)},F(i),params,.5f);
      if (i>20) {raw_error+=(measurement-truth)*(measurement-truth);const auto e=noise.Tracks().front().detection.oriented_bounds.center.x-truth;filtered_error+=e*e;}
    }
    Check(filtered_error<raw_error*.8,"Kalman update did not reduce measurement jitter");
    ObjectTracker bounded;std::vector<algorithms::Detection> many;
    for (int i=0;i<700;++i) many.push_back(D(static_cast<float>(i)*3));
    bounded.Update(many,F(0),params,.5f);
    Check(bounded.Tracks().size()==ObjectTracker::kMaxTracks && bounded.Stats().capacity_dropped==700-ObjectTracker::kMaxTracks,
      "Tracking capacity was not bounded and reported");
    auto invalid=D(0);invalid.oriented_bounds.center.x=std::numeric_limits<float>::quiet_NaN();
    ObjectTracker robust;robust.Update({invalid,D(0,0,3,0,.2f)},F(0),params,.5f);
    Check(robust.Tracks().empty(),"Invalid or below-threshold boxes created tracks");
    params.max_missed_s=std::numeric_limits<float>::infinity();bool rejected=false;
    try {robust.Update({},F(1),params,.5f);} catch (const std::invalid_argument &) {rejected=true;}
    Check(rejected,"Invalid tracking parameters were accepted");
    std::cout<<"Tracking: identity, crossing, class gates, sensor time, occlusion, yaw, reset, smoothing and bounds passed\n";
    return 0;
  } catch (const std::exception & error) {std::cerr<<error.what()<<'\n';return 1;}
}
