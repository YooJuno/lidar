#include "lidar_viewer/object_tracker.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace lidar_viewer
{
namespace
{
using State=Eigen::Matrix<double,10,1>;
using Covariance=Eigen::Matrix<double,10,10>;
using Measurement=Eigen::Matrix<double,7,1>;
using Noise=Eigen::Matrix<double,7,7>;
constexpr double pi=3.14159265358979323846, gate=16.27, forbidden=1e6, unmatched=1.1;

bool Valid(const algorithms::Detection & d)
{
  const auto & b=d.oriented_bounds;
  if (!b.valid || d.class_id<0 || d.class_id>4 || !std::isfinite(d.confidence) ||
      d.confidence<0 || d.confidence>1) return false;
  for (float v:{b.center.x,b.center.y,b.center.z,b.axis_x.x,b.axis_x.y,b.half_length,b.half_width,b.min_z,b.max_z})
    if (!std::isfinite(v)) return false;
  return b.half_length>0 && b.half_length<=50 && b.half_width>0 && b.half_width<=50 &&
    b.max_z>b.min_z && b.max_z-b.min_z<=30 && std::abs(b.center.x)<1e6 &&
    std::abs(b.center.y)<1e6 && std::abs(b.center.z)<1e6 && std::hypot(b.axis_x.x,b.axis_x.y)>.5;
}

Measurement Measure(const algorithms::Detection & d)
{
  const auto & b=d.oriented_bounds;
  Measurement z;z<<b.center.x,b.center.y,b.center.z,std::atan2(b.axis_x.y,b.axis_x.x),
    b.half_length*2,b.half_width*2,b.max_z-b.min_z;
  return z;
}

Noise MeasurementNoise(const State & x)
{
  Noise r=Noise::Zero();
  const double sigma=.20+.004*std::hypot(x[0],x[1]);
  r.diagonal()<<sigma*sigma,sigma*sigma,.09,.09,.16,.16,.16;
  return r;
}

algorithms::Detection Box(const State & x,int class_id,float score)
{
  algorithms::Detection result;result.class_id=class_id;result.confidence=score;
  auto & b=result.oriented_bounds;
  b.center={static_cast<float>(x[0]),static_cast<float>(x[1]),static_cast<float>(x[2])};
  b.axis_x={static_cast<float>(std::cos(x[3])),static_cast<float>(std::sin(x[3])),0};
  b.axis_y={-b.axis_x.y,b.axis_x.x,0};
  b.half_length=static_cast<float>(std::max(.01,x[4])*.5);
  b.half_width=static_cast<float>(std::max(.01,x[5])*.5);
  b.min_z=b.center.z-static_cast<float>(std::max(.01,x[6])*.5);
  b.max_z=b.center.z+static_cast<float>(std::max(.01,x[6])*.5);b.valid=true;
  const float ex=std::abs(b.axis_x.x)*b.half_length+std::abs(b.axis_y.x)*b.half_width;
  const float ey=std::abs(b.axis_x.y)*b.half_length+std::abs(b.axis_y.y)*b.half_width;
  result.bounds={{b.center.x-ex,b.center.y-ey,b.min_z},{b.center.x+ex,b.center.y+ey,b.max_z}};
  return result;
}

// Rectangular shortest-augmenting-path Hungarian assignment. Dummy columns
// allow each track to remain unmatched instead of forcing an invalid link.
std::vector<int> Assign(const std::vector<double> & costs,std::size_t rows,std::size_t real_columns)
{
  if (!rows || !real_columns) return std::vector<int>(rows,-1);
  const auto columns=real_columns+rows;
  std::vector<double> u(rows+1),v(columns+1),minimum(columns+1);
  std::vector<int> p(columns+1),way(columns+1),result(rows,-1);
  std::vector<unsigned char> used(columns+1);
  for (std::size_t i=1;i<=rows;++i) {
    p[0]=static_cast<int>(i);std::fill(minimum.begin(),minimum.end(),forbidden*2);
    std::fill(used.begin(),used.end(),0);std::size_t column=0;
    do {
      used[column]=1;const auto row=static_cast<std::size_t>(p[column]);double delta=forbidden*2;std::size_t next=0;
      for (std::size_t j=1;j<=columns;++j) if (!used[j]) {
        const double cost=(j<=real_columns ? costs[(row-1)*real_columns+j-1] : unmatched)-u[row]-v[j];
        if (cost<minimum[j]) {minimum[j]=cost;way[j]=static_cast<int>(column);}
        if (minimum[j]<delta) {delta=minimum[j];next=j;}
      }
      for (std::size_t j=0;j<=columns;++j) {
        if (used[j]) {u[static_cast<std::size_t>(p[j])]+=delta;v[j]-=delta;}
        else minimum[j]-=delta;
      }
      column=next;
    } while (p[column]!=0);
    do {const auto previous=static_cast<std::size_t>(way[column]);p[column]=p[previous];column=previous;} while(column);
  }
  for (std::size_t j=1;j<=real_columns;++j) if (p[j]>0 &&
      costs[(static_cast<std::size_t>(p[j])-1)*real_columns+j-1]<unmatched)
    result[static_cast<std::size_t>(p[j])-1]=static_cast<int>(j-1);
  return result;
}

struct TrackState
{
  State x=State::Zero();Covariance covariance=Covariance::Identity();
  std::uint64_t id{0};int class_id{-1},streak{1};bool confirmed{false};
  float score{0};std::int64_t last_measurement{0};
  std::optional<std::size_t> measurement;
  std::vector<Vec3> trail;
};

void Predict(TrackState & track,double dt)
{
  Covariance f=Covariance::Identity(),q=Covariance::Zero();
  for (int i=0;i<3;++i) {
    f(i,i+7)=dt;const double variance=i==2 ? 4. : 16.;
    q(i,i)=variance*dt*dt*dt*dt*.25;
    q(i,i+7)=q(i+7,i)=variance*dt*dt*dt*.5;q(i+7,i+7)=variance*dt*dt;
  }
  q(3,3)=.09*dt;for(int i=4;i<7;++i) q(i,i)=.01*dt;
  track.x=f*track.x;track.covariance=f*track.covariance*f.transpose()+q;track.measurement.reset();
}

void Correct(TrackState & track,const algorithms::Detection & detection,std::size_t index,std::int64_t stamp)
{
  auto z=Measure(detection);
  // A 3D box's heading is ambiguous by pi. Correct the nearest equivalent
  // orientation, including wraparound at +/-pi, before the linear update.
  z[3]=track.x[3]+std::remainder(z[3]-track.x[3],pi);
  const auto r=MeasurementNoise(track.x);
  const Noise s=track.covariance.topLeftCorner<7,7>()+r;
  const Eigen::LDLT<Noise> solve(s);
  const Eigen::Matrix<double,10,7> k=solve.solve(track.covariance.leftCols<7>().transpose()).transpose();
  track.x+=k*(z-track.x.head<7>());track.x[3]=std::remainder(track.x[3],2*pi);
  Covariance residual=Covariance::Identity();residual.leftCols<7>()-=k;
  track.covariance=residual*track.covariance*residual.transpose()+k*r*k.transpose();
  track.covariance=(track.covariance+track.covariance.transpose()).eval()*.5;
  track.score=detection.confidence;track.last_measurement=stamp;track.measurement=index;++track.streak;
}
}  // namespace

struct ObjectTracker::Impl
{
  std::vector<TrackState> states;
  std::vector<ObjectTrack> output;
  std::optional<TrackingFrame> last;
  std::uint64_t next_id{1};
  TrackingStats stats;
};

ObjectTracker::ObjectTracker():impl_(std::make_unique<Impl>()) {}
ObjectTracker::~ObjectTracker()=default;
void ObjectTracker::Reset() {impl_->states.clear();impl_->output.clear();impl_->last.reset();impl_->stats={};}
const std::vector<ObjectTrack> & ObjectTracker::Tracks() const {return impl_->output;}
const TrackingStats & ObjectTracker::Stats() const {return impl_->stats;}

const std::vector<ObjectTrack> & ObjectTracker::Update(const std::vector<algorithms::Detection> & detections,
  const TrackingFrame & frame,const TrackingParams & params,float min_confidence)
{
  if (frame.timestamp_ns<0 || !std::isfinite(min_confidence) || min_confidence<.1f || min_confidence>.99f ||
      !std::isfinite(params.max_missed_s) || params.max_missed_s<.05f || params.max_missed_s>10 ||
      params.confirmation_hits<1 || params.confirmation_hits>20) throw std::invalid_argument("Invalid tracking frame or settings");
  auto & core=*impl_;
  if (core.last && core.last->session==frame.session && core.last->sequence==frame.sequence) return core.output;
  const auto started=std::chrono::steady_clock::now();
  double dt=0;std::string reason;
  if (core.last) {
    dt=static_cast<double>((static_cast<long double>(frame.timestamp_ns)-core.last->timestamp_ns)*1e-9L);
    if (frame.session!=core.last->session) reason="source changed";
    else if (frame.sequence<core.last->sequence || (!frame.allow_skipped_frames && frame.sequence-core.last->sequence!=1)) reason="seek or loop";
    else if (dt<=0 || dt>std::max(2.,static_cast<double>(params.max_missed_s)*2)) reason="timestamp discontinuity";
  }
  if (!reason.empty()) {Reset();dt=0;}
  core.stats={};core.stats.reset_reason=reason;core.output.clear();
  auto age=[&](const TrackState & t) {return static_cast<double>((static_cast<long double>(frame.timestamp_ns)-t.last_measurement)*1e-9L);};
  core.states.erase(std::remove_if(core.states.begin(),core.states.end(),[&](const auto & t) {
    return age(t)>(t.confirmed ? params.max_missed_s : std::min(.25f,params.max_missed_s));
  }),core.states.end());
  for (auto & track:core.states) Predict(track,dt);

  // Keep the strongest bounded set without allocating a buffer proportional to
  // an arbitrarily large input vector. Equal scores retain their input order.
  using Candidate=std::pair<float,std::size_t>;
  std::priority_queue<Candidate,std::vector<Candidate>,std::greater<Candidate>> strongest;
  std::size_t eligible=0;
  for (std::size_t i=0;i<detections.size();++i) if (Valid(detections[i]) && detections[i].confidence>=min_confidence) {
    ++eligible;const Candidate candidate{detections[i].confidence,i};
    if (strongest.size()<kMaxMeasurements) strongest.push(candidate);
    else if (candidate.first>strongest.top().first) {strongest.pop();strongest.push(candidate);}
  }
  core.stats.capacity_dropped=eligible-strongest.size();
  std::vector<std::size_t> indices;indices.reserve(strongest.size());
  while(!strongest.empty()) {indices.push_back(strongest.top().second);strongest.pop();}
  std::sort(indices.begin(),indices.end(),[&](auto a,auto b) {
    return detections[a].confidence!=detections[b].confidence ? detections[a].confidence>detections[b].confidence : a<b;
  });
  std::vector<double> costs(core.states.size()*indices.size(),forbidden);
  for (std::size_t i=0;i<core.states.size();++i) {
    const auto & track=core.states[i];
    const Eigen::Matrix3d innovation=track.covariance.topLeftCorner<3,3>()+MeasurementNoise(track.x).topLeftCorner<3,3>();
    const Eigen::Matrix3d inverse=innovation.ldlt().solve(Eigen::Matrix3d::Identity());
    for (std::size_t j=0;j<indices.size();++j) {
      const auto & detection=detections[indices[j]];
      if (track.class_id!=detection.class_id) continue;
      const auto z=Measure(detection);const Eigen::Vector3d delta=z.head<3>()-track.x.head<3>();
      const double distance=delta.norm(),mahal=delta.dot(inverse*delta);
      const double max_distance=track.class_id==algorithms::DetectionClassPerson ? 4. : 12.;
      if (!std::isfinite(mahal) || mahal>gate || distance>max_distance) continue;
      double size_cost=0;bool size_ok=true;
      for(int axis=4;axis<7;++axis) {
        const double ratio=z[axis]/std::max(.01,track.x[axis]);
        if (ratio<.25 || ratio>4) size_ok=false;
        size_cost+=std::abs(std::log(ratio));
      }
      if (!size_ok) continue;
      const double heading=track.class_id==algorithms::DetectionClassPerson ? 0. : std::abs(std::remainder(z[3]-track.x[3],pi))/(pi*.5);
      costs[i*indices.size()+j]=.75*mahal/gate+.15*std::min(1.,size_cost/3)+.05*heading+.05*(1-detection.confidence);
    }
  }
  const auto assignments=Assign(costs,core.states.size(),indices.size());
  std::vector<unsigned char> matched(indices.size());
  for (std::size_t i=0;i<core.states.size();++i) {
    auto & track=core.states[i];const auto j=assignments[i];
    if (j<0) {track.streak=0;continue;}
    const auto index=indices[static_cast<std::size_t>(j)];
    Correct(track,detections[index],index,frame.timestamp_ns);matched[static_cast<std::size_t>(j)]=1;
    track.confirmed=track.confirmed || track.streak>=params.confirmation_hits;
  }
  for (std::size_t j=0;j<indices.size();++j) if (!matched[j]) {
    if (core.states.size()>=kMaxTracks) {++core.stats.capacity_dropped;continue;}
    if (core.next_id==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Tracking ID exhausted");
    const auto index=indices[j];const auto & detection=detections[index];TrackState track;
    track.x.head<7>()=Measure(detection);track.covariance.topLeftCorner<7,7>()=MeasurementNoise(track.x);
    track.covariance.bottomRightCorner<3,3>()*=100;
    track.id=core.next_id++;track.class_id=detection.class_id;track.score=detection.confidence;
    track.last_measurement=frame.timestamp_ns;track.measurement=index;track.confirmed=params.confirmation_hits==1;
    track.trail.reserve(kTrailLength);core.states.push_back(std::move(track));
  }
  core.states.erase(std::remove_if(core.states.begin(),core.states.end(),[](const auto & track) {
    return !track.x.allFinite() || !track.covariance.allFinite();
  }),core.states.end());
  for (auto & track:core.states) {
    if (!track.confirmed && !track.measurement) continue;
    const Vec3 center{static_cast<float>(track.x[0]),static_cast<float>(track.x[1]),static_cast<float>(track.x[2])};
    if (track.trail.size()==kTrailLength) track.trail.erase(track.trail.begin());
    track.trail.push_back(center);
    ObjectTrack output;output.detection=Box(track.x,track.class_id,track.score);output.id=track.id;
    output.velocity={static_cast<float>(track.x[7]),static_cast<float>(track.x[8]),static_cast<float>(track.x[9])};
    output.confirmed=track.confirmed;output.predicted=!track.measurement;output.missed_s=static_cast<float>(age(track));
    output.measurement_index=track.measurement;output.trail=track.trail;core.output.push_back(std::move(output));
    if (track.confirmed) ++core.stats.confirmed;else ++core.stats.tentative;
    if (!track.measurement) ++core.stats.coasting;
  }
  core.last=frame;
  core.stats.processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  return core.output;
}
}  // namespace lidar_viewer
