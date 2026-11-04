#include "lidar_viewer/roi_export.hpp"
#include "lidar_viewer/settings_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace lidar_viewer
{
namespace
{
struct Canceled {};
void CheckCancel(const RoiExportProgress & progress)
{
  if (progress.cancel.load()) throw Canceled{};
}
std::string Json(const std::string & value)
{
  std::ostringstream out;out<<'"';
  constexpr char hex[]="0123456789abcdef";
  for (const unsigned char c:value) {
    if (c=='"' || c=='\\') out<<'\\'<<static_cast<char>(c);
    else if (c<32) out<<"\\u00"<<hex[c>>4]<<hex[c&15];
    else out<<static_cast<char>(c);
  }
  return out.str()+'"';
}
std::ostringstream JsonStream()
{
  std::ostringstream out;out.imbue(std::locale::classic());
  out<<std::setprecision(std::numeric_limits<double>::max_digits10);
  return out;
}
void Vector(std::ostream & out,Vec3 p) {out<<'['<<p.x<<','<<p.y<<','<<p.z<<']';}
void Matrix(std::ostream & out,const RoiCoordinateFrame & frame)
{
  out<<'[';
  for (std::size_t row=0;row<3;++row) {
    for (std::size_t col=0;col<3;++col) out<<frame.rotation[row*3+col]<<',';
    out<<(row==0 ? frame.translation.x : row==1 ? frame.translation.y : frame.translation.z)<<',';
  }
  out<<"0,0,0,1]";
}
RoiCoordinateFrame Alignment(const DisplayAlignment & c,float ground_offset)
{
  const float a=c.yaw_deg*kPi/180.f,b=c.pitch_deg*kPi/180.f,d=c.roll_deg*kPi/180.f;
  const float cy=std::cos(a),sy=std::sin(a),cp=std::cos(b),sp=std::sin(b),cr=std::cos(d),sr=std::sin(d);
  RoiCoordinateFrame frame;
  frame.rotation={cp*cy,sr*sp*cy-cr*sy,cr*sp*cy+sr*sy,
    cp*sy,sr*sp*sy+cr*cy,cr*sp*sy-sr*cy,-sp,sr*cp,cr*cp};
  frame.translation=frame.Rotate({0,0,-ground_offset})+Vec3{0,0,c.z_m};
  return frame;
}
void WriteBytes(std::ofstream & out,const char * data,std::size_t size,const RoiExportProgress & progress)
{
  constexpr std::size_t chunk=1024*1024;
  while (size) {
    CheckCancel(progress);
    const auto n=std::min(size,chunk);
    out.write(data,static_cast<std::streamsize>(n));data+=n;size-=n;
  }
}
void WritePoints(std::ofstream & out,const std::vector<KittiPoint> & points,const RoiExportProgress & progress)
{
  static_assert(sizeof(KittiPoint)==4*sizeof(float),"XYZI layout must be packed");
  static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE float32 required");
  const std::uint16_t endian=1;
  if (*reinterpret_cast<const unsigned char *>(&endian)==1) {
    WriteBytes(out,reinterpret_cast<const char *>(points.data()),points.size()*sizeof(KittiPoint),progress);
  } else {
    std::array<char,65536> buffer{};
    const auto * input=reinterpret_cast<const char *>(points.data());
    for (std::size_t offset=0;offset<points.size()*sizeof(KittiPoint);) {
      const auto size=std::min(buffer.size(),points.size()*sizeof(KittiPoint)-offset);
      for (std::size_t i=0;i<size;i+=4) for (std::size_t j=0;j<4;++j) buffer[i+j]=input[offset+i+3-j];
      WriteBytes(out,buffer.data(),size,progress);offset+=size;
    }
  }
}
std::ofstream BinaryFile(const fs::path & path)
{
  std::ofstream out;out.exceptions(std::ios::badbit|std::ios::failbit);
  out.open(path,std::ios::binary);out.imbue(std::locale::classic());return out;
}
void WriteNpy(const fs::path & path,const std::vector<KittiPoint> & points,const RoiExportProgress & progress)
{
  auto out=BinaryFile(path);
  std::string header="{'descr': '<f4', 'fortran_order': False, 'shape': ("+std::to_string(points.size())+", 4), }";
  header.append((64-(10+header.size()+1)%64)%64,' ');header+='\n';
  const unsigned char magic[]={0x93,'N','U','M','P','Y',1,0,
    static_cast<unsigned char>(header.size()&255),static_cast<unsigned char>(header.size()>>8)};
  out.write(reinterpret_cast<const char *>(magic),sizeof(magic));out<<header;
  WritePoints(out,points,progress);out.close();
}
void WritePcd(const fs::path & path,const std::vector<KittiPoint> & points,const RoiExportProgress & progress)
{
  auto out=BinaryFile(path);
  out<<"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
    "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH "<<points.size()<<
    "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "<<points.size()<<"\nDATA binary\n";
  WritePoints(out,points,progress);out.close();
}
void WriteRegions(std::ostream & out,const RoiSettings & regions)
{
  out<<'[';bool first=true;
  for (const auto & box:regions.boxes) {
    if (!box.enabled) continue;
    if (!first) out<<',';
    first=false;out<<"{\"name\":"<<Json(box.name)<<",\"center_m\":";Vector(out,box.center);
    out<<",\"size_m\":";Vector(out,box.size);out<<",\"yaw_deg\":"<<box.yaw_deg<<'}';
  }
  out<<']';
}
void Validate(const RoiExportRequest & request)
{
  if (request.destination.empty() || request.destination.filename().empty() ||
      request.destination.filename()=="." || request.destination.filename()=="..")
    throw std::runtime_error("Choose a new output directory");
  if (!fs::is_directory(request.destination.parent_path().empty() ? fs::path(".") : request.destination.parent_path()))
    throw std::runtime_error("The export parent folder does not exist");
  if ((request.live_snapshot.has_value() && !request.sources.empty()) ||
      (!request.live_snapshot && request.sources.empty()))
    throw std::runtime_error("Choose a current frame or frame range to export");
  if (request.regions.boxes.size()>kMaxRoiBoxes) throw std::runtime_error("Too many export regions");
  bool active=false;
  for (const auto & box:request.regions.boxes) {
    ValidateRoiBox(box);
    if (box.enabled) active=true;
  }
  if (!active) throw std::runtime_error("Check at least one display ROI to export");
  for (const float v:{request.calibration.yaw_deg,request.calibration.pitch_deg,request.calibration.roll_deg,
      request.calibration.z_m,request.global_ground_offset,request.imu_mount.roll_deg,
      request.imu_mount.pitch_deg,request.imu_mount.yaw_deg})
    if (!std::isfinite(v)) throw std::runtime_error("Export alignment must be finite");
}
}  // namespace

RoiExportResult ExportRoiDataset(const RoiExportRequest & request,const std::shared_ptr<RoiExportProgress> & progress)
{
  if (!progress) throw std::invalid_argument("Export requires a progress state");
  Validate(request);
  if (progress->cancel.load()) return {request.destination,0,0,0,true};
  RoiSettings regions=request.regions;regions.enabled=true;
  for (auto & box:regions.boxes) box.mode=RoiBoxMode::Include;
  const PreparedRoi mask(regions);
  RoiExportResult result;result.destination=request.destination;
  // Exclusive creation protects previous exports and arbitrary user folders.
  if (!fs::create_directory(request.destination)) throw std::runtime_error("Export directory already exists; choose a new name");
  try {
    for (const auto & name:{"points","pointcloud","metadata"}) fs::create_directory(request.destination/name);
    settings::AtomicWriteText(request.destination/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"version\":1,\"status\":\"exporting\"}\n");
    const std::size_t count=request.live_snapshot ? 1 : request.sources.size();
    auto samples=JsonStream(),empty=JsonStream();bool first_sample=true,first_empty=true;
    for (std::size_t i=0;i<count;++i) {
      CheckCancel(*progress);
      FrameData raw=request.live_snapshot ? *request.live_snapshot :
        LoadFrame(request.sources[i].handle,FrameIntensity::Original,FrameReadIsolation::Independent);
      if (raw.source_intensities && raw.source_intensities->size()!=raw.points.size())
        throw std::runtime_error("Original intensity buffer does not match the source frame");
      const auto source_count=raw.points.size();
      const auto leveling=request.imu_level_enabled ? ComputeImuLeveling(raw.imu,request.imu_mount) : ImuLevelingResult{};
      const auto resolved=ResolveDisplayAlignment(request.calibration,request.imu_level_enabled,leveling);
      const auto alignment=Alignment(resolved.effective,request.global_ground_offset);
      const auto output_transform=request.aligned_coordinates ? alignment : RoiCoordinateFrame{};
      std::size_t retained=0,missing_returns=0;
      for (std::size_t j=0;j<raw.points.size();++j) {
        if (j%65536==0) CheckCancel(*progress);
        auto p=raw.points[j];
        if (raw.source_intensities) p.intensity=(*raw.source_intensities)[j];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.x==0 && p.y==0 && p.z==0) {++missing_returns;continue;}
        if (!mask.Contains({p.x,p.y,p.z})) continue;
        if (request.aligned_coordinates) {
          const auto transformed=output_transform.Transform({p.x,p.y,p.z});
          p.x=transformed.x;p.y=transformed.y;p.z=transformed.z;
        }
        if (!std::isfinite(p.intensity)) p.intensity=0;
        raw.points[retained++]=p;
      }
      raw.points.resize(retained);
      const auto frame_index=request.live_snapshot ? request.live_sequence : request.sources[i].frame_index;
      if (raw.points.empty()) {
        if (!first_empty) empty<<',';
        first_empty=false;empty<<frame_index;++result.empty_frames;progress->empty.store(result.empty_frames);
        progress->completed.store(i+1);continue;
      }
      std::ostringstream id;id.imbue(std::locale::classic());id<<std::setw(8)<<std::setfill('0')<<i;
      const auto stem=id.str();
      WriteNpy(request.destination/"points"/(stem+".npy"),raw.points,*progress);
      WritePcd(request.destination/"pointcloud"/(stem+".pcd"),raw.points,*progress);
      auto metadata=JsonStream();
      metadata<<"{\"schema\":\"lidar_studio.roi_sample\",\"version\":1,\"id\":"<<Json(stem)
        <<",\"point_count\":"<<raw.points.size()<<",\"source_point_count\":"<<source_count
        <<",\"missing_returns_removed\":"<<missing_returns<<",\"coordinate_space\":"<<Json(request.aligned_coordinates ? "aligned" : "sensor")
        <<",\"units\":\"metres\",\"features\":[\"x\",\"y\",\"z\",\"intensity\"],\"dtype\":\"float32\",\"source\":{\"kind\":";
      if (request.live_snapshot) metadata<<"\"live\",\"sequence\":"<<frame_index<<",\"topic\":"<<Json(request.live_topic);
      else {
        const auto & source=request.sources[i];const auto & handle=source.handle;
        metadata<<Json(handle.format==InputFormat::Ros2Bag ? "rosbag" : handle.format==InputFormat::KittiXyzi ? "bin" : "ouster_legacy")
          <<",\"path\":"<<Json(handle.path.u8string())<<",\"topic\":"<<Json(handle.bag_topic)
          <<",\"frame_index\":"<<source.frame_index<<",\"bag_duplicate_index\":"<<handle.bag_duplicate_index
          <<",\"payload_offset\":"<<handle.payload_offset;
      }
      metadata<<",\"frame_id\":"<<Json(raw.frame_id)<<",\"timestamp_ns\":";
      if (raw.timestamp_ns) metadata<<Json(std::to_string(*raw.timestamp_ns));else metadata<<"null";
      metadata<<"},\"sensor_to_output_row_major\":";Matrix(metadata,output_transform);
      metadata<<",\"sensor_to_aligned_row_major\":";Matrix(metadata,alignment);
      metadata<<",\"imu\":{\"requested\":"<<(request.imu_level_enabled ? "true" : "false")
        <<",\"applied\":"<<(resolved.imu_active ? "true" : "false")<<",\"axes_corrected\":"<<(leveling.axes_corrected ? "true" : "false")
        <<",\"status\":"<<Json(request.imu_level_enabled ? leveling.status : "Off");
      if (raw.imu) {
        metadata<<",\"topic\":"<<Json(raw.imu->topic)<<",\"imu_frame\":"<<Json(raw.imu->imu_frame)
          <<",\"lidar_frame\":"<<Json(raw.imu->lidar_frame)<<",\"skew_ms\":";
        if (std::isfinite(raw.imu->skew_ms)) metadata<<raw.imu->skew_ms;else metadata<<"null";
        if (raw.imu->up && std::isfinite(raw.imu->up->x) && std::isfinite(raw.imu->up->y) && std::isfinite(raw.imu->up->z))
          metadata<<",\"up\":["<<raw.imu->up->x<<','<<raw.imu->up->y<<','<<raw.imu->up->z<<']';
      }
      metadata<<"},\"effective_alignment_deg\":{\"yaw\":"<<resolved.effective.yaw_deg
        <<",\"pitch\":"<<resolved.effective.pitch_deg<<",\"roll\":"<<resolved.effective.roll_deg
        <<"},\"ground_offset_m\":"<<request.global_ground_offset<<",\"height_offset_m\":"<<resolved.effective.z_m
        <<",\"roi_sensor_frame\":";WriteRegions(metadata,regions);
      metadata<<",\"annotation_status\":\"unlabeled\",\"ground_filter_applied\":false,\"ground_exclusions_applied\":false}\n";
      settings::AtomicWriteText(request.destination/"metadata"/(stem+".json"),metadata.str());
      if (!first_sample) samples<<',';
      first_sample=false;samples<<"{\"id\":"<<Json(stem)<<",\"source_frame_index\":"<<frame_index
        <<",\"points\":"<<Json("points/"+stem+".npy")<<",\"pcd\":"<<Json("pointcloud/"+stem+".pcd")
        <<",\"metadata\":"<<Json("metadata/"+stem+".json")<<",\"point_count\":"<<raw.points.size()<<'}';
      ++result.frames;result.points+=raw.points.size();
      progress->points.store(result.points);progress->exported.store(result.frames);progress->completed.store(i+1);
    }
    CheckCancel(*progress);
    if (!result.frames) throw std::runtime_error("No points inside the checked display regions in the selected frames");
    auto manifest=JsonStream();
    manifest<<"{\"schema\":\"lidar_studio.roi_dataset\",\"version\":1,\"status\":\"complete\",\"annotation_status\":\"unlabeled\","
      "\"features\":[\"x\",\"y\",\"z\",\"intensity\"],\"dtype\":\"float32\",\"units\":\"metres\","
      "\"intensity\":\"original_decoded_values_nonfinite_replaced_with_zero\",\"point_order\":\"source_order\","
      "\"zero_xyz_returns\":\"removed\",\"coordinate_space\":"<<Json(request.aligned_coordinates ? "aligned" : "sensor")
      <<",\"profile\":"<<Json(request.profile)<<",\"frames_requested\":"<<count<<",\"frames_exported\":"<<result.frames
      <<",\"points_exported\":"<<result.points<<",\"empty_source_frames\":["<<empty.str()<<"],\"manual_alignment\":{"
      <<"\"yaw_deg\":"<<request.calibration.yaw_deg<<",\"pitch_deg\":"<<request.calibration.pitch_deg
      <<",\"roll_deg\":"<<request.calibration.roll_deg<<",\"z_m\":"<<request.calibration.z_m
      <<"},\"imu_level_requested\":"<<(request.imu_level_enabled ? "true" : "false")
      <<",\"imu_mount\":{\"confirmed\":"<<(request.imu_mount.confirmed ? "true" : "false")
      <<",\"auto_axes\":"<<(request.imu_mount.auto_axes ? "true" : "false")
      <<",\"imu_frame\":"<<Json(request.imu_mount.imu_frame)<<",\"lidar_frame\":"<<Json(request.imu_mount.lidar_frame)
      <<",\"roll_deg\":"<<request.imu_mount.roll_deg<<",\"pitch_deg\":"<<request.imu_mount.pitch_deg
      <<",\"yaw_deg\":"<<request.imu_mount.yaw_deg<<"},\"roi_sensor_frame\":";
    WriteRegions(manifest,regions);manifest<<",\"samples\":["<<samples.str()<<"]}\n";
    settings::AtomicWriteText(request.destination/"metadata.json",manifest.str());
    return result;
  } catch (...) {
    std::error_code error;fs::remove_all(request.destination,error);
    if (error) throw std::runtime_error("Export did not complete; could not remove incomplete folder: "+request.destination.u8string());
    try {throw;} catch (const Canceled &) {result.canceled=true;return result;} catch (...) {throw;}
  }
}
}  // namespace lidar_viewer
