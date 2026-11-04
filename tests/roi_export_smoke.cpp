#include "lidar_viewer/roi_export.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>

using namespace lidar_viewer;
namespace
{
void Check(bool value,const char * message) {if (!value) throw std::runtime_error(message);}
std::string Text(const fs::path & path)
{
  std::ifstream in(path,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};
}
std::vector<KittiPoint> Npy(const fs::path & path)
{
  const auto bytes=Text(path);
  Check(bytes.size()>=10 && bytes.substr(1,5)=="NUMPY" && bytes[6]==1 && bytes[7]==0,"Invalid NPY magic/version");
  const auto length=static_cast<unsigned char>(bytes[8])+256u*static_cast<unsigned char>(bytes[9]);
  Check((10+length)%64==0 && bytes.size()>=10+length && bytes[9+length]=='\n',"Invalid NPY header alignment");
  const auto header=bytes.substr(10,length);
  Check(header.find("'<f4'")!=std::string::npos && header.find("'fortran_order': False")!=std::string::npos,"Wrong NPY dtype/layout");
  const auto size=bytes.size()-10-length;
  Check(size%sizeof(KittiPoint)==0,"Truncated NPY data");
  std::vector<KittiPoint> points(size/sizeof(KittiPoint));
  std::ifstream in(path,std::ios::binary);in.seekg(10+length);
  in.read(reinterpret_cast<char *>(points.data()),static_cast<std::streamsize>(size));
  return points;
}
template<class F> void Reject(F action)
{
  bool rejected=false;try {action();} catch (const std::exception &) {rejected=true;}
  Check(rejected,"Expected export to reject invalid input");
}
}

int main()
{
  const auto root=fs::temp_directory_path()/("lidar-export-test-"+std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    fs::create_directories(root);
    RoiExportRequest request;request.destination=root/"sensor";
    request.regions={false,true,{{"near",true,{1,0,0},{2,2,4},0},
      {"overlap",true,{2,0,0},{4,2,4},0},{"disabled",false,{8,2,0},{2,2,2},0}}};
    FrameData frame;frame.frame_id="lidar\"\\\nframe";frame.timestamp_ns=1793710800123456789LL;
    frame.points={{1,0,-.5f,42},{1,0,-.5f,84},{3,0,-.5f,512},{8,2,0,1000},{0,0,0,200}};
    request.live_snapshot=frame;request.live_sequence=57;
    request.calibration={90,0,0,2};request.global_ground_offset=.5f;
    auto progress=std::make_shared<RoiExportProgress>();
    const auto result=ExportRoiDataset(request,progress);
    Check(result.frames==1 && result.points==3 && progress->completed==1,"Wrong exported counts");
    const auto points=Npy(result.destination/"points/00000000.npy");
    Check(points.size()==3 && points[0].intensity==42 && points[1].intensity==84 && points[2].intensity==512 &&
      points[0].x==1 && points[0].z==-.5f,"Lost original XYZI, duplicate points, ordering or ROI union");
    const auto pcd=Text(result.destination/"pointcloud/00000000.pcd");
    const auto at=pcd.find("DATA binary\n");
    Check(at!=std::string::npos && pcd.find("FIELDS x y z intensity\n")!=std::string::npos &&
      pcd.find("WIDTH 3\n")!=std::string::npos && pcd.substr(at+12)==
      std::string(reinterpret_cast<const char *>(points.data()),points.size()*sizeof(KittiPoint)),
      "PCD and NPY point payloads differ");
    const auto meta=Text(result.destination/"metadata/00000000.json");
    Check(meta.find("\"annotation_status\":\"unlabeled\"")!=std::string::npos &&
      meta.find("\"timestamp_ns\":\"1793710800123456789\"")!=std::string::npos &&
      meta.find("lidar\\\"\\\\\\u000aframe")!=std::string::npos,"Incomplete or improperly escaped metadata");
    request.destination=root/"aligned";request.aligned_coordinates=true;
    auto intensities=std::make_shared<std::vector<float>>();
    for (const auto & p:frame.points) intensities->push_back(p.intensity);
    request.live_snapshot->source_intensities=intensities;
    for (auto & p:request.live_snapshot->points) p.intensity=.5f;
    ExportRoiDataset(request,std::make_shared<RoiExportProgress>());
    const auto aligned=Npy(request.destination/"points/00000000.npy");
    Check(std::abs(aligned[0].x)<1e-5 && std::abs(aligned[0].y-1)<1e-5 &&
      std::abs(aligned[0].z-1)<1e-5 && aligned[0].intensity==42,"Aligned coordinates/intensity are incorrect");
    // Exercise the original-intensity file-loading path, which differs from
    // the viewer's normalized intensity path.
    const auto input=root/"input.bin";
    {std::ofstream out(input,std::ios::binary);out.write(reinterpret_cast<const char *>(frame.points.data()),
        static_cast<std::streamsize>(frame.points.size()*sizeof(KittiPoint)));}
    const auto original=LoadFrame({input},FrameIntensity::Original);
    const auto normalized=LoadFrame({input});
    Check(original.points[2].intensity==512 && normalized.points[2].intensity<=1,"Original loading changed display behavior or normalized training values");
    request.live_snapshot.reset();request.sources={{{input},7},{{input},9}};request.aligned_coordinates=false;
    request.destination=root/"batch";ExportRoiDataset(request,std::make_shared<RoiExportProgress>());
    Check(Npy(request.destination/"points/00000001.npy").size()==3 &&
      Text(request.destination/"metadata/00000001.json").find("\"frame_index\":9")!=std::string::npos,
      "Batch source indices or frame payloads were lost");
    const auto saved=Text(request.destination/"metadata.json");
    Reject([&]{ExportRoiDataset(request,std::make_shared<RoiExportProgress>());});
    Check(Text(request.destination/"metadata.json")==saved,"Existing export was overwritten");
    request.destination=root/"empty";request.regions.boxes={RoiBox{"empty",true,{90,90,90},{1,1,1},0}};
    Reject([&]{ExportRoiDataset(request,std::make_shared<RoiExportProgress>());});
    Check(!fs::exists(request.destination),"Empty failed export left a misleading dataset");
    request.regions.boxes[0].enabled=false;request.destination=root/"disabled";
    Reject([&]{ExportRoiDataset(request,std::make_shared<RoiExportProgress>());});
    Check(!fs::exists(request.destination),"Disabled ROI created an export");
    request.regions.boxes={RoiBox{"all",true,{0,0,0},{20,20,20},0}};
    request.destination=root/"missing";request.sources[1].handle.path=root/"does-not-exist.bin";
    Reject([&]{ExportRoiDataset(request,std::make_shared<RoiExportProgress>());});
    Check(!fs::exists(request.destination),"Source read failure left a partial dataset");
    request.sources.assign(1000,{{input},0});request.destination=root/"canceled";
    progress=std::make_shared<RoiExportProgress>();
    auto future=std::async(std::launch::async,[&]{return ExportRoiDataset(request,progress);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (progress->completed==0 && std::chrono::steady_clock::now()<deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    progress->cancel=true;
    Check(future.get().canceled && !fs::exists(request.destination),"Canceled export was retained as complete");
    request.destination=root/"pre-canceled";
    Check(ExportRoiDataset(request,progress).canceled && !fs::exists(request.destination),"Cancel before worker start failed");
    fs::remove_all(root);
    std::cout<<"ROI export: original intensity, NPY/PCD, union, transforms, metadata, batch, collisions and cancellation passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::error_code ignored;fs::remove_all(root,ignored);std::cerr<<error.what()<<'\n';return 1;
  }
}
