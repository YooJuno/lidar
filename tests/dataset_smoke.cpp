#include "lidar_viewer/dataset.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void Check(bool ok,const char * error) {if(!ok) throw std::runtime_error(error);}
void Text(const fs::path & path,const std::string & text) {std::ofstream file(path);file<<text;}
void Npy(const fs::path & path,const std::string & type,std::size_t rows,std::size_t columns,const std::string & payload,bool fortran=false)
{
  const auto header="{'descr': '"+type+"', 'fortran_order': "+(fortran ? "True" : "False")+", 'shape': ("+std::to_string(rows)+", "+std::to_string(columns)+"), }\n";
  std::ofstream file(path,std::ios::binary);file.write("\x93NUMPY\1\0",8);file.put(static_cast<char>(header.size()&255));file.put(static_cast<char>(header.size()>>8));file<<header;file.write(payload.data(),static_cast<std::streamsize>(payload.size()));
}
template<class F> void Reject(F action) {bool failed=false;try {action();} catch(const std::exception &) {failed=true;}Check(failed,"Invalid dataset was accepted");}
}
int main()
{
  const auto root=fs::temp_directory_path()/("studio-dataset-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    fs::create_directories(root/"points");fs::create_directory(root/"metadata");fs::create_directory(root/"pointcloud");
    const KittiPoint values[]{{1,2,-3,512},{2,3,-2,1024}};
    const std::string bytes(reinterpret_cast<const char *>(values),sizeof(values));
    Npy(root/"points/second.npy","<f4",2,4,bytes);
    Text(root/"metadata/second.json","{\"coordinate_space\":\"aligned\",\"point_count\":2,\"source\":{\"path\":\"bag\",\"frame_number\":14797},\"annotation_status\":\"unlabeled\"}");
    Text(root/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"complete\",\"samples\":[{\"id\":\"second\",\"points\":\"points/second.npy\",\"pcd\":\"pointcloud/second.pcd\",\"metadata\":\"metadata/second.json\"}]}");
    auto frames=ResolveInputFrames(root);Check(frames.size()==1 && frames[0].format==InputFormat::Numpy,"Manifest did not resolve exactly one sample");
    auto cloud=LoadFrame(frames[0],FrameIntensity::Original);Check(cloud.points.size()==2 && cloud.points[0].z==-3 && cloud.points[1].intensity==1024,"Stored coordinates/intensity changed");
    Check(frames[0].dataset_sample && frames[0].dataset_sample->source_frame==14797 && frames[0].dataset_sample->coordinate_space=="aligned","Dataset provenance lost");
    Text(root/"ascii.pcd","VERSION .7\nFIELDS intensity z x y\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nPOINTS 2\nDATA ascii\n512 -3 1 2\n1024 -2 2 3\n");
    auto ascii=LoadFrame(ResolveInputFrames(root/"ascii.pcd")[0],FrameIntensity::Original);
    Check(ascii.points[0].x==1 && ascii.points[1].intensity==1024,"ASCII PCD field mapping failed");
    {std::ofstream file(root/"binary.pcd",std::ios::binary);file<<"FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nPOINTS 2\nDATA binary\n";file.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));}
    Check(LoadFrame(ResolveInputFrames(root/"binary.pcd")[0],FrameIntensity::Original).points[0].intensity==512,"Binary PCD failed");
    fs::create_directory(root/"organized");float coords[]{1,2,3,4,5,6},intensity[]{42,84};
    Npy(root/"organized/coord.npy","<f4",2,3,{reinterpret_cast<char *>(coords),sizeof(coords)});
    Npy(root/"organized/intensity.npy","<f4",2,1,{reinterpret_cast<char *>(intensity),sizeof(intensity)});
    auto organized=ResolveInputFrames(root/"organized");Check(organized.size()==1,"Helper arrays became extra samples");
    Check(LoadFrame(organized[0],FrameIntensity::Original).points[1].intensity==84,"Separate intensity array failed");
    Npy(root/"object.npy","|O8",2,4,bytes);Reject([&]{LoadFrame(ResolveInputFrames(root/"object.npy")[0]);});
    Npy(root/"fortran.npy","<f4",2,4,bytes,true);Reject([&]{LoadFrame(ResolveInputFrames(root/"fortran.npy")[0]);});
    Npy(root/"truncated.npy","<f4",3,4,bytes);Reject([&]{LoadFrame(ResolveInputFrames(root/"truncated.npy")[0]);});
    Text(root/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"complete\",\"samples\":[{\"points\":\"../outside.npy\"}]}");Reject([&]{ResolveInputFrames(root);});
    Text(root/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"exporting\"}");Reject([&]{ResolveInputFrames(root);});
    fs::remove_all(root);std::cout<<"Dataset: NPY/PCD, manifest, coordinates, organized arrays and invalid inputs passed\n";return 0;
  } catch(const std::exception & error) {fs::remove_all(root);std::cerr<<error.what()<<'\n';return 1;}
}
