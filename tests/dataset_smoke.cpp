#include "lidar_viewer/dataset.hpp"
#include "lidar_viewer/dataset_curation.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void Check(bool ok,const char * error) {if(!ok) throw std::runtime_error(error);}
std::string Read(const fs::path & path) {std::ifstream file(path,std::ios::binary);return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};}
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
    Check(ResolveInputFrames(root/"points").size()==1,"Opening points/ did not preserve its dataset manifest");
    auto cloud=LoadFrame(frames[0],FrameIntensity::Original);Check(cloud.points.size()==2 && cloud.points[0].z==-3 && cloud.points[1].intensity==1024,"Stored coordinates/intensity changed");
    Check(frames[0].dataset_sample && frames[0].dataset_sample->source_frame==14797 && frames[0].dataset_sample->coordinate_space=="aligned","Dataset provenance lost");
    fs::create_directory(root/"labels");std::int32_t file_labels[]{0,1};
    Npy(root/"labels/second.npy","<i4",2,1,{reinterpret_cast<char *>(file_labels),sizeof(file_labels)});
    auto single=lidar_viewer::ResolveDatasetFrames(root/"points/second.npy");
    Check(single.size()==1 && single[0].dataset_sample->coordinate_space=="aligned" &&
      LoadFrame(single[0]).annotations->point_labels==std::vector<std::int32_t>({0,1}),"Opening one exported file lost sibling metadata/labels");
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
    std::int64_t labels[]{-1,2};
    Npy(root/"organized/segment.npy","<i8",2,1,{reinterpret_cast<char *>(labels),sizeof(labels)});
    organized=ResolveInputFrames(root/"organized");auto annotated=LoadFrame(organized[0]);
    Check(annotated.annotations && annotated.annotations->point_labels==std::vector<std::int32_t>({-1,2}) &&
      !annotated.annotations->draft,"Manual point labels failed to load in source order");
    labels[0]=-1;labels[1]=-1;Npy(root/"organized/segment.npy","<i8",2,1,{reinterpret_cast<char *>(labels),sizeof(labels)});
    labels[0]=1;labels[1]=2;Npy(root/"organized/draft_segment.npy","<i8",2,1,{reinterpret_cast<char *>(labels),sizeof(labels)});
    annotated=LoadFrame(organized[0]);Check(annotated.annotations->draft && annotated.annotations->source=="draft_segment.npy","Draft labels were not distinguished from manual annotations");
    Npy(root/"organized/draft_segment.npy","<i8",1,1,{reinterpret_cast<char *>(labels),sizeof(labels[0])});
    annotated=LoadFrame(organized[0]);Check(!annotated.annotations->error.empty() && annotated.annotations->point_labels.empty() && annotated.points.size()==2,"Invalid label counts must leave points viewable with an error");
    fs::remove(root/"organized/draft_segment.npy");
    labels[0]=1;labels[1]=2;Npy(root/"organized/segment.npy","<i8",2,1,{reinterpret_cast<char *>(labels),sizeof(labels)});
    coords[0]=std::numeric_limits<float>::quiet_NaN();Npy(root/"organized/coord.npy","<f4",2,3,{reinterpret_cast<char *>(coords),sizeof(coords)});
    Text(root/"organized/annotations.json","{\"coordinate_space\":\"sensor\",\"boxes\":[{\"class_name\":\"person\",\"center_m\":[4,5,6],\"size_m\":[1,1,2],\"yaw_deg\":30}]}");
    annotated=LoadFrame(organized[0]);Check(annotated.points.size()==1 && annotated.annotations->point_labels==std::vector<std::int32_t>({2}) &&
      annotated.annotations->box_annotations && annotated.annotations->boxes.size()==1,"Invalid XYZ removal misaligned labels or lost box annotations");
    fs::create_directory(root/"raw");{std::ofstream file(root/"raw/sample.bin",std::ios::binary);file.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));}
    auto raw=lidar_viewer::ResolveDatasetFrames(root/"raw",nullptr,true);
    Check(raw.size()==1 && raw[0].format==InputFormat::DatasetXyzi,"Review did not discover BIN samples");
    auto raw_frame=LoadFrame(raw[0],FrameIntensity::Original);
    Check(raw_frame.annotations && raw_frame.points[0].z==-3 && raw_frame.points[1].intensity==1024,"Review BIN transformed stored coordinates or intensity");
    Check(lidar_viewer::ResolveDatasetFrames(root/"raw").empty(),"Review BIN changed normal bag/KITTI discovery");
    Check(ResolveInputFrames(root/"raw/sample.bin")[0].format==InputFormat::KittiXyzi,"Normal BIN format changed");
    Text(root/"raw/broken.bin","bad");Reject([&]{LoadFrame(lidar_viewer::ResolveDatasetFrames(root/"raw/broken.bin",nullptr,true)[0]);});
    Reject([&]{lidar_viewer::ResolveDatasetFrames(root/"metadata.json");});
    fs::create_directory(root/"numbered");
    const std::vector<std::string> expected{"scan_frame_2.npy","scan_frame_0003.npy","scan_frame_10.npy","scan_frame_1864.npy","scan_frame_1868.npy","scan_frame_15506.npy","scan_frame_15510.npy","scan_frame_15514.npy","scan_frame_999999999999999999999999999999.npy"};
    for(auto it=expected.rbegin();it!=expected.rend();++it) Npy(root/"numbered"/(*it),"<f4",2,4,bytes);
    const auto numbered=lidar_viewer::ResolveDatasetFrames(root/"numbered");
    Check(numbered.size()==expected.size(),"Numbered cloud scan lost files");
    for(std::size_t i=0;i<expected.size();++i) Check(numbered[i].path.filename()==expected[i],"Point files are not in numeric frame order");
    Text(root/"numbered/metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"complete\",\"samples\":[{\"id\":\"frame_15506\",\"points\":\"scan_frame_15506.npy\",\"metadata\":\"missing.json\"},{\"id\":\"frame_1864\",\"points\":\"scan_frame_1864.npy\",\"metadata\":\"missing.json\"}]}");
    const auto manifest_sorted=lidar_viewer::ResolveDatasetFrames(root/"numbered");
    Check(manifest_sorted.size()==2 && manifest_sorted[0].label=="frame_1864","Manifest samples are not in numeric frame order");
    fs::create_directory(root/"numbered_folders");
    for(const auto * name:{"sample_15506","sample_1864","sample_10","sample_2"}) {
      fs::create_directory(root/"numbered_folders"/name);Npy(root/"numbered_folders"/name/"coord.npy","<f4",2,4,bytes);
    }
    const auto folders=lidar_viewer::ResolveDatasetFrames(root/"numbered_folders");
    Check(folders.size()==4 && folders[0].label=="sample_2" && folders[1].label=="sample_10" && folders[2].label=="sample_1864" && folders[3].label=="sample_15506","Sample folders are not in numeric frame order");
    const auto curate=root/"curation";for(const auto * dir:{"points","pointcloud","metadata","labels","annotations"}) fs::create_directories(curate/dir);
    for(const auto * id:{"a","b"}) {
      Npy(curate/"points"/(std::string(id)+".npy"),"<f4",2,4,bytes);
      Npy(curate/"labels"/(std::string(id)+".npy"),"<i4",2,1,{reinterpret_cast<char *>(file_labels),sizeof(file_labels)});
      Text(curate/"metadata"/(std::string(id)+".json"),"{\"point_count\":2}");
    }
    Npy(curate/"labels/a-custom.npy","<i4",2,1,{reinterpret_cast<char *>(file_labels),sizeof(file_labels)});Text(curate/"annotations/a-custom.json","{\"boxes\":[]}");
    Text(curate/"annotations/a.json","{\"boxes\":[]}");Text(curate/"pointcloud/a.pcd","PCD alias");
    const std::string manifest="{\"schema\":\"lidar_studio.roi_dataset\",\"version\":1,\"flag\":true,\"status\":\"complete\",\"frames_exported\":2,\"points_exported\":4,\"samples\":[{\"id\":\"a-custom\",\"points\":\"points/a.npy\",\"pcd\":\"pointcloud/a.pcd\",\"metadata\":\"metadata/a.json\"},{\"id\":\"b\",\"points\":\"points/b.npy\",\"metadata\":\"metadata/b.json\"}]}";
    Text(curate/"metadata.json",manifest);auto editable=lidar_viewer::ResolveDatasetFrames(curate);
    auto removed=lidar_viewer::TrashDatasetSample(editable[0]);
    Check(!fs::exists(curate/"points/a.npy") && !fs::exists(curate/"pointcloud/a.pcd") && !fs::exists(curate/"labels/a.npy") && !fs::exists(curate/"metadata/a.json") && !fs::exists(curate/"annotations/a.json"),"Sample deletion left its own annotations or alternate point format");
    Check(!fs::exists(curate/"labels/a-custom.npy") && !fs::exists(curate/"annotations/a-custom.json"),"Deletion missed annotations named by manifest ID");
    Check(fs::exists(curate/"points/b.npy") && lidar_viewer::ResolveDatasetFrames(curate).size()==1,"Deleting one sample damaged its neighbor");
    const auto changed=Read(curate/"metadata.json");
    Check(changed.find("\"version\":1")!=std::string::npos && changed.find("\"flag\":true")!=std::string::npos && changed.find("\"frames_exported\":1")!=std::string::npos && changed.find("\"points_exported\":2")!=std::string::npos,"Manifest edit changed JSON types or left stale sample counts");
    Text(curate/"points/a.npy","new file");Reject([&]{lidar_viewer::RestoreDatasetSample(removed);});Check(Read(curate/"points/a.npy")=="new file","Recovery overwrote a new file");fs::remove(curate/"points/a.npy");
    Text(curate/"metadata.json",changed+" ");Reject([&]{lidar_viewer::RestoreDatasetSample(removed);});Check(!fs::exists(curate/"points/a.npy"),"Recovery ignored a changed manifest");Text(curate/"metadata.json",changed);
    lidar_viewer::RestoreDatasetSample(removed);Check(Read(curate/"metadata.json")==manifest && Read(curate/"points/a.npy")==Read(root/"points/second.npy") && fs::exists(curate/"labels/a.npy"),"Recovery did not restore original files and manifest bytes");
    auto first_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(curate)[0]);
    auto last_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(curate)[0]);
    Check(lidar_viewer::ResolveDatasetFrames(curate).empty(),"Deleting the last sample left a malformed or nonempty manifest");
    lidar_viewer::RestoreDatasetSample(last_deleted);lidar_viewer::RestoreDatasetSample(first_deleted);
    Check(Read(curate/"metadata.json")==manifest,"Multiple deletions could not be undone in reverse order");
    auto shared=manifest;const auto at=shared.find("metadata/b.json");shared.replace(at,std::string("metadata/b.json").size(),"metadata/a.json");Text(curate/"metadata.json",shared);
    auto shared_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(curate)[0]);Check(fs::exists(curate/"metadata/a.json"),"Deletion removed metadata shared by another sample");lidar_viewer::RestoreDatasetSample(shared_deleted);
    const auto curated_coord=root/"curated_coord";fs::create_directory(curated_coord);
    Npy(curated_coord/"coord.npy","<f4",2,4,bytes);Npy(curated_coord/"segment.npy","<i4",2,1,{reinterpret_cast<char *>(file_labels),sizeof(file_labels)});Text(curated_coord/"meta.json","{}");Text(curated_coord/"keep.txt","unrelated");
    auto coord_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(curated_coord)[0]);
    Check(lidar_viewer::ResolveDatasetFrames(curated_coord).empty() && !fs::exists(curated_coord/"segment.npy") && Read(curated_coord/"keep.txt")=="unrelated","Organized deletion did not skip trash or preserve unrelated files");
    lidar_viewer::RestoreDatasetSample(coord_deleted);Check(LoadFrame(lidar_viewer::ResolveDatasetFrames(curated_coord)[0]).annotations->point_labels==std::vector<std::int32_t>({0,1}),"Organized recovery lost labels");
    const auto relative_curate=fs::relative(curate,fs::current_path());
    auto relative_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(relative_curate)[0]);lidar_viewer::RestoreDatasetSample(relative_deleted);
    const auto flat=root/"curated_flat";fs::create_directory(flat);Npy(flat/"first.npy","<f4",2,4,bytes);Npy(flat/"second.npy","<f4",2,4,bytes);
    Npy(flat/"segment.npy","<i4",2,1,{reinterpret_cast<char *>(file_labels),sizeof(file_labels)});
    auto flat_deleted=lidar_viewer::TrashDatasetSample(lidar_viewer::ResolveDatasetFrames(flat)[0]);
    Check(fs::exists(flat/"segment.npy") && LoadFrame(lidar_viewer::ResolveDatasetFrames(flat)[0]).annotations->point_labels.size()==2,"Deletion removed a shared flat-folder label mask");lidar_viewer::RestoreDatasetSample(flat_deleted);
    Reject([&]{lidar_viewer::TrashDatasetSample(ResolveInputFrames(root/"raw/sample.bin")[0]);});
    Npy(root/"object.npy","|O8",2,4,bytes);Reject([&]{LoadFrame(ResolveInputFrames(root/"object.npy")[0]);});
    Npy(root/"fortran.npy","<f4",2,4,bytes,true);Reject([&]{LoadFrame(ResolveInputFrames(root/"fortran.npy")[0]);});
    Npy(root/"truncated.npy","<f4",3,4,bytes);Reject([&]{LoadFrame(ResolveInputFrames(root/"truncated.npy")[0]);});
    Text(root/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"complete\",\"samples\":[{\"points\":\"../outside.npy\"}]}");Reject([&]{ResolveInputFrames(root);});
    Text(root/"metadata.json","{\"schema\":\"lidar_studio.roi_dataset\",\"status\":\"exporting\"}");Reject([&]{ResolveInputFrames(root);});
    fs::remove_all(root);std::cout<<"Dataset: NPY/PCD, manifest, coordinates, organized arrays and invalid inputs passed\n";return 0;
  } catch(const std::exception & error) {fs::remove_all(root);std::cerr<<error.what()<<'\n';return 1;}
}
