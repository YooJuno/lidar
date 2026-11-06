#include "lidar_viewer/dataset.hpp"
#include "lidar_viewer/roi.hpp"
#include <cmath>
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lidar_viewer
{
namespace
{
using Tree=boost::property_tree::ptree;
constexpr std::size_t kMaxPoints=20000000;
Tree Json(const fs::path & path)
{
  if (fs::file_size(path)>8*1024*1024) throw std::runtime_error("Dataset JSON is too large: "+path.u8string());
  std::ifstream file(path);Tree tree;boost::property_tree::read_json(file,tree);return tree;
}
fs::path Inside(const fs::path & root,const std::string & name)
{
  const auto path=fs::weakly_canonical(root/fs::u8path(name));
  const auto relative=path.lexically_relative(fs::weakly_canonical(root));
  if (relative.empty() || *relative.begin()==".." || relative.is_absolute())
    throw std::runtime_error("Dataset file must stay inside its folder");
  return path;
}
std::uint64_t Integer(std::istream & in,unsigned bytes)
{
  std::uint64_t value=0;
  for (unsigned i=0;i<bytes;++i) {const auto c=in.get();if (c<0) throw std::runtime_error("Truncated NumPy header");value|=std::uint64_t(c)<<(8*i);}
  return value;
}
double Scalar(const char * bytes,std::size_t width,char type,bool big=false)
{
  std::array<char,8> value{};
  const std::uint16_t endian=1;const bool host_little=*reinterpret_cast<const unsigned char *>(&endian)==1;
  if (big==host_little) std::reverse_copy(bytes,bytes+width,value.begin());else std::copy(bytes,bytes+width,value.begin());
  if (type=='f' || type=='F') {if (width==4) {float x;std::memcpy(&x,value.data(),4);return x;}
    if (width==8) {double x;std::memcpy(&x,value.data(),8);return x;}}
  if (type=='u' || type=='U') {if(width==1) return static_cast<unsigned char>(bytes[0]);
    if(width==2) {std::uint16_t x;std::memcpy(&x,value.data(),2);return x;}
    if(width==4) {std::uint32_t x;std::memcpy(&x,value.data(),4);return x;}
    if(width==8) {std::uint64_t x;std::memcpy(&x,value.data(),8);return static_cast<double>(x);}}
  if (type=='i' || type=='I') {if(width==1) return static_cast<signed char>(bytes[0]);
    if(width==2) {std::int16_t x;std::memcpy(&x,value.data(),2);return x;}
    if(width==4) {std::int32_t x;std::memcpy(&x,value.data(),4);return x;}
    if(width==8) {std::int64_t x;std::memcpy(&x,value.data(),8);return static_cast<double>(x);}}
  throw std::runtime_error("Unsupported numeric point format");
}
struct Array
{
  std::ifstream file;std::size_t rows{},columns{},width{};char type{};bool big{};
  explicit Array(const fs::path & path):file(path,std::ios::binary) {
    char magic[6];file.read(magic,6);
    if (!file || std::memcmp(magic,"\x93NUMPY",6)) throw std::runtime_error("Invalid NumPy file: "+path.u8string());
    const int version=file.get(),minor=file.get();(void)minor;
    if (version<1 || version>3) throw std::runtime_error("Unsupported NumPy version");
    const auto size=Integer(file,version==1 ? 2 : 4);
    if (size>65536) throw std::runtime_error("NumPy header exceeds 64 KiB");
    std::string header(static_cast<std::size_t>(size),' ');file.read(header.data(),static_cast<std::streamsize>(size));
    std::smatch match;
    if (!file || !std::regex_search(header,match,std::regex("['\"]descr['\"]\\s*:\\s*['\"]([<>=|])([fiu])([1248])['\"]")))
      throw std::runtime_error("Only numeric NumPy arrays are supported (no objects or pickle)");
    big=match[1]==">";type=match.str(2)[0];width=static_cast<std::size_t>(std::stoul(match[3]));
    if (!std::regex_search(header,std::regex("['\"]fortran_order['\"]\\s*:\\s*False")))
      throw std::runtime_error("NumPy array must use C row order");
    if (!std::regex_search(header,match,std::regex("['\"]shape['\"]\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]*)\\s*,?\\s*\\)")))
      throw std::runtime_error("Expected a one- or two-dimensional NumPy array");
    rows=static_cast<std::size_t>(std::stoull(match[1]));columns=match[2].length() ? static_cast<std::size_t>(std::stoull(match[2])) : 1;
    if (rows>kMaxPoints || columns<1 || columns>64 || width>8) throw std::runtime_error("NumPy shape exceeds viewer limits");
    const auto offset=file.tellg();
    if (offset<0 || fs::file_size(path)-static_cast<std::uintmax_t>(offset)!=rows*columns*width)
      throw std::runtime_error("NumPy payload size does not match its shape");
  }
};
FrameData Numpy(const fs::path & path)
{
  Array array(path);
  if (array.type!='f' || (array.width!=4 && array.width!=8) || (array.columns!=3 && array.columns!=4))
    throw std::runtime_error("Point arrays must be float32/float64 N x 3 or N x 4");
  FrameData frame;frame.points.resize(array.rows);
  std::array<char,32> row{};
  const std::uint16_t endian=1;
  const bool packed=array.columns==4 && array.width==4 && !array.big && *reinterpret_cast<const unsigned char *>(&endian)==1;
  if(packed) {
    static_assert(sizeof(KittiPoint)==16,"Packed XYZI required");
    array.file.read(reinterpret_cast<char *>(frame.points.data()),static_cast<std::streamsize>(frame.points.size()*sizeof(KittiPoint)));
    if(!array.file) throw std::runtime_error("Truncated NumPy point payload");
  }
  else for (auto & point:frame.points) {
    array.file.read(row.data(),static_cast<std::streamsize>(array.columns*array.width));
    point={static_cast<float>(Scalar(row.data(),array.width,'f',array.big)),
      static_cast<float>(Scalar(row.data()+array.width,array.width,'f',array.big)),
      static_cast<float>(Scalar(row.data()+2*array.width,array.width,'f',array.big)),
      array.columns==4 ? static_cast<float>(Scalar(row.data()+3*array.width,array.width,'f',array.big)) : 1.f};
  }
  const auto strength=path.parent_path()/"intensity.npy";
  if(!array.file) throw std::runtime_error("Truncated point array: "+path.u8string());
  if (path.filename()=="coord.npy" && array.columns==3 && fs::is_regular_file(strength)) {
    Array values(strength);
    if (values.rows!=array.rows || values.columns!=1) throw std::runtime_error("Intensity array does not match coordinate count");
    for (auto & point:frame.points) {values.file.read(row.data(),static_cast<std::streamsize>(values.width));point.intensity=static_cast<float>(Scalar(row.data(),values.width,values.type,values.big));}
    if(!values.file) throw std::runtime_error("Truncated intensity array");
  }
  return frame;
}
FrameData Pcd(const fs::path & path)
{
  std::ifstream file(path,std::ios::binary);std::vector<std::string> names;std::vector<std::size_t> sizes,counts;
  std::vector<char> types;std::size_t points=0,width=0,height=1,header_size=0;std::string format,line;
  while (std::getline(file,line)) {
    header_size+=line.size()+1;if(header_size>65536) throw std::runtime_error("PCD header exceeds 64 KiB");
    std::istringstream row(line);std::string key;row>>key;if (key.empty() || key[0]=='#') continue;
    if(key=="FIELDS" || key=="FIELD") {std::string v;while(row>>v) names.push_back(v);}
    else if(key=="SIZE") {std::size_t v;while(row>>v) sizes.push_back(v);}
    else if(key=="TYPE") {char v;while(row>>v) types.push_back(v);}
    else if(key=="COUNT") {std::size_t v;while(row>>v) counts.push_back(v);}
    else if(key=="POINTS") row>>points;else if(key=="WIDTH") row>>width;else if(key=="HEIGHT") row>>height;
    else if(key=="DATA") {row>>format;break;}
  }
  if (!points && width<=kMaxPoints && height<=kMaxPoints && height && width<=kMaxPoints/height) points=width*height;
  if (counts.empty()) counts.assign(names.size(),1);
  if (names.empty() || names.size()>64 || sizes.size()!=names.size() || types.size()!=names.size() || counts.size()!=names.size() || points>kMaxPoints)
    throw std::runtime_error("Invalid PCD fields or point count");
  std::vector<std::size_t> offsets(names.size()),columns(names.size());std::size_t stride=0,fields=0;
  std::array<int,4> selected{{-1,-1,-1,-1}};
  for(std::size_t i=0;i<names.size();++i) {
    if((sizes[i]!=1 && sizes[i]!=2 && sizes[i]!=4 && sizes[i]!=8) || counts[i]<1 || counts[i]>64)
      throw std::runtime_error("Invalid PCD field size");
    offsets[i]=stride;columns[i]=fields;stride+=sizes[i]*counts[i];fields+=counts[i];
    for(int j=0;j<4;++j) if(names[i]==std::array<const char *,4>{{"x","y","z","intensity"}}[static_cast<std::size_t>(j)]) selected[static_cast<std::size_t>(j)]=static_cast<int>(i);
  }
  if(selected[0]<0 || selected[1]<0 || selected[2]<0 || (format!="binary" && format!="ascii"))
    throw std::runtime_error("PCD needs x/y/z fields and binary or ascii data; binary_compressed is not supported");
  FrameData frame;frame.points.reserve(points);std::vector<char> bytes(stride);std::vector<double> values(fields);
  for(std::size_t i=0;i<points;++i) {
    if(format=="binary") {file.read(bytes.data(),static_cast<std::streamsize>(stride));if(!file) throw std::runtime_error("Truncated PCD data");}
    else for(auto & value:values) if(!(file>>value)) throw std::runtime_error("Invalid ASCII PCD data");
    std::array<float,4> point{{0,0,0,1}};
    for(std::size_t j=0;j<4;++j) if(selected[j]>=0) {const auto k=static_cast<std::size_t>(selected[j]);
      point[j]=static_cast<float>(format=="binary" ? Scalar(bytes.data()+offsets[k],sizes[k],types[k]) : values[columns[k]]);}
    frame.points.push_back({point[0],point[1],point[2],point[3]});
  }
  return frame;
}
FrameData BinaryXyzi(const fs::path & path)
{
  const auto bytes=fs::file_size(path);
  if(bytes%sizeof(KittiPoint) || bytes/sizeof(KittiPoint)>kMaxPoints)
    throw std::runtime_error("BIN points must contain float32 XYZI records (16 bytes per point)");
  FrameData frame;frame.points.resize(static_cast<std::size_t>(bytes/sizeof(KittiPoint)));
  std::ifstream input(path,std::ios::binary);
  input.read(reinterpret_cast<char *>(frame.points.data()),static_cast<std::streamsize>(bytes));
  if(!input) throw std::runtime_error("Cannot read BIN point cloud: "+path.u8string());
  return frame;
}
void Add(std::vector<FrameHandle> & frames,const fs::path & root,const fs::path & points,const fs::path & metadata,const std::string & name={})
{
  if (!fs::is_regular_file(points)) throw std::runtime_error("Missing dataset sample: "+points.u8string());
  auto sample=std::make_shared<DatasetSample>();sample->root=root;sample->metadata=metadata;
  sample->name=name.empty() ? points.stem().u8string() : name;sample->coordinate_space="stored";sample->annotation_status="unlabeled";
  if(sample->name.size()>512) throw std::runtime_error("Dataset sample name exceeds 512 bytes");
  for(auto & c:sample->name) if(static_cast<unsigned char>(c)<32) c=' ';
  if(fs::is_regular_file(metadata)) {
    const auto info=Json(metadata);sample->coordinate_space=info.get<std::string>("coordinate_space","sensor");
    sample->annotation_status=info.get<std::string>("annotation_status",info.get<std::string>("label_type","unlabeled"));
    sample->category=info.get<std::string>("category","");sample->source=info.get<std::string>("source.path",info.get<std::string>("source_bag",""));
    if(const auto hints=info.get_child_optional("labeling.regions")) for(const auto & entry:*hints) {
      const auto name=entry.second.get<std::string>("class_name","");
      if(name.size()>128 || sample->class_hints.size()>=128) throw std::runtime_error("Class hints exceed review limits");
      if(!name.empty() && std::find(sample->class_hints.begin(),sample->class_hints.end(),name)==sample->class_hints.end()) sample->class_hints.push_back(name);
    }
    sample->source_frame=info.get<std::size_t>("source.frame_number",info.get<std::size_t>("frame_index",0));
    sample->point_count=info.get<std::size_t>("point_count",0);
    if(!sample->source_frame) if(const auto index=info.get_optional<std::size_t>("source.frame_index")) sample->source_frame=*index+1;
    if(sample->source.size()>32768 || sample->coordinate_space.size()>64 || sample->annotation_status.size()>256 || sample->category.size()>256)
      throw std::runtime_error("Dataset metadata field exceeds viewer limits");
  }
  for(const auto & labels:{points.parent_path()/"segment.npy",root/"labels"/fs::u8path(points.stem().u8string()+".npy")})
    if(fs::is_regular_file(labels)) {sample->labels=labels;break;}
  FrameHandle frame;frame.path=points;frame.format=points.extension()==".npy" ? InputFormat::Numpy : points.extension()==".bin" ? InputFormat::DatasetXyzi : InputFormat::Pcd;
  frame.label=sample->name;frame.dataset_sample=sample;frames.push_back(std::move(frame));
}
void AddFile(std::vector<FrameHandle> & frames,const fs::path & points)
{
  auto root=points.parent_path();
  if((root.filename()=="points" || root.filename()=="pointcloud") &&
      (fs::is_directory(root.parent_path()/"metadata") || fs::is_directory(root.parent_path()/"labels") || fs::is_regular_file(root.parent_path()/"metadata.json"))) root=root.parent_path();
  const auto metadata=points.filename()=="coord.npy" && fs::is_regular_file(root/"meta.json") ? root/"meta.json" : root/"metadata"/fs::u8path(points.stem().u8string()+".json");
  Add(frames,root,points,metadata);
}
void Scan(const fs::path & root,std::vector<FrameHandle> & frames,InputLoadState * load,bool include_bin,unsigned depth=0)
{
  if(load) load->CheckCancelled();
  if(depth>32) throw std::runtime_error("Dataset nesting exceeds 32 folders");
  if(fs::is_regular_file(root/"metadata.json")) {
    const auto manifest=Json(root/"metadata.json");
    if(manifest.get<std::string>("schema","")=="lidar_studio.roi_dataset") {
      if(manifest.get<std::string>("status","")!="complete") throw std::runtime_error("Dataset export is incomplete");
      for(const auto & entry:manifest.get_child("samples")) {
        if(load) load->CheckCancelled();
        const auto & sample=entry.second;
        auto points=Inside(root,sample.get<std::string>("points"));
        if(!fs::is_regular_file(points)) points=Inside(root,sample.get<std::string>("pcd"));
        Add(frames,root,points,Inside(root,sample.get<std::string>("metadata")),sample.get<std::string>("id",""));
      }
      return;
    }
  }
  if(fs::is_regular_file(root/"coord.npy")) {Add(frames,root,root/"coord.npy",root/"meta.json",root.filename().u8string());return;}
  std::vector<fs::path> children,files;
  for(const auto & entry:fs::directory_iterator(root,fs::directory_options::skip_permission_denied)) {
    if(entry.is_symlink()) continue;
    if(entry.is_directory()) children.push_back(entry.path());
    else if(entry.is_regular_file() && (IsDatasetFile(entry.path()) || (include_bin && entry.path().extension()==".bin"))) files.push_back(entry.path());
  }
  std::sort(files.begin(),files.end());std::sort(children.begin(),children.end());
  for(const auto & file:files) {
    if(load) load->CheckCancelled();
    const auto stem=file.stem().u8string();
    if(stem=="intensity" || stem=="segment" || stem=="weak_segment" || stem=="ring" || stem=="point_time" || stem=="point_index" || stem=="draft_segment" || stem=="label_confidence" || stem=="source_cache_point_index") continue;
    if(file.extension()==".pcd" && fs::is_regular_file(file.parent_path()/fs::u8path(file.stem().u8string()+".npy"))) continue;
    AddFile(frames,file);
  }
  for(const auto & child:children) {
    const auto name=child.filename().u8string();
    if(name=="labels" || name=="annotations" || name=="metadata" || name=="training" || name==".git" || name=="__pycache__") continue;
    Scan(child,frames,load,include_bin,depth+1);
  }
}
}  // namespace
bool IsDatasetFile(const fs::path & path) {return path.extension()==".npy" || path.extension()==".pcd";}
std::vector<FrameHandle> ResolveDatasetFrames(const fs::path & path,InputLoadState * load,bool include_bin)
{
  if((path.filename()=="points" || path.filename()=="pointcloud") && fs::is_regular_file(path.parent_path()/"metadata.json"))
    return ResolveDatasetFrames(path.parent_path(),load,include_bin);
  std::vector<FrameHandle> frames;
  if(fs::is_regular_file(path)) {
    if(!IsDatasetFile(path) && !(include_bin && path.extension()==".bin")) throw std::runtime_error("Choose an NPY, PCD or XYZI BIN point cloud");
    AddFile(frames,path);
  } else Scan(path,frames,load,include_bin);
  std::map<std::string,std::size_t> names;
  for(const auto & frame:frames) ++names[frame.label];
  for(auto & frame:frames) if(names[frame.label]>1) {
    const auto prefix=frame.dataset_sample->root.lexically_relative(path).generic_u8string();
    if(!prefix.empty() && prefix!=".") frame.label=frame.dataset_sample->root.filename().u8string()==frame.label ? prefix : prefix+"/"+frame.label;
  }
  return frames;
}
FrameData LoadDatasetFrame(const FrameHandle & handle)
{
  auto frame=handle.format==InputFormat::Numpy ? Numpy(handle.path) : handle.format==InputFormat::DatasetXyzi ? BinaryXyzi(handle.path) : Pcd(handle.path);
  auto annotation=std::make_shared<DatasetAnnotations>();
  const auto root=handle.path.parent_path();
  const auto sample=handle.dataset_sample;
  bool draft_metadata=false;
  try {
    if(sample && fs::is_regular_file(sample->metadata)) {
      const auto meta=Json(sample->metadata);
      const auto status=meta.get<std::string>("annotation_status","")+" "+meta.get<std::string>("label_type","")+" "+meta.get<std::string>("annotation.state","");
      for(const auto * word:{"weak","draft","pseudo","predicted"}) draft_metadata|=status.find(word)!=std::string::npos;
      if(const auto classes=meta.get_child_optional("classes")) {
        int id=0;for(const auto & entry:*classes) {
          const auto name=entry.second.get_value<std::string>();
          if(name.size()>128 || id>=256) throw std::runtime_error("Class names exceed review limits");
          annotation->class_names.emplace_back(id++,name);
        }
      }
      if(const auto target=meta.get_optional<int>("target_class_id"))
        annotation->class_names.emplace_back(*target,meta.get<std::string>("category","Class "+std::to_string(*target)));
    }
    const auto manual=sample ? sample->labels : fs::path{};
    for(const auto & file:{manual,root/"draft_segment.npy",root/"weak_segment.npy"}) {
      if(file.empty() || !fs::is_regular_file(file)) continue;
      Array values(file);
      if(values.rows!=frame.points.size() || values.columns!=1 || (values.type!='i' && values.type!='u'))
        throw std::runtime_error("Point labels must be N integers in the same order as points: "+file.u8string());
      std::vector<std::int32_t> labels;labels.reserve(values.rows);std::array<char,8> bytes{};bool known=false;
      for(std::size_t i=0;i<values.rows;++i) {
        values.file.read(bytes.data(),static_cast<std::streamsize>(values.width));
        if(!values.file) throw std::runtime_error("Truncated point label array: "+file.u8string());
        const auto value=Scalar(bytes.data(),values.width,values.type,values.big);
        if(!std::isfinite(value) || value < -1 || value>65535) throw std::runtime_error("Label IDs must be -1 or 0..65535");
        labels.push_back(static_cast<std::int32_t>(value));known|=value>=0;
      }
      annotation->point_labels=std::move(labels);annotation->source=file.filename().u8string();annotation->draft=file!=manual || draft_metadata;
      if(known) break;
    }
    const auto box_name=fs::u8path(handle.path.stem().u8string()+".json");
    const auto boxes=sample && fs::is_regular_file(sample->root/"annotations"/box_name) ?
      sample->root/"annotations"/box_name : root/"annotations.json";
    if(fs::is_regular_file(boxes)) {
      const auto info=Json(boxes);
      if(sample && sample->coordinate_space!="stored" && info.get<std::string>("coordinate_space",sample->coordinate_space)!=sample->coordinate_space)
        throw std::runtime_error("Box annotation coordinate space differs from points");
      const auto vector=[](const Tree & tree) {
        if(tree.size()!=3) throw std::runtime_error("Annotation vectors need three numbers");
        auto it=tree.begin();Vec3 p;p.x=(it++)->second.get_value<float>();p.y=(it++)->second.get_value<float>();p.z=it->second.get_value<float>();return p;
      };
      for(const auto & entry:info.get_child("boxes")) {
        if(annotation->boxes.size()>=4096) throw std::runtime_error("Too many annotation boxes");
        const auto & data=entry.second;RoiBox box;
        box.name=data.get<std::string>("class_name");box.center=vector(data.get_child("center_m"));box.size=vector(data.get_child("size_m"));box.yaw_deg=data.get<float>("yaw_deg");ValidateRoiBox(box);
        annotation->boxes.push_back({box.name,box.center,box.size,box.yaw_deg});
      }
      annotation->box_annotations=true;
    }
  } catch(const std::exception & error) {
    annotation->error=error.what();std::vector<std::int32_t>().swap(annotation->point_labels);std::vector<DatasetBox>().swap(annotation->boxes);
  }
  // Keep labels in lockstep when invalid XYZ samples are discarded. The
  // normal frame finalizer then sees only valid coordinates.
  if(!annotation->point_labels.empty()) {
    std::size_t retained=0;
    for(std::size_t i=0;i<frame.points.size();++i) {
      const auto p=frame.points[i];if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)) continue;
      frame.points[retained]=p;annotation->point_labels[retained++]=annotation->point_labels[i];
    }
    frame.points.resize(retained);annotation->point_labels.resize(retained);
  }
  std::map<int,std::size_t> counts;for(const auto label:annotation->point_labels) ++counts[label];
  for(const auto & entry:counts) annotation->label_counts.push_back(entry);
  frame.annotations=std::move(annotation);return frame;
}
}  // namespace lidar_viewer
