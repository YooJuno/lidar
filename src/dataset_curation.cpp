#include "lidar_viewer/dataset_curation.hpp"
#include "lidar_viewer/settings_io.hpp"
#include "lidar_viewer/platform.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <fstream>
#include <exception>
#include <iomanip>
#include <iterator>
#include <set>
#include <sstream>

namespace lidar_viewer {
namespace {
using Tree=boost::property_tree::ptree;
std::string Read(const fs::path & path) {
  if(fs::file_size(path)>8*1024*1024) throw std::runtime_error("Dataset JSON exceeds 8 MiB");
  std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot read "+path.u8string());
  return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
Tree Parse(const std::string & text) {std::istringstream in(text);Tree value;boost::property_tree::read_json(in,value);return value;}
void Space(const std::string & s,std::size_t & i) {while(i<s.size() && (s[i]==' ' || s[i]=='\t' || s[i]=='\r' || s[i]=='\n')) ++i;}
std::size_t End(const std::string & s,std::size_t i,unsigned depth=0) {
  Space(s,i);if(i>=s.size() || depth>64) throw std::runtime_error("Invalid dataset JSON structure");
  if(s[i]=='"') {for(++i;i<s.size();++i) {if(s[i]=='\\') ++i;else if(s[i]=='"') return i+1;}throw std::runtime_error("Unterminated JSON string");}
  if(s[i]=='{' || s[i]=='[') {
    const char close=s[i]=='{' ? '}' : ']';++i;
    while(i<s.size()) {Space(s,i);if(i<s.size() && s[i]==close) return i+1;
      if(i<s.size() && (s[i]==',' || s[i]==':')) ++i;else i=End(s,i,depth+1);}
    throw std::runtime_error("Unterminated JSON container");
  }
  const auto start=i;while(i<s.size() && s[i]!=',' && s[i]!=']' && s[i]!='}' && s[i]!=' ' && s[i]!='\r' && s[i]!='\n' && s[i]!='\t') ++i;
  if(i==start) throw std::runtime_error("Invalid JSON value");
  return i;
}
using Span=std::pair<std::size_t,std::size_t>;
Span Member(const std::string & text,const std::string & key) {
  std::size_t i=0;Space(text,i);if(i>=text.size() || text[i++]!='{') throw std::runtime_error("Expected dataset JSON object");
  while(i<text.size()) {Space(text,i);if(text[i]=='}') break;
    const auto start=i;const auto end=End(text,i);i=end;Space(text,i);
    if(i>=text.size() || text[i++]!=':') throw std::runtime_error("Invalid dataset JSON member");
    Space(text,i);const auto value=i;i=End(text,i);
    if(text.compare(start,end-start,"\""+key+"\"")==0) return {value,i};
    Space(text,i);if(i<text.size() && text[i]==',') ++i;else break;
  }
  return {std::string::npos,std::string::npos};
}
std::vector<Span> Elements(const std::string & text,Span array) {
  if(array.first==std::string::npos || text[array.first]!='[') throw std::runtime_error("Dataset needs a samples array");
  std::vector<Span> result;auto i=array.first+1;
  while(i<array.second) {Space(text,i);if(text[i]==']') break;const auto start=i;i=End(text,i);result.emplace_back(start,i);Space(text,i);if(text[i]==',') ++i;else break;}
  return result;
}
void Number(std::string & text,const std::string & key,std::size_t value) {const auto at=Member(text,key);if(at.first!=std::string::npos) text.replace(at.first,at.second-at.first,std::to_string(value));}
fs::path Inside(const fs::path & root,const fs::path & value) {
  const auto path=fs::weakly_canonical(value.is_absolute() ? value : root/value);
  const auto relative=path.lexically_relative(root);
  if(relative.empty() || relative.is_absolute() || *relative.begin()=="..") throw std::runtime_error("Sample files must stay inside their dataset");
  return path;
}
void Move(const fs::path & from,const fs::path & to) {
  if(fs::exists(to) || fs::is_symlink(fs::symlink_status(to))) throw std::runtime_error("File already exists; recovery will not overwrite it: "+to.u8string());
  fs::create_directories(to.parent_path());fs::rename(from,to);
}
void Rollback(const std::vector<std::pair<fs::path,fs::path>> & completed) {
  for(auto i=completed.rbegin();i!=completed.rend();++i) Move(i->second,i->first);
}
}
DatasetTrashEntry TrashDatasetSample(const FrameHandle & frame) {
  if(!frame.dataset_sample || (frame.format!=InputFormat::Numpy && frame.format!=InputFormat::Pcd && frame.format!=InputFormat::DatasetXyzi))
    throw std::runtime_error("Only point-cloud dataset samples can be deleted; bag recordings stay intact");
  if(fs::is_symlink(fs::symlink_status(frame.path))) throw std::runtime_error("Cannot delete a symlinked point cloud");
  const auto & sample=*frame.dataset_sample;DatasetTrashEntry entry;entry.root=fs::canonical(sample.root);
  const auto points=Inside(entry.root,fs::absolute(frame.path));if(!fs::is_regular_file(points)) throw std::runtime_error("The sample file is already missing");
  std::set<fs::path> candidates;
  const auto add=[&](const fs::path & path) {
    if(path.empty()) return;
    if(fs::is_symlink(fs::symlink_status(path))) throw std::runtime_error("Cannot delete a symlinked annotation");
    if(fs::is_regular_file(path)) candidates.insert(Inside(entry.root,fs::absolute(path)));
  };
  add(frame.path);add(sample.metadata);
  if(frame.path.filename()=="coord.npy") {
    for(const auto * name:{"intensity.npy","segment.npy","draft_segment.npy","weak_segment.npy","label_confidence.npy","ring.npy","point_time.npy","point_index.npy","source_cache_point_index.npy","meta.json","annotations.json"}) add(frame.path.parent_path()/name);
  } else {
    const auto stem=frame.path.stem().u8string();
    add(entry.root/"labels"/fs::u8path(stem+".npy"));add(entry.root/"annotations"/fs::u8path(stem+".json"));
    if(!sample.labels.empty() && sample.labels.filename()!="segment.npy") add(sample.labels);
    if(frame.format==InputFormat::Numpy || frame.format==InputFormat::Pcd) {
      auto twin=frame.path;twin.replace_extension(frame.format==InputFormat::Numpy ? ".pcd" : ".npy");add(twin);
      add(entry.root/(frame.format==InputFormat::Numpy ? "pointcloud" : "points")/fs::u8path(stem+(frame.format==InputFormat::Numpy ? ".pcd" : ".npy")));
    }
  }
  std::string original,updated;
  const auto manifest=entry.root/"metadata.json";
  if(fs::is_regular_file(manifest)) {
    original=Read(manifest);const auto data=Parse(original);
    if(data.get<std::string>("schema","")=="lidar_studio.roi_dataset") {
      if(data.get<std::string>("status","")!="complete") throw std::runtime_error("Cannot edit an incomplete dataset export");
      const auto & samples=data.get_child("samples");std::size_t index=0,target=0,matches=0;
      const auto refers=[&](const Tree & item) {
        for(const auto * key:{"points","pcd"}) if(const auto value=item.get_optional<std::string>(key)) if(Inside(entry.root,fs::u8path(*value))==points) return true;
        return false;
      };
      for(const auto & item:samples) {
        if(refers(item.second)) {
          target=index;++matches;
          for(const auto * key:{"points","pcd","metadata"}) if(const auto value=item.second.get_optional<std::string>(key)) add(entry.root/fs::u8path(*value));
          if(const auto id=item.second.get_optional<std::string>("id")) {
            add(entry.root/"labels"/fs::u8path(*id+".npy"));add(entry.root/"annotations"/fs::u8path(*id+".json"));
          }
        }
        ++index;
      }
      if(matches!=1) throw std::runtime_error("Sample must have exactly one matching manifest entry");
      index=0;for(const auto & item:samples) {if(index++==target) continue;
        for(const auto * key:{"points","pcd","metadata"}) if(const auto value=item.second.get_optional<std::string>(key)) candidates.erase(Inside(entry.root,fs::u8path(*value)));
        if(const auto id=item.second.get_optional<std::string>("id")) {
          candidates.erase(Inside(entry.root,entry.root/"labels"/fs::u8path(*id+".npy")));
          candidates.erase(Inside(entry.root,entry.root/"annotations"/fs::u8path(*id+".json")));
        }
      }
      const auto array=Member(original,"samples");const auto spans=Elements(original,array);
      if(spans.size()!=samples.size()) throw std::runtime_error("Dataset samples array does not match its manifest");
      auto first=spans[target].first,last=spans[target].second;
      if(target+1<spans.size()) last=spans[target+1].first;else if(target) first=spans[target-1].second;
      updated=original;updated.erase(first,last-first);Number(updated,"frames_exported",samples.size()-1);
      if(sample.point_count) if(const auto count=data.get_optional<std::size_t>("points_exported")) if(*count>=sample.point_count) Number(updated,"points_exported",*count-sample.point_count);
      Parse(updated);entry.manifest=manifest;candidates.erase(manifest);
    }
  }
  const auto trash=entry.root/".lidar_studio_trash";
  if(fs::is_symlink(fs::symlink_status(trash))) throw std::runtime_error("Dataset trash must not be a symlink");
  fs::create_directories(trash);
  for(unsigned attempt=0;attempt<100;++attempt) {
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory=trash/(std::to_string(stamp)+"-"+std::to_string(ProcessId())+"-"+std::to_string(attempt));
    if(fs::create_directory(directory)) {entry.directory=directory;break;}
  }
  if(entry.directory.empty()) throw std::runtime_error("Cannot create sample recovery folder");
  std::vector<std::pair<fs::path,fs::path>> completed;
  try {
    for(const auto & path:candidates) entry.files.emplace_back(path,entry.directory/"files"/path.lexically_relative(entry.root));
    Tree receipt;receipt.put("schema","lidar_studio.dataset_trash");receipt.put("sample",frame.path.u8string());receipt.put("manifest",entry.manifest.u8string());Tree files;
    for(const auto & move:entry.files) {Tree file;file.put("original",move.first.u8string());file.put("stored",move.second.u8string());files.push_back({"",file});}
    receipt.add_child("files",files);std::ostringstream text;boost::property_tree::write_json(text,receipt);settings::AtomicWriteText(entry.directory/"receipt.json",text.str());
    if(!entry.manifest.empty()) {settings::AtomicWriteText(entry.directory/"manifest.before.json",original);settings::AtomicWriteText(entry.directory/"manifest.after.json",updated);}
    completed.reserve(entry.files.size());
    for(const auto & move:entry.files) {Move(move.first,move.second);completed.push_back(move);}
    if(!entry.manifest.empty()) {
      if(Read(entry.manifest)!=original) throw std::runtime_error("Dataset manifest changed while deleting; try again");
      settings::AtomicWriteText(entry.manifest,updated);
    }
    return entry;
  } catch(...) {
    const auto failure=std::current_exception();
    try {Rollback(completed);std::error_code ignored;fs::remove_all(entry.directory,ignored);}
    catch(const std::exception & error) {throw std::runtime_error("Could not finish deletion or rollback; recover files from "+entry.directory.u8string()+": "+error.what());}
    std::rethrow_exception(failure);
  }
}
void RestoreDatasetSample(const DatasetTrashEntry & entry) {
  if(!entry.manifest.empty() && Read(entry.manifest)!=Read(entry.directory/"manifest.after.json"))
    throw std::runtime_error("Dataset manifest changed after deletion; recovery would overwrite those changes");
  for(const auto & file:entry.files) {
    Inside(entry.root,file.first);Inside(entry.root,file.second);
    if(fs::exists(file.first) || fs::is_symlink(fs::symlink_status(file.first))) throw std::runtime_error("A new file uses this sample name; recovery will not overwrite it");
    if(!fs::is_regular_file(file.second) || fs::is_symlink(fs::symlink_status(file.second))) throw std::runtime_error("A sample recovery file is missing");
  }
  const auto original=entry.manifest.empty() ? std::string{} : Read(entry.directory/"manifest.before.json");
  std::vector<std::pair<fs::path,fs::path>> completed;
  try {
    completed.reserve(entry.files.size());
    for(const auto & file:entry.files) {Move(file.second,file.first);completed.emplace_back(file.second,file.first);}
    if(!entry.manifest.empty()) settings::AtomicWriteText(entry.manifest,original);
  } catch(...) {
    const auto failure=std::current_exception();
    try {Rollback(completed);} catch(const std::exception & error) {throw std::runtime_error("Recovery was interrupted; inspect "+entry.directory.u8string()+": "+error.what());}
    std::rethrow_exception(failure);
  }
  std::error_code ignored;fs::remove_all(entry.directory,ignored);
}
} // namespace lidar_viewer
