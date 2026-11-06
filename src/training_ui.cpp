#include "lidar_viewer/training_ui.hpp"
#include "lidar_viewer/training_job.hpp"
#include "lidar_viewer/file_dialog.hpp"
#include "lidar_viewer/config.hpp"
#include "lidar_viewer/platform.hpp"
#include "lidar_viewer/settings_io.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <imgui.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace lidar_viewer
{
namespace
{
namespace fs=std::filesystem;using Tree=boost::property_tree::ptree;
void Trace(const std::string & name) {
  if(!std::getenv("LIDAR_VIEWER_UI_TRACE") || ImGui::GetFrameCount()%2 || !ImGui::IsItemVisible()) return;
  const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
  std::cout<<"UI training_"<<name<<' '<<a.x<<' '<<a.y<<' '<<b.x<<' '<<b.y<<std::endl;
}
Tree Read(const fs::path & path) {Tree tree;std::ifstream file(path);if(file) boost::property_tree::read_json(file,tree);return tree;}
std::string Text(const Tree & tree) {std::ostringstream out;boost::property_tree::write_json(out,tree);return out.str();}
using PathBuffer=std::array<char,4096>;
void Set(PathBuffer & buffer,const std::string & value) {
  if(value.size()>=buffer.size()) throw std::runtime_error("Training path exceeds 4095 bytes");
  buffer.fill(0);std::copy(value.begin(),value.end(),buffer.begin());
}
}
struct TrainingWorkspace::Impl
{
  struct Recipe {std::string id,title,task;std::set<std::string> parameters;bool config{false};};
  std::vector<Recipe> recipes;int selected{0};
  PathBuffer python{},project{},dataset{},config{},output{};std::array<char,256> classes{};
  int epochs{30},max_points{4096},seed{42},batch_size{2},workers{2};float learning_rate{.001f},validation_fraction{.2f},yaw_range{25};bool allow_weak{false};
  FileDialog picker;PathBuffer * picker_target{nullptr};TrainingJob job;fs::path run;
  std::string error,log;bool checking{false},cancelled{false},initialized{false};double polled{-1};
  int progress_epoch{0};float progress_loss{0};
  Tree settings;fs::path settings_path=UserConfigurationDirectory()/"training.json";
  Tree Values() const {
    Tree value;value.put("adapter",recipes[static_cast<std::size_t>(selected)].id);
    for(const auto & entry:std::array<std::pair<const char *,const PathBuffer *>,5>{{{"python",&python},{"project",&project},{"dataset",&dataset},{"config",&config},{"output",&output}}}) value.put(entry.first,entry.second->data());
    value.put("epochs",epochs);value.put("max_points",max_points);value.put("seed",seed);value.put("batch_size",batch_size);value.put("workers",workers);
    value.put("learning_rate",learning_rate);value.put("validation_fraction",validation_fraction);value.put("yaw_range",yaw_range);value.put("allow_weak",allow_weak);value.put("classes",classes.data());return value;
  }
  void Save() {
    settings.put("selected",selected);settings.put_child("profiles."+recipes[static_cast<std::size_t>(selected)].id,Values());
    fs::create_directories(settings_path.parent_path());lidar_viewer::settings::AtomicWriteText(settings_path,Text(settings));
  }
  void Load() {
    const auto value=settings.get_child("profiles."+recipes[static_cast<std::size_t>(selected)].id,Tree{});
#ifdef _WIN32
    Set(python,value.get<std::string>("python","python"));
#else
    Set(python,value.get<std::string>("python","python3"));
#endif
    Set(project,value.get<std::string>("project",""));Set(dataset,value.get<std::string>("dataset",""));Set(config,value.get<std::string>("config",""));
    Set(output,value.get<std::string>("output",(UserHome()/"lidar_training_runs").u8string()));
    const auto names=value.get<std::string>("classes","person,car,bicycle,truck,bus");classes.fill(0);std::copy_n(names.begin(),std::min(names.size(),classes.size()-1),classes.begin());
    epochs=value.get<int>("epochs",30);max_points=value.get<int>("max_points",4096);seed=value.get<int>("seed",42);
    batch_size=value.get<int>("batch_size",2);workers=value.get<int>("workers",2);learning_rate=value.get<float>("learning_rate",.001f);
    validation_fraction=value.get<float>("validation_fraction",.2f);yaw_range=value.get<float>("yaw_range",25);allow_weak=value.get<bool>("allow_weak",false);
  }
  void Initialize() {
    if(initialized) return;
    recipes.clear();
    for(const auto & item:Read(ProjectResourcePath("python/lidar_training/adapters.json"))) {
      const auto & value=item.second;Recipe recipe;recipe.id=value.get<std::string>("id");recipe.title=value.get<std::string>("title");recipe.task=value.get<std::string>("task");recipe.config=value.get<bool>("uses_config",false);
      for(const auto & parameter:value.get_child("parameters")) recipe.parameters.insert(parameter.second.get_value<std::string>());
      recipes.push_back(std::move(recipe));
    }
    if(recipes.empty()) throw std::runtime_error("Training adapter profiles are missing");
    try {settings=Read(settings_path);selected=std::clamp(settings.get<int>("selected",0),0,static_cast<int>(recipes.size())-1);Load();}
    catch(const std::exception & saved_error) {error=std::string("Saved training settings could not be read: ")+saved_error.what();settings.clear();selected=0;Load();}
    initialized=true;
    // Reuse an explicitly configured local PyTorch environment when available.
    if(!project[0] && recipes[static_cast<std::size_t>(selected)].id=="litept") {
      std::ifstream detector(UserConfigurationDirectory()/"detector.cfg");std::string line,model;
      while(std::getline(detector,line)) {
        if(line.rfind("structure_python=",0)==0) Set(python,line.substr(17));
        if(line.rfind("structure_model_dir=",0)==0) model=line.substr(20);
      }
      if(!model.empty()) {
        auto checkpoint=fs::u8path(Read(fs::u8path(model)/"model.json").get<std::string>("source_checkpoint",""));
        for(int i=0;i<10 && !checkpoint.empty();++i) {
          if(fs::is_regular_file(checkpoint/"training/run.py")) {Set(project,checkpoint.u8string());break;}
          const auto parent=checkpoint.parent_path();if(parent==checkpoint) break;checkpoint=parent;
        }
      }
    }
  }
  void Start(bool validate) {
    if(job.Running()) return;
    Save();error.clear();cancelled=false;checking=validate;log.clear();progress_epoch=0;progress_loss=0;
    const auto parent=fs::u8path(output.data());if(parent.empty()) throw std::runtime_error("Choose an output parent folder");fs::create_directories(parent);
    const auto id=recipes[static_cast<std::size_t>(selected)].id;
    for(std::size_t number=1;;++number) {std::ostringstream name;name<<id<<'_'<<std::setw(3)<<std::setfill('0')<<number;
      run=parent/name.str();if(fs::create_directory(run)) break;}
    lidar_viewer::settings::AtomicWriteText(run/"request.json",Text(Values()));
    job.Start(fs::u8path(python.data()),ProjectResourcePath("python/studio_train.py"),run/"request.json",run/"training.log",validate);polled=-1;
  }
};
TrainingWorkspace::TrainingWorkspace():impl_(std::make_unique<Impl>()) {}
TrainingWorkspace::~TrainingWorkspace()=default;
bool TrainingWorkspace::Running() const {return impl_->job.Running();}
void TrainingWorkspace::Draw(bool & open,const fs::path & active_dataset,float scale)
{
  auto & state=*impl_;state.job.Poll();
  if(!open) return;
  try {
    state.Initialize();
    if(!state.dataset[0] && !active_dataset.empty()) Set(state.dataset,active_dataset.u8string());
    if(const auto result=state.picker.Poll()) {
      if(!result->error.empty()) state.error=result->error;
      else if(!result->path.empty() && state.picker_target) Set(*state.picker_target,result->path.u8string());
      state.picker_target=nullptr;
    }
  } catch(const std::exception & error) {state.error=error.what();}
  ImGui::SetNextWindowSize({std::min(820*scale,ImGui::GetIO().DisplaySize.x-32*scale),std::min(760*scale,ImGui::GetIO().DisplaySize.y-32*scale)},ImGuiCond_FirstUseEver);
  if(!ImGui::Begin("Training",&open,ImGuiWindowFlags_NoCollapse)) {ImGui::End();return;}
  if(state.recipes.empty()) {ImGui::TextWrapped("%s",state.error.c_str());ImGui::End();return;}
  const auto & recipe=state.recipes[static_cast<std::size_t>(state.selected)];
  const float reserve=(state.run.empty() ? 90.f : 250.f)*scale;
  ImGui::BeginChild("TrainingSetup",{0,std::max(120*scale,ImGui::GetContentRegionAvail().y-reserve)});
  ImGui::BeginDisabled(state.job.Running() || state.picker.Active());
  ImGui::SetNextItemWidth(-1);
  if(ImGui::BeginCombo("##TrainingAdapter",recipe.title.c_str())) {
    for(std::size_t i=0;i<state.recipes.size();++i) {
      if(ImGui::Selectable(state.recipes[i].title.c_str(),i==static_cast<std::size_t>(state.selected))) {
        try {state.Save();state.selected=static_cast<int>(i);state.Load();state.error.clear();state.run.clear();state.log.clear();state.progress_epoch=0;state.cancelled=false;}
        catch(const std::exception & error) {state.error=error.what();}
      }
      Trace("adapter_"+state.recipes[i].id);
    }
    ImGui::EndCombo();
  }
  Trace("adapter");ImGui::TextDisabled("%s",recipe.task.c_str());ImGui::Spacing();
  const auto path_field=[&](const char * label,const char * id,PathBuffer & value,bool directory,bool data=false) {
    ImGui::TextUnformatted(label);ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x-90*scale);
    ImGui::InputText((std::string("##Train_")+id).c_str(),value.data(),value.size());Trace(id);ImGui::SameLine();ImGui::PushID(id);
    const auto choose=[&](bool folder) {try {state.picker_target=&value;auto initial=fs::u8path(value.data());
      if(!fs::is_directory(initial)) initial=initial.parent_path();
      if(!fs::is_directory(initial)) initial=UserHome();
      if(folder) state.picker.OpenDirectory(initial);else state.picker.OpenFile(initial);}
      catch(const std::exception & error) {state.error=error.what();}};
    if(ImGui::Button("Browse...",{82*scale,0})) {if(data && !recipe.config) ImGui::OpenPopup("Input kind");else choose(directory);}
    Trace(std::string(id)+"_browse");
    if(ImGui::BeginPopup("Input kind")) {if(ImGui::MenuItem("Annotated dataset folder")) choose(true);if(ImGui::MenuItem("Prepared split JSON")) choose(false);ImGui::EndPopup();}
    ImGui::PopID();
  };
  ImGui::PushID(recipe.id.c_str());
  ImGui::SetNextItemOpen(!state.project[0] || (recipe.config && !state.config[0]),ImGuiCond_Once);
  if(ImGui::CollapsingHeader("Environment (Python / trainer)")) {
    path_field("Training Python","python",state.python,false);path_field("Trainer project","project",state.project,true);
    if(recipe.config) path_field("CenterPoint base config (.py)","config",state.config,false);
  }
  ImGui::PopID();
  path_field("Training data","dataset",state.dataset,true,true);
  if(recipe.config) {ImGui::TextUnformatted("Classes (annotation names, comma separated)");ImGui::SetNextItemWidth(-1);ImGui::InputText("##TrainingClasses",state.classes.data(),state.classes.size());Trace("classes");}
  path_field("Output parent folder","output",state.output,true);ImGui::Spacing();
  if(ImGui::BeginTable("TrainingParameters",2,ImGuiTableFlags_SizingStretchSame)) {
    const auto integer=[&](const char * key,const char * label,int & value) {if(!recipe.parameters.count(key)) return;ImGui::TableNextColumn();ImGui::TextUnformatted(label);ImGui::SetNextItemWidth(-1);ImGui::InputInt((std::string("##TrainParam_")+key).c_str(),&value);Trace(key);};
    integer("epochs","Epochs",state.epochs);integer("batch_size","Batch size",state.batch_size);integer("max_points","Points per sample",state.max_points);integer("workers","Data workers",state.workers);integer("seed","Random seed",state.seed);
    if(recipe.parameters.count("learning_rate")) {ImGui::TableNextColumn();ImGui::TextUnformatted("Learning rate");ImGui::SetNextItemWidth(-1);ImGui::InputFloat("##TrainLR",&state.learning_rate,0,0,"%.6f");Trace("learning_rate");}
    if(recipe.parameters.count("yaw_range")) {ImGui::TableNextColumn();ImGui::TextUnformatted("Yaw augmentation (degrees)");ImGui::SetNextItemWidth(-1);ImGui::InputFloat("##TrainYaw",&state.yaw_range);Trace("yaw_range");}
    ImGui::TableNextColumn();ImGui::TextUnformatted("Validation fraction (folder input)");ImGui::SetNextItemWidth(-1);ImGui::InputFloat("##TrainValidation",&state.validation_fraction,0,0,"%.2f");Trace("validation_fraction");ImGui::EndTable();
  }
  if(!recipe.config) {ImGui::Checkbox("Allow existing weak ROI labels",&state.allow_weak);Trace("allow_weak");ImGui::TextDisabled("Optimizer and one-scan batches follow the selected project recipe.");}
  ImGui::EndDisabled();ImGui::EndChild();ImGui::Spacing();
  if(!state.job.Running()) {
    ImGui::BeginDisabled(state.picker.Active());
    if(ImGui::Button("Check setup",{130*scale,0})) try {state.Start(true);} catch(const std::exception & error) {state.error=error.what();}
    Trace("validate");ImGui::SameLine();
    if(ImGui::Button("Start training",{150*scale,0})) try {state.Start(false);} catch(const std::exception & error) {state.error=error.what();}
    Trace("start");ImGui::EndDisabled();
  } else {
    if(ImGui::Button(state.job.Stopping() ? "Stopping..." : "Stop training",{150*scale,0})) {state.job.Cancel();state.cancelled=true;}
    Trace("stop");ImGui::SameLine();ImGui::TextUnformatted(state.checking ? "Checking environment and labels..." : "Training is running");
    ImGui::SameLine();if(ImGui::Button("Hide")) open=false;Trace("hide");
  }
  if(!state.run.empty()) if(const auto code=state.job.ExitCode()) ImGui::TextColored(*code==0 ? ImVec4(.4f,.85f,.65f,1) : ImVec4(1,.65f,.4f,1),"%s",state.cancelled ? "Stopped" : *code==0 ? (state.checking ? "Setup is ready" : "Training completed") : "Job failed - see the log below");
  if(!state.error.empty()) ImGui::TextWrapped("%s",state.error.c_str());
  if(!state.run.empty()) {
    ImGui::TextWrapped("Run: %s",state.run.u8string().c_str());
    if(ImGui::GetTime()-state.polled>.25) {
      state.polled=ImGui::GetTime();state.log=state.job.LogTail();
      try {const auto progress=Read(state.run/"progress.json");state.progress_epoch=progress.get<int>("epoch",state.progress_epoch);state.progress_loss=progress.get<float>("train_loss",state.progress_loss);} catch(...) {}
    }
    if(state.progress_epoch>0) {ImGui::ProgressBar(std::min(1.f,static_cast<float>(state.progress_epoch)/std::max(1,state.epochs)),{-1,0});ImGui::Text("Epoch %d / %d  |  loss %.4f",state.progress_epoch,state.epochs,state.progress_loss);}
    ImGui::BeginChild("TrainingLog",{0,std::max(80*scale,ImGui::GetContentRegionAvail().y-8*scale)},ImGuiChildFlags_Borders);
    ImGui::TextUnformatted(state.log.c_str());if(ImGui::GetScrollY()>=ImGui::GetScrollMaxY()-20*scale) ImGui::SetScrollHereY(1);ImGui::EndChild();
  }
  ImGui::End();if(!open) try {state.Save();} catch(...) {}
}
}  // namespace lidar_viewer
