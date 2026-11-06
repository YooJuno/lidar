#include "lidar_viewer/training_job.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

int main(int argc,char ** argv)
{
  namespace fs=std::filesystem;
  const auto root=fs::temp_directory_path()/("studio-job-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    if(argc!=2) throw std::runtime_error("Python argument required");fs::create_directories(root);
    const auto script=root/"worker 'literal'.py",request=root/"request.json",log=root/"training.log";
    {std::ofstream file(script);file<<"import sys,time,subprocess\nprint('literal path worker',flush=True)\nif '--validate' not in sys.argv: time.sleep(60)\n";}
    {std::ofstream file(request);file<<"{}";}
    lidar_viewer::TrainingJob job;job.Start(fs::u8path(argv[1]),script,request,log,true);
    auto until=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    while(job.Running() && std::chrono::steady_clock::now()<until) {job.Poll();std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    if(job.Running() || job.ExitCode()!=0 || job.LogTail().find("literal path worker")==std::string::npos) throw std::runtime_error("Async validation/log or literal arguments failed");
    job.Start(fs::u8path(argv[1]),script,request,log,false);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));job.Cancel();until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(job.Running() && std::chrono::steady_clock::now()<until) {job.Poll();std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    if(job.Running() || job.ExitCode()==0) throw std::runtime_error("Training stop did not finish");
    job.Start(fs::u8path(argv[1]),script,request,log,true);
    until=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    while(job.Running() && std::chrono::steady_clock::now()<until) {job.Poll();std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    if(job.Running() || job.ExitCode()!=0) throw std::runtime_error("Job was not reusable after stop");
    fs::remove_all(root);std::cout<<"Training job: async execution, logs, cancellation and reuse passed\n";return 0;
  } catch(const std::exception & error) {fs::remove_all(root);std::cerr<<error.what()<<'\n';return 1;}
}
