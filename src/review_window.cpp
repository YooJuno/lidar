#include "lidar_viewer/review_window.hpp"
#include "lidar_viewer/platform.hpp"
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
extern char ** environ;
#endif

namespace lidar_viewer {
struct ReviewWindowProcess::Impl {
#ifdef _WIN32
  HANDLE process{nullptr};
#else
  pid_t pid{-1};
#endif
};
ReviewWindowProcess::ReviewWindowProcess():impl_(std::make_unique<Impl>()) {}
ReviewWindowProcess::~ReviewWindowProcess() {
#ifdef _WIN32
  if(impl_->process) CloseHandle(impl_->process);
#else
  if(impl_->pid>0) {
    const auto pid=impl_->pid;
    std::thread([pid] {while(waitpid(pid,nullptr,0)<0 && errno==EINTR) {}}).detach();
  }
#endif
}
bool ReviewWindowProcess::Open(const std::filesystem::path & executable,const std::filesystem::path & log) {
  Poll();
  std::filesystem::create_directories(log.parent_path());
#ifdef _WIN32
  if(impl_->process) {
    if(WaitForSingleObject(impl_->process,0)==WAIT_TIMEOUT) return false;
    CloseHandle(impl_->process);impl_->process=nullptr;
  }
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
  HANDLE output=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
  HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
  if(output==INVALID_HANDLE_VALUE || input==INVALID_HANDLE_VALUE) {
    if(output!=INVALID_HANDLE_VALUE) CloseHandle(output);
    if(input!=INVALID_HANDLE_VALUE) CloseHandle(input);
    throw std::runtime_error("Cannot open point review log");
  }
  std::wstring command=L"\""+executable.wstring()+L"\" --point-review";
  STARTUPINFOW start{};start.cb=sizeof(start);start.dwFlags=STARTF_USESTDHANDLES;
  start.hStdInput=input;start.hStdOutput=output;start.hStdError=output;PROCESS_INFORMATION process{};
  const bool ok=CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP,nullptr,nullptr,&start,&process)!=0;
  CloseHandle(output);CloseHandle(input);
  if(!ok) throw std::runtime_error("Cannot open point cloud review window");
  CloseHandle(process.hThread);impl_->process=process.hProcess;
#else
  if(impl_->pid>0) {
    const auto done=waitpid(impl_->pid,nullptr,WNOHANG);
    if(done==0 || (done<0 && errno==EINTR)) return false;
    impl_->pid=-1;
  }
  const int output=open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600),input=open("/dev/null",O_RDONLY|O_CLOEXEC);
  if(output<0 || input<0) {if(output>=0) close(output);if(input>=0) close(input);throw std::runtime_error("Cannot open point review log");}
  posix_spawn_file_actions_t actions;posix_spawnattr_t attributes;
  int error=posix_spawn_file_actions_init(&actions);const bool have_actions=!error;
  const int attribute_error=posix_spawnattr_init(&attributes);const bool have_attributes=!attribute_error;if(!error) error=attribute_error;
  if(!error) error=posix_spawn_file_actions_adddup2(&actions,input,STDIN_FILENO);
  if(!error) error=posix_spawn_file_actions_adddup2(&actions,output,STDOUT_FILENO);
  if(!error) error=posix_spawn_file_actions_adddup2(&actions,output,STDERR_FILENO);
  if(!error) error=posix_spawn_file_actions_addclose(&actions,input);
  if(!error) error=posix_spawn_file_actions_addclose(&actions,output);
  if(!error) error=posix_spawnattr_setflags(&attributes,POSIX_SPAWN_SETPGROUP);
  if(!error) error=posix_spawnattr_setpgroup(&attributes,0);
  std::string path=executable.string(),option="--point-review";char * argv[]{path.data(),option.data(),nullptr};pid_t pid=-1;
  if(!error) error=posix_spawn(&pid,path.c_str(),&actions,&attributes,argv,environ);
  if(have_actions) posix_spawn_file_actions_destroy(&actions);
  if(have_attributes) posix_spawnattr_destroy(&attributes);
  close(input);close(output);
  if(error) throw std::runtime_error("Cannot open point cloud review window");
  impl_->pid=pid;
#endif
  return true;
}
void ReviewWindowProcess::Poll() {
#ifdef _WIN32
  if(impl_->process && WaitForSingleObject(impl_->process,0)!=WAIT_TIMEOUT) {CloseHandle(impl_->process);impl_->process=nullptr;}
#else
  if(impl_->pid>0) {const auto done=waitpid(impl_->pid,nullptr,WNOHANG);if(done>0 || (done<0 && errno==ECHILD)) impl_->pid=-1;}
#endif
}
}
