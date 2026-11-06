#include "lidar_viewer/training_job.hpp"
#include "lidar_viewer/platform.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>
#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
extern char ** environ;
#endif

namespace lidar_viewer
{
struct TrainingJob::Impl
{
  std::filesystem::path log;
  std::optional<int> exit;
  bool stopping{false};std::chrono::steady_clock::time_point stopped_at;
#ifdef _WIN32
  HANDLE process{nullptr},job{nullptr};
#else
  pid_t pid{-1};
#endif
};
TrainingJob::TrainingJob():impl_(std::make_unique<Impl>()) {}
TrainingJob::~TrainingJob()
{
  if(!Running()) return;
#ifdef _WIN32
  TerminateJobObject(impl_->job,1);WaitForSingleObject(impl_->process,5000);
  CloseHandle(impl_->process);CloseHandle(impl_->job);
#else
  kill(-impl_->pid,SIGKILL);while(waitpid(impl_->pid,nullptr,0)<0 && errno==EINTR) {}
#endif
}
bool TrainingJob::Running() const {
#ifdef _WIN32
  return impl_->process!=nullptr;
#else
  return impl_->pid>0;
#endif
}
bool TrainingJob::Stopping() const {return impl_->stopping && Running();}
std::optional<int> TrainingJob::ExitCode() const {return impl_->exit;}
void TrainingJob::Start(const std::filesystem::path & python,const std::filesystem::path & script,
  const std::filesystem::path & request,const std::filesystem::path & log,bool validate_only)
{
  if(Running()) throw std::runtime_error("A training job is already running");
  if(!std::filesystem::is_regular_file(script) || !std::filesystem::is_regular_file(request))
    throw std::runtime_error("Training script or request is missing");
#ifdef _WIN32
  const auto quote=[](const std::wstring & value) {
    std::wstring out=L"\"";std::size_t slashes=0;
    for(const auto c:value) {if(c==L'\\') {++slashes;continue;}
      out.append(c==L'"' ? slashes*2+1 : slashes,L'\\');slashes=0;out+=c;}
    out.append(slashes*2,L'\\');return out+L'"';
  };
  std::wstring command=quote(python.wstring())+L" -u "+quote(script.wstring())+L" --request "+quote(request.wstring());
  if(validate_only) command+=L" --validate";
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
  HANDLE output=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
  HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
  if(output==INVALID_HANDLE_VALUE || input==INVALID_HANDLE_VALUE) {
    if(output!=INVALID_HANDLE_VALUE) CloseHandle(output);if(input!=INVALID_HANDLE_VALUE) CloseHandle(input);
    throw std::runtime_error("Cannot open training log");
  }
  HANDLE job=CreateJobObjectW(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
  info.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if(!job || !SetInformationJobObject(job,JobObjectExtendedLimitInformation,&info,sizeof(info))) {
    CloseHandle(output);CloseHandle(input);if(job) CloseHandle(job);throw std::runtime_error("Cannot create training process group");
  }
  STARTUPINFOW start{};start.cb=sizeof(start);start.dwFlags=STARTF_USESTDHANDLES;
  start.hStdInput=input;start.hStdOutput=output;start.hStdError=output;PROCESS_INFORMATION child{};
  const bool created=CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&start,&child)!=0;
  CloseHandle(output);CloseHandle(input);
  if(!created || !AssignProcessToJobObject(job,child.hProcess)) {
    if(created) {TerminateProcess(child.hProcess,1);CloseHandle(child.hProcess);CloseHandle(child.hThread);}
    CloseHandle(job);throw std::runtime_error("Cannot launch the selected training Python");
  }
  ResumeThread(child.hThread);CloseHandle(child.hThread);impl_->process=child.hProcess;impl_->job=job;
#else
  const int output=open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600),input=open("/dev/null",O_RDONLY|O_CLOEXEC);
  if(output<0 || input<0) {if(output>=0) close(output);if(input>=0) close(input);throw std::runtime_error("Cannot open training log");}
  posix_spawn_file_actions_t actions;posix_spawnattr_t attributes;
  int rc=posix_spawn_file_actions_init(&actions);const bool actions_ok=rc==0;
  if(!rc) rc=posix_spawn_file_actions_adddup2(&actions,input,STDIN_FILENO);
  if(!rc) rc=posix_spawn_file_actions_adddup2(&actions,output,STDOUT_FILENO);
  if(!rc) rc=posix_spawn_file_actions_adddup2(&actions,output,STDERR_FILENO);
  const int ar=posix_spawnattr_init(&attributes);const bool attributes_ok=ar==0;if(!rc) rc=ar;
  if(!rc) rc=posix_spawnattr_setflags(&attributes,POSIX_SPAWN_SETPGROUP);
  if(!rc) rc=posix_spawnattr_setpgroup(&attributes,0);
  std::vector<std::string> arguments{python.string(),"-u",script.string(),"--request",request.string()};
  if(validate_only) arguments.push_back("--validate");
  std::vector<char *> argv;for(auto & arg:arguments) argv.push_back(arg.data());argv.push_back(nullptr);
  pid_t child=-1;if(!rc) rc=posix_spawnp(&child,argv[0],&actions,&attributes,argv.data(),environ);
  if(actions_ok) posix_spawn_file_actions_destroy(&actions);
  if(attributes_ok) posix_spawnattr_destroy(&attributes);
  close(output);close(input);if(rc) throw std::runtime_error("Cannot launch the selected training Python");
  impl_->pid=child;
#endif
  impl_->log=log;impl_->exit.reset();impl_->stopping=false;
}
void TrainingJob::Cancel()
{
  if(!Running() || impl_->stopping) return;
  impl_->stopping=true;impl_->stopped_at=std::chrono::steady_clock::now();
#ifdef _WIN32
  TerminateJobObject(impl_->job,1);
#else
  kill(-impl_->pid,SIGTERM);
#endif
}
void TrainingJob::Poll()
{
  if(!Running()) return;
#ifdef _WIN32
  if(WaitForSingleObject(impl_->process,0)==WAIT_TIMEOUT) return;
  DWORD code=1;GetExitCodeProcess(impl_->process,&code);impl_->exit=static_cast<int>(code);
  CloseHandle(impl_->process);CloseHandle(impl_->job);impl_->process=nullptr;impl_->job=nullptr;
#else
  if(impl_->stopping && std::chrono::steady_clock::now()-impl_->stopped_at>std::chrono::seconds(2)) kill(-impl_->pid,SIGKILL);
  int status=0;const auto result=waitpid(impl_->pid,&status,WNOHANG);
  if(result==0 || (result<0 && errno==EINTR)) return;
  impl_->exit=result<0 ? 1 : WIFEXITED(status) ? WEXITSTATUS(status) : 128+WTERMSIG(status);
  // A trainer must not leave dataloader grandchildren running after it exits.
  kill(-impl_->pid,SIGKILL);impl_->pid=-1;
#endif
}
std::string TrainingJob::LogTail(std::size_t limit) const
{
  std::ifstream file(impl_->log,std::ios::binary|std::ios::ate);if(!file) return {};
  const auto size=file.tellg();if(size<0) return {};
  const auto length=std::min<std::uint64_t>(static_cast<std::uint64_t>(size),std::min<std::size_t>(limit,65536));
  file.seekg(static_cast<std::streamoff>(static_cast<std::uint64_t>(size)-length));
  std::string result(static_cast<std::size_t>(length),' ');file.read(result.data(),static_cast<std::streamsize>(length));return result;
}
}  // namespace lidar_viewer
