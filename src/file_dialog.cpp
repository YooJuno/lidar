#include "lidar_viewer/file_dialog.hpp"

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char ** environ;

namespace lidar_viewer
{
FileDialog::~FileDialog() {Cancel();}
bool FileDialog::Active() const {return child_ > 0;}

void FileDialog::OpenBag(const std::filesystem::path & initial_folder)
{
  if (Active()) return;
  std::string filename = initial_folder.string();
  if (!filename.empty() && filename.back() != '/') filename += '/';
#ifdef __APPLE__
  const std::string script="on run argv\ntry\nset picked to choose file with prompt \"Open LiDAR data\" default location (POSIX file (item 1 of argv))\nreturn POSIX path of picked\non error detail number code\nif code is -128 then return \"\"\nerror detail number code\nend try\nend run";
  std::vector<std::string> arguments{"osascript","-e",script, filename.empty() ? "/" : filename};
#else
  std::vector<std::string> arguments{"zenity", "--file-selection", "--title=Open LiDAR data",
    "--file-filter=LiDAR data | metadata.yaml *.db3 *.mcap *.bin",
    "--file-filter=All files | *", "--filename=" + filename};
#endif
  std::vector<char *> argv;
  for (auto & argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0)
    throw std::system_error(errno, std::generic_category(), "Cannot create file chooser pipe");
  if (fcntl(pipe_fds[0], F_SETFD, FD_CLOEXEC)==-1 || fcntl(pipe_fds[1], F_SETFD, FD_CLOEXEC)==-1 ||
      fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK) == -1) {
    const int error = errno;
    close(pipe_fds[0]); close(pipe_fds[1]);
    throw std::system_error(error, std::generic_category(), "Cannot configure file chooser pipe");
  }
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  const bool initialized = error == 0;
  if (!error) error = posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  if (!error) error = posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
  if (!error) error = posix_spawn_file_actions_addclose(&actions, pipe_fds[1]);
  pid_t child = -1;
  if (!error) error = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
  if (initialized) posix_spawn_file_actions_destroy(&actions);
  close(pipe_fds[1]);
  if (error) {
    close(pipe_fds[0]);
    throw std::system_error(error, std::generic_category(),
      "Cannot open native file chooser");
  }
  child_ = child;
  output_fd_ = pipe_fds[0];
  output_.clear();
}

void FileDialog::ReadOutput()
{
  char buffer[4096];
  for (;;) {
    const auto count = read(output_fd_, buffer, sizeof(buffer));
    if (count > 0) {
      output_.append(buffer, static_cast<std::size_t>(count));
      if (output_.size() > 65536) throw std::runtime_error("File chooser returned an excessively long path");
    } else if (count == 0 || errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    } else if (errno != EINTR) {
      throw std::system_error(errno, std::generic_category(), "Cannot read file chooser result");
    }
  }
}

std::optional<FileDialogResult> FileDialog::Poll()
{
  if (!Active()) return std::nullopt;
  try {
    ReadOutput();
    int status = 0;
    const auto result = waitpid(child_, &status, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) return std::nullopt;
    if (result < 0) throw std::system_error(errno, std::generic_category(), "Cannot wait for file chooser");
    child_ = -1;
    ReadOutput();
    close(output_fd_);
    output_fd_ = -1;
#ifndef __APPLE__
    if (WIFEXITED(status) && WEXITSTATUS(status) == 1) return FileDialogResult{};
#endif
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
      return FileDialogResult{{}, "File chooser failed. Select Open file to try again."};
    if (!output_.empty() && output_.back() == '\n') output_.pop_back();
    if (output_.empty()) {
#ifdef __APPLE__
      return FileDialogResult{};
#else
      return FileDialogResult{{}, "File chooser returned no file"};
#endif
    }
    if (output_.find('\0') != std::string::npos || output_.find('\n') != std::string::npos ||
        output_.find('\r') != std::string::npos)
      return FileDialogResult{{}, "File paths with line breaks or NUL are not supported"};
    return FileDialogResult{std::filesystem::path(output_), {}};
  } catch (...) {
    Cancel();
    throw;
  }
}

void FileDialog::Cancel() noexcept
{
  if (child_ > 0) {
    kill(child_, SIGKILL);
    while (waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {}
    child_ = -1;
  }
  if (output_fd_ >= 0) {close(output_fd_); output_fd_ = -1;}
  output_.clear();
}
}  // namespace lidar_viewer
