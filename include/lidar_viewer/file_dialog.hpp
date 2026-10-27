#pragma once

#include <filesystem>
#include <optional>
#include <memory>
#include <string>

namespace lidar_viewer
{
struct FileDialogResult
{
  std::filesystem::path path;
  std::string error;
  // Empty path and error mean the user cancelled.
};

// Native GTK file chooser via zenity. No shell; paths are literal arguments.
// Polling keeps the viewer responsive. Destruction closes a pending chooser.
class FileDialog
{
public:
  FileDialog() = default;
  ~FileDialog();
  FileDialog(const FileDialog &) = delete;
  FileDialog & operator=(const FileDialog &) = delete;
  void OpenBag(const std::filesystem::path & initial_folder);
  bool Active() const;
  std::optional<FileDialogResult> Poll();
  void Cancel() noexcept;

private:
#ifdef _WIN32
  struct NativeState;
  std::shared_ptr<NativeState> native_;
#else
  void ReadOutput();
  int child_{-1};
  int output_fd_{-1};
  std::string output_;
#endif
};
}  // namespace lidar_viewer
