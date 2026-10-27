#include "lidar_viewer/file_dialog.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace
{
void Check(bool value, const char * message)
{
  if (!value) throw std::runtime_error(message);
}
lidar_viewer::FileDialogResult Await(lidar_viewer::FileDialog & dialog)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto result = dialog.Poll()) return *result;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  throw std::runtime_error("File chooser did not finish");
}
}

int main(int argc, char ** argv)
{
  // Spawn this executable through a private zenity alias to validate the
  // process protocol without opening windows or driving the user's desktop.
  if (argc > 1 && std::string(argv[1]) == "--file-selection") {
    const std::string mode = std::getenv("LIDAR_TEST_CHOOSER_MODE");
    if (mode == "cancel") return 1;
    if (mode == "error") return 2;
    if (mode == "wait") {std::this_thread::sleep_for(std::chrono::seconds(10)); return 1;}
    bool literal = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--filename=/tmp/bag 'quoted' $(false); space/한글/") literal = true;
    }
    if (!literal) return 2;
    std::cout << "/tmp/bag 'quoted' $(false); space/한글/metadata.yaml\n";
    return 0;
  }

  const auto root = fs::temp_directory_path() / ("lidar-file-dialog-test-" + std::to_string(getpid()));
  const std::string previous_path = std::getenv("PATH") ? std::getenv("PATH") : "";
  try {
    fs::create_directories(root);
    fs::create_symlink(fs::canonical("/proc/self/exe"), root / "zenity");
    setenv("PATH", root.c_str(), 1);
    setenv("LIDAR_TEST_CHOOSER_MODE", "accept", 1);
    lidar_viewer::FileDialog dialog;
    Check(!dialog.Poll(), "inactive poll produced a result");
    dialog.OpenBag("/tmp/bag 'quoted' $(false); space/한글");
    const auto selected = Await(dialog);
    Check(selected.error.empty() && selected.path == "/tmp/bag 'quoted' $(false); space/한글/metadata.yaml",
      "chooser changed literal path arguments or output");
    Check(!dialog.Active(), "completed chooser still active");
    setenv("LIDAR_TEST_CHOOSER_MODE", "cancel", 1);
    dialog.OpenBag({});
    const auto cancelled = Await(dialog);
    Check(cancelled.path.empty() && cancelled.error.empty(), "cancel was not benign");
    setenv("LIDAR_TEST_CHOOSER_MODE", "error", 1);
    dialog.OpenBag({});
    Check(!Await(dialog).error.empty(), "chooser failure was ignored");
    setenv("LIDAR_TEST_CHOOSER_MODE", "wait", 1);
    dialog.OpenBag({});
    Check(dialog.Active() && !dialog.Poll(), "running chooser blocked or ended early");
    const auto started = std::chrono::steady_clock::now();
    dialog.Cancel();
    Check(!dialog.Active() && std::chrono::steady_clock::now() - started < std::chrono::seconds(1),
      "closing a pending chooser hung");
    fs::remove(root / "zenity");
    bool missing_rejected = false;
    try {dialog.OpenBag({});} catch (const std::exception &) {missing_rejected = true;}
    Check(missing_rejected && !dialog.Active(), "missing chooser was not reported");
    fs::remove_all(root);
    setenv("PATH", previous_path.c_str(), 1);
    unsetenv("LIDAR_TEST_CHOOSER_MODE");
    std::cout << "File chooser literal paths, cancel, failure and lifecycle passed\n";
    return 0;
  } catch (const std::exception & error) {
    fs::remove_all(root);
    setenv("PATH", previous_path.c_str(), 1);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
