#include "lidar_viewer/config.hpp"
#include "lidar_viewer/data_io.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "lidar_viewer/platform.hpp"

namespace
{
void Check(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

void Write(const fs::path & path, const std::string & text)
{
  std::ofstream stream(path);
  stream << text;
}
}  // namespace

int main()
{
  const auto original = fs::current_path();
  const auto temporary = fs::temp_directory_path() / ("lidar-config-data-" + std::to_string(lidar_viewer::ProcessId()));
  try {
    fs::create_directories(temporary);
    fs::current_path(temporary);
    // First resolution happens away from the checkout, including with a
    // conflicting local configuration that must not override project profiles.
    fs::create_directory("config");
    Write("config/algorithms.yaml", "ground_removal:\n  mode: wrong\n");
    const auto resource = lidar_viewer::ProjectResourcePath("config/algorithms.yaml");
    Check(resource == fs::path(LIDAR_VIEWER_SOURCE_DIR) / "config/algorithms.yaml",
      "Resource lookup depends on the working directory");
    const auto config_path = temporary / "test.yaml";
    Write(config_path, "ground_removal:\n  mode: fast_grid\n  enabled: true\nclustering:\n  min_cluster_size: 12\ndetection:\n  classes: 8\n  model: short_range\n");
    const auto config = lidar_viewer::LoadPipelineConfig(config_path);
    Check(config.ground_removal.enabled && config.ground_removal.mode == "fast_grid" &&
      config.clustering.min_cluster_size == 12 && config.detection.classes==8 && config.detection.model=="short_range", "Valid algorithm configuration changed");
    for (const auto & contents : {
      "ground_removal:\n  distance_threshold_m: nan\n",
      "ground_removal:\n  distance_threshold_m: inf\n",
      "ground_removal:\n  distance_threshold_m: 0.2junk\n",
      "clustering:\n  min_cluster_size: 12junk\n",
      "clustering:\n  enabled: maybe\n",
      "detection:\n  model: bogus\n",
      "detection:\n  classes: -1\n",
      "detection:\n  classes: 32\n"}) {
      Write(config_path, contents);
      bool rejected = false;
      try { lidar_viewer::LoadPipelineConfig(config_path); }
      catch (const std::exception & error) {
        rejected = std::string(error.what()).find(":2:") != std::string::npos;
      }
      Check(rejected, "Malformed configuration was accepted or lacks line diagnostics");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const KittiPoint samples[] = {{1, 2, 3, nan}, {nan, 0, 0, 1}, {0, inf, 0, 1}, {-1, -2, -3, 10}};
    const auto bin = temporary / "points.bin";
    {
      std::ofstream stream(bin, std::ios::binary);
      stream.write(reinterpret_cast<const char *>(samples), sizeof(samples));
    }
    const auto frame = LoadFrame(ResolveInputFrames(bin).front());
    Check(frame.points.size() == 2 && std::isfinite(frame.radius) && frame.center.x == 0 &&
      frame.points.front().intensity == 0 && frame.points.back().intensity == 1,
      "Invalid BIN samples contaminated bounds or intensity normalization");
    fs::current_path(original);
    fs::remove_all(temporary);
    std::cout << "config_data_smoke: all checks passed\n";
    return 0;
  } catch (const std::exception & error) {
    fs::current_path(original);
    fs::remove_all(temporary);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
