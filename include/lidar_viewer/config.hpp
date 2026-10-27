#pragma once

#include "lidar_viewer/algorithms/pipeline.hpp"

#include <filesystem>

namespace lidar_viewer
{
namespace fs = std::filesystem;

algorithms::PipelineConfig LoadPipelineConfig(const fs::path & path);
fs::path ProjectResourcePath(const fs::path & relative);
}  // namespace lidar_viewer
