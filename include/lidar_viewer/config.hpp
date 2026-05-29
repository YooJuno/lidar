#pragma once

#include "lidar_viewer/algorithms/pipeline.hpp"

#include <filesystem>

namespace lidar_viewer
{
namespace fs = std::filesystem;

algorithms::PipelineConfig LoadPipelineConfig(const fs::path & path);
}  // namespace lidar_viewer
