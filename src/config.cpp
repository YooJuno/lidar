#include "lidar_viewer/config.hpp"
#include "lidar_viewer/settings_io.hpp"
#include "lidar_viewer/platform.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <stdexcept>

namespace lidar_viewer
{
namespace
{
std::string Trim(std::string value)
{
  const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

bool ParseBool(const std::string & value)
{
  const std::string normalized = Trim(value);
  if (normalized == "true" || normalized == "True" || normalized == "TRUE" || normalized == "1") return true;
  if (normalized == "false" || normalized == "False" || normalized == "FALSE" || normalized == "0") return false;
  throw std::invalid_argument("Expected true or false");
}

float ParseFloat(const std::string & value)
{
  return settings::ParseFiniteFloat(Trim(value));
}

int ParseInt(const std::string & value)
{
  return settings::ParseStrictInt(Trim(value));
}

void ApplyGroundRemovalValue(
  algorithms::GroundRemovalParams & params,
  const std::string & key,
  const std::string & value)
{
  if (key == "enabled") {
    params.enabled = ParseBool(value);
  } else if (key == "use_cuda") {
    params.use_cuda = ParseBool(value);
  } else if (key == "mode") {
    params.mode = Trim(value);
  } else if (key == "distance_threshold_m") {
    params.distance_threshold_m = ParseFloat(value);
  } else if (key == "final_distance_threshold_m") {
    params.final_distance_threshold_m = ParseFloat(value);
  } else if (key == "local_ground_margin_m") {
    params.local_ground_margin_m = ParseFloat(value);
  } else if (key == "max_above_plane_m") {
    params.max_above_plane_m = ParseFloat(value);
  } else if (key == "max_below_plane_m") {
    params.max_below_plane_m = ParseFloat(value);
  } else if (key == "adaptive_threshold_scale") {
    params.adaptive_threshold_scale = ParseFloat(value);
  } else if (key == "max_adaptive_threshold_m") {
    params.max_adaptive_threshold_m = ParseFloat(value);
  } else if (key == "max_ground_angle_deg") {
    params.max_ground_angle_deg = ParseFloat(value);
  } else if (key == "min_candidate_z") {
    params.min_candidate_z = ParseFloat(value);
  } else if (key == "max_candidate_z") {
    params.max_candidate_z = ParseFloat(value);
  } else if (key == "max_candidate_radius_m") {
    params.max_candidate_radius_m = ParseFloat(value);
  } else if (key == "candidate_grid_cell_m") {
    params.candidate_grid_cell_m = ParseFloat(value);
  } else if (key == "grid_max_height_step_m") {
    params.grid_max_height_step_m = ParseFloat(value);
  } else if (key == "grid_max_local_slope_m_per_m") {
    params.grid_max_local_slope_m_per_m = ParseFloat(value);
  } else if (key == "grid_seed_radius_m") {
    params.grid_seed_radius_m = ParseFloat(value);
  } else if (key == "grid_max_vertical_extent_m") {
    params.grid_max_vertical_extent_m = ParseFloat(value);
  } else if (key == "max_iterations") {
    params.max_iterations = ParseInt(value);
  } else if (key == "min_inliers") {
    params.min_inliers = ParseInt(value);
  } else if (key == "patchwork_num_zones") {
    params.patchwork_num_zones = ParseInt(value);
  } else if (key == "patchwork_num_sectors") {
    params.patchwork_num_sectors = ParseInt(value);
  } else if (key == "patchwork_min_points_per_patch") {
    params.patchwork_min_points_per_patch = ParseInt(value);
  } else if (key == "patchwork_seed_count") {
    params.patchwork_seed_count = ParseInt(value);
  } else if (key == "patchwork_refine_iterations") {
    params.patchwork_refine_iterations = ParseInt(value);
  } else if (key == "patchwork_min_range_m") {
    params.patchwork_min_range_m = ParseFloat(value);
  } else if (key == "patchwork_seed_threshold_m") {
    params.patchwork_seed_threshold_m = ParseFloat(value);
  } else if (key == "patchwork_distance_threshold_m") {
    params.patchwork_distance_threshold_m = ParseFloat(value);
  } else if (key == "patchwork_max_flatness_m2") {
    params.patchwork_max_flatness_m2 = ParseFloat(value);
  } else if (key == "gseg3d_cell_size_x_m") {
    params.gseg3d_cell_size_x_m = ParseFloat(value);
  } else if (key == "gseg3d_cell_size_y_m") {
    params.gseg3d_cell_size_y_m = ParseFloat(value);
  } else if (key == "gseg3d_cell_size_z_phase1_m") {
    params.gseg3d_cell_size_z_phase1_m = ParseFloat(value);
  } else if (key == "gseg3d_cell_size_z_phase2_m") {
    params.gseg3d_cell_size_z_phase2_m = ParseFloat(value);
  } else if (key == "gseg3d_slope_threshold_deg") {
    params.gseg3d_slope_threshold_deg = ParseFloat(value);
  } else if (key == "gseg3d_ground_inlier_threshold_m") {
    params.gseg3d_ground_inlier_threshold_m = ParseFloat(value);
  } else if (key == "gseg3d_centroid_search_radius_m") {
    params.gseg3d_centroid_search_radius_m = ParseFloat(value);
  } else if (key == "gseg3d_ground_height_m") {
    params.gseg3d_ground_height_m = ParseFloat(value);
  } else if (key == "gseg3d_max_ground_height_deviation_m") {
    params.gseg3d_max_ground_height_deviation_m = ParseFloat(value);
  } else if (key == "gseg3d_cuda_seed_radius_m") {
    params.gseg3d_cuda_seed_radius_m = ParseFloat(value);
  } else if (key == "gseg3d_cuda_cell_size_m") {
    params.gseg3d_cuda_cell_size_m = ParseFloat(value);
  }
}

void ApplyClusteringValue(
  algorithms::ClusteringParams & params,
  const std::string & key,
  const std::string & value)
{
  if (key == "enabled") {
    params.enabled = ParseBool(value);
  } else if (key == "use_cuda") {
    params.use_cuda = ParseBool(value);
  } else if (key == "mode") {
    params.mode = Trim(value);
  } else if (key == "tolerance_m") {
    params.tolerance_m = ParseFloat(value);
  } else if (key == "grid_cell_m") {
    params.grid_cell_m = ParseFloat(value);
  } else if (key == "range_tolerance_scale") {
    params.range_tolerance_scale = ParseFloat(value);
  } else if (key == "max_tolerance_m") {
    params.max_tolerance_m = ParseFloat(value);
  } else if (key == "max_vertical_gap_m") {
    params.max_vertical_gap_m = ParseFloat(value);
  } else if (key == "min_height_m") {
    params.min_height_m = ParseFloat(value);
  } else if (key == "max_height_m") {
    params.max_height_m = ParseFloat(value);
  } else if (key == "min_width_m") {
    params.min_width_m = ParseFloat(value);
  } else if (key == "max_width_m") {
    params.max_width_m = ParseFloat(value);
  } else if (key == "min_length_m") {
    params.min_length_m = ParseFloat(value);
  } else if (key == "max_length_m") {
    params.max_length_m = ParseFloat(value);
  } else if (key == "min_cluster_size") {
    params.min_cluster_size = ParseInt(value);
  } else if (key == "max_cluster_size") {
    params.max_cluster_size = ParseInt(value);
  }
}

void ApplyDetectionValue(
  algorithms::DetectionParams & params,
  const std::string & key,
  const std::string & value)
{
  if (key == "enabled") {
    params.enabled = ParseBool(value);
  } else if (key == "min_confidence") {
    params.min_confidence = ParseFloat(value);
  } else if (key == "min_points") {
    params.min_points = ParseInt(value);
  }
}
}  // namespace

fs::path ProjectResourcePath(const fs::path & relative)
{
  // Search beside the executable first so moving a complete checkout works.
  // Never pick up unrelated profiles from the shell's working directory.
  static const fs::path root = [] {
    std::error_code error;
    fs::path candidate = ExecutablePath().parent_path();
    for (int depth = 0; depth < 4 && !candidate.empty(); ++depth) {
      error.clear();
      if (fs::is_regular_file(candidate / "Resources/config/algorithms.yaml", error)) return candidate / "Resources";
      if (fs::is_regular_file(candidate / "share/lidar_viewer/config/algorithms.yaml", error)) return candidate / "share/lidar_viewer";
      if (fs::is_regular_file(candidate / "config/algorithms.yaml", error)) return candidate;
      candidate = candidate.parent_path();
    }
#ifdef LIDAR_VIEWER_SOURCE_DIR
    return fs::path(LIDAR_VIEWER_SOURCE_DIR);
#else
    throw std::runtime_error("Cannot locate LiDAR viewer resources");
#endif
  }();
  return root / relative;
}

algorithms::PipelineConfig LoadPipelineConfig(const fs::path & path)
{
  algorithms::PipelineConfig config;

  std::ifstream in(path);
  if (!in) {
    return config;
  }

  std::string section;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    if (Trim(line).empty()) {
      continue;
    }

    const bool nested = !line.empty() && std::isspace(static_cast<unsigned char>(line.front())) != 0;
    line = Trim(line);
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }

    const std::string key = Trim(line.substr(0, colon));
    const std::string value = Trim(line.substr(colon + 1));
    if (!nested && value.empty()) {
      section = key;
      continue;
    }
    try {
      if (section == "ground_removal") {
        ApplyGroundRemovalValue(config.ground_removal, key, value);
      } else if (section == "clustering") {
        ApplyClusteringValue(config.clustering, key, value);
      } else if (section == "detection") {
        ApplyDetectionValue(config.detection, key, value);
      }
    } catch (const std::exception & error) {
      throw std::runtime_error(path.string() + ':' + std::to_string(line_number) +
        ": invalid " + section + '.' + key + ": " + error.what());
    }
  }

  return config;
}
}  // namespace lidar_viewer
