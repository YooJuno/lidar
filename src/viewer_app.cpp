#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "lidar_viewer/data_io.hpp"
#include "lidar_viewer/bag_io.hpp"
#include "lidar_viewer/bag_catalog.hpp"
#include "lidar_viewer/config.hpp"
#include "lidar_viewer/algorithms/pipeline.hpp"
#include "lidar_viewer/algorithms/ground_removal.hpp"
#include "lidar_viewer/app.hpp"
#include "lidar_viewer/ui_layout.hpp"
#include "lidar_viewer/settings_io.hpp"
#include "lidar_viewer/camera_2d.hpp"
#include "lidar_viewer/roi.hpp"
#include "lidar_viewer/roi_editor.hpp"
#include "lidar_viewer/roi_export.hpp"
#include "lidar_viewer/display_alignment.hpp"
#include "lidar_viewer/ui_icons.hpp"
#include "lidar_viewer/app_icon.hpp"
#include "lidar_viewer/file_dialog.hpp"
#include "lidar_viewer/live_io.hpp"
#include "lidar_viewer/learned_detector.hpp"
#include "lidar_viewer/object_tracker.hpp"
#include "lidar_viewer/platform.hpp"
#include "lidar_viewer/background_worker.hpp"
#include "lidar_viewer/dataset.hpp"
#include "lidar_viewer/training_ui.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
constexpr int kWindowWidth = 1600;
constexpr int kWindowHeight = 960;
constexpr float kOrbitSensitivity = 0.0045f;
constexpr float kPanBaseScale = 0.002f;
constexpr const char * kSettingsFileName = "lidar_viewer_settings.cfg";
constexpr const char * kProfilesDir = "config/profiles";
constexpr const char * kAlgorithmsConfigPath = "config/algorithms.yaml";
constexpr const char * kDefaultInputPath = "dataset/gangnam/2019_10_04_12_28_25.bin";

float Dot(const Vec3 & a, const Vec3 & b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Cross(const Vec3 & a, const Vec3 & b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

float Length(const Vec3 & v)
{
  return std::sqrt(Dot(v, v));
}

Vec3 Normalize(const Vec3 & v)
{
  const float len = Length(v);
  if (len < 1e-6f) {
    return {0.0f, 0.0f, 0.0f};
  }
  return v * (1.0f / len);
}

struct Mat4
{
  std::array<float, 16> m{};
};

Mat4 Identity()
{
  return Mat4{{{
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  }}};
}

Mat4 Multiply(const Mat4 & a, const Mat4 & b)
{
  Mat4 out{};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      float value = 0.0f;
      for (int k = 0; k < 4; ++k) {
        value += a.m[k * 4 + row] * b.m[col * 4 + k];
      }
      out.m[col * 4 + row] = value;
    }
  }
  return out;
}

Mat4 Perspective(float fov_y_rad, float aspect, float z_near, float z_far)
{
  const float f = 1.0f / std::tan(fov_y_rad * 0.5f);
  return Mat4{{{
    f / aspect, 0.0f, 0.0f, 0.0f,
    0.0f, f, 0.0f, 0.0f,
    0.0f, 0.0f, (z_far + z_near) / (z_near - z_far), -1.0f,
    0.0f, 0.0f, (2.0f * z_far * z_near) / (z_near - z_far), 0.0f
  }}};
}

Mat4 Orthographic(float left, float right, float bottom, float top, float z_near, float z_far)
{
  Mat4 out = Identity();
  out.m[0] = 2.0f / (right - left);
  out.m[5] = 2.0f / (top - bottom);
  out.m[10] = -2.0f / (z_far - z_near);
  out.m[12] = -(right + left) / (right - left);
  out.m[13] = -(top + bottom) / (top - bottom);
  out.m[14] = -(z_far + z_near) / (z_far - z_near);
  return out;
}

Mat4 LookAt(const Vec3 & eye, const Vec3 & center, const Vec3 & up)
{
  const Vec3 forward = Normalize(center - eye);
  const Vec3 right = Normalize(Cross(forward, up));
  const Vec3 camera_up = Cross(right, forward);

  Mat4 out = Identity();
  out.m[0] = right.x;
  out.m[1] = camera_up.x;
  out.m[2] = -forward.x;
  out.m[4] = right.y;
  out.m[5] = camera_up.y;
  out.m[6] = -forward.y;
  out.m[8] = right.z;
  out.m[9] = camera_up.z;
  out.m[10] = -forward.z;
  out.m[12] = -Dot(right, eye);
  out.m[13] = -Dot(camera_up, eye);
  out.m[14] = Dot(forward, eye);
  return out;
}

GLuint CompileShader(GLenum type, const char * source)
{
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok == GL_TRUE) {
    return shader;
  }

  GLint log_len = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
  std::string log(static_cast<std::size_t>(log_len), '\0');
  glGetShaderInfoLog(shader, log_len, nullptr, log.data());
  throw std::runtime_error("Shader compile failed:\n" + log);
}

GLuint CreateProgram()
{
  constexpr const char * kVertexShader = R"(
    #version 330 core
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in float a_intensity;

    uniform mat4 u_mvp;
    uniform float u_point_size;
    out float v_intensity;
    out float v_height;

    void main() {
      gl_Position = u_mvp * vec4(a_position, 1.0);
      gl_PointSize = u_point_size;
      v_intensity = a_intensity;
      v_height = a_position.z;
    }
  )";

  constexpr const char * kFragmentShader = R"(
    #version 330 core
    in float v_intensity;
    in float v_height;
    out vec4 frag_color;

    uniform int u_color_mode;
    uniform float u_min_height;
    uniform float u_max_height;

    vec3 turbo(float x) {
      x = clamp(x, 0.0, 1.0);
      return vec3(
        0.13572138 + 4.61539260 * x - 42.66032258 * x * x + 132.13108234 * x * x * x - 152.94239396 * x * x * x * x + 59.28637943 * x * x * x * x * x,
        0.09140261 + 2.19418839 * x + 4.84296658 * x * x - 14.18503333 * x * x * x + 4.27729857 * x * x * x * x + 2.82956604 * x * x * x * x * x,
        0.10667330 + 12.64194608 * x - 60.58204836 * x * x + 110.36276771 * x * x * x - 89.90310912 * x * x * x * x + 27.34824973 * x * x * x * x * x
      );
    }

    void main() {
      if (v_intensity < -0.5) {
        frag_color = vec4(1.0, 0.12, 0.08, 1.0);
        return;
      }
      if (u_color_mode == 2) {
        frag_color = vec4(0.82, 0.88, 0.94, 1.0);
        return;
      }
      float value = v_intensity;
      if (u_color_mode == 1) {
        float denom = max(1e-6, u_max_height - u_min_height);
        value = (v_height - u_min_height) / denom;
      }
      frag_color = vec4(clamp(turbo(value), 0.0, 1.0), 1.0);
    }
  )";

  const GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  GLint ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(log_len), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    throw std::runtime_error("Program link failed:\n" + log);
  }

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}


GLuint CreateGridProgram()
{
  constexpr const char * kVertexShader = R"(
    #version 330 core
    layout(location = 0) in vec3 a_position;
    uniform mat4 u_mvp;
    void main() {
      gl_Position = u_mvp * vec4(a_position, 1.0);
    }
  )";

  constexpr const char * kFragmentShader = R"(
    #version 330 core
    uniform vec4 u_color;
    out vec4 frag_color;
    void main() {
      frag_color = u_color;
    }
  )";

  const GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  GLint ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(log_len), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    throw std::runtime_error("Grid program link failed:\n" + log);
  }

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}


struct Color
{
  float r;
  float g;
  float b;
  float a;
};


struct GridRenderer
{
  GLuint program{0};
  GLuint vao{0};
  GLuint vbo{0};
  GLint mvp_loc{-1};
  GLint color_loc{-1};
  std::vector<Vec3> vertices;
  bool dirty{true};
};

struct ClusterBoxRenderer
{
  GLuint vao{0};
  GLuint fill_vbo{0};
  GLuint line_vbo{0};
  struct Batch
  {
    Color fill_color{};
    Color line_color{};
    std::vector<Vec3> fill_vertices;
    std::vector<Vec3> line_vertices;
    std::size_t fill_first{0},line_first{0};
  };
  std::vector<Batch> batches;
  bool dirty{true};
};

struct GridSettings
{
  bool enabled{true};
  float spacing{5.0f};
  float thickness{1.0f};
  int count{20};
};

using CalibrationSettings=lidar_viewer::DisplayAlignment;

std::string FileDisplayName(const fs::path & path)
{
  return path.has_filename() ? path.filename().u8string() : path.u8string();
}

struct CameraState
{
  enum class ProjectionMode
  {
    View3D,
    View2D,
  };

  struct ViewBookmark
  {
    Vec3 target{};
    float yaw{0.0f};
    float pitch{0.0f};
    float distance{40.0f};
    bool valid{false};
  };

  Vec3 target{};
  float yaw{0.2f};
  float pitch{-0.35f};
  float distance{40.0f};
  float point_size{2.0f};
  ProjectionMode projection_mode{ProjectionMode::View3D};
  bool has_saved_view{false};
  ViewBookmark saved_3d_view{{}, 0.2f, -0.35f, 40.0f, false};
  ViewBookmark saved_2d_view{{}, 0.0f, -1.5707f, 40.0f, false};
  bool dragging_left{false};
  bool dragging_right{false};
  double last_x{0.0};
  double last_y{0.0};
};

struct ViewerState
{
  enum class GroundView {All, Ground, Obstacles};
  struct BagLoadResult
  {
    std::vector<FrameHandle> frames;
    FrameData first_frame;
    std::unique_ptr<ViewerState> prepared;
    float ground_offset{0};
  };
  struct FolderScanResult
  {
    fs::path folder;
    std::vector<BagSource> sources;
  };

  struct FileGroup
  {
    fs::path path;
    std::string display_name;
    std::size_t first_index{0};
    std::size_t last_index{0};
  };

  std::vector<FrameHandle> frames;
  std::vector<FileGroup> file_groups;
  std::vector<BagSource> bag_sources;
  struct BagSourceInfo {fs::path path;std::optional<std::uint64_t> duration;bool read_duration,available;};
  using BagDurations=std::vector<BagSourceInfo>;
  std::unordered_set<fs::path> unavailable_sources;
  std::vector<fs::path> hidden_sources;
  fs::path hidden_sources_config;
  bool check_source_availability{false};
  std::unordered_map<fs::path,std::string> bag_durations;
  std::unique_ptr<lidar_viewer::BackgroundWorker> duration_worker;
  std::future<BagDurations> duration_future;
  std::shared_ptr<std::atomic_bool> duration_cancel;
  std::optional<std::size_t> current_bag_source;
  std::optional<std::size_t> loading_bag_source;
  std::shared_ptr<lidar_viewer::InputLoadState> input_load;
  std::chrono::steady_clock::time_point input_load_started;
  bool input_load_autoplay{false};
  bool input_load_previous_playing{false};
  std::future<BagLoadResult> bag_load_future;
  ~ViewerState() {
    if (input_load) input_load->Cancel();
    if (duration_cancel) duration_cancel->store(true);
    if (export_progress) export_progress->cancel.store(true);
  }
  lidar_viewer::FileDialog bag_picker;
  lidar_viewer::LiveSession live;
  lidar_viewer::LiveStats live_stats;
  bool live_mode{false}, live_paused{false}, live_has_frame{false}, open_live_dialog{false};
  std::uint64_t live_display_sequence{0};
  std::array<char, 256> live_points_topic{}, live_imu_topic{};
  std::array<char, 2048> live_record_folder{};
  std::string bag_topic{"/b2/points"};
  std::string bag_imu_topic;
  bool imu_level_enabled{false};
  bool imu_level_cli_override{false};
  lidar_viewer::ImuMountSettings imu_mount;
  lidar_viewer::ImuLevelingResult imu_level_result;
  std::optional<float> bag_sensor_height_override;
  std::string bag_date_filter;
  std::array<char, 128> bag_search{};
  std::vector<fs::path> data_folders;
  fs::path data_folders_config;
  bool open_folder_picker{false};
  fs::path browsing_folder;
  std::array<char, 4096> folder_path_input{};
  std::vector<fs::path> folder_children;
  std::string folder_error;
  std::future<FolderScanResult> folder_scan_future;
  std::size_t current_index{0};
  std::size_t current_file_group{0};
  FrameData base_frame;
  FrameData frame;
  FrameData processed_frame;
  std::size_t processed_obstacles{0};
  double processing_ms{0}, roi_ms{0};
  double status_sample_time{0},status_processing_ms{0},status_roi_ms{0},status_fps{0};
  std::string status_source;
  std::size_t status_points{0};
  std::uint64_t status_live_display{0},status_live_received{0};
  std::size_t status_ground{0};
  std::optional<double> pending_read_started;
  std::uint64_t ground_passes{0};
  bool alignment_initialized{false};
  float global_ground_offset{0.0f};
  CameraState camera;
  GLuint vao{0};
  GLuint vbo{0};
  bool frame_dirty{false};
  bool fit_view_requested{false};
  int color_mode{0};
  bool is_playing{false};
  bool loop_playback{true};
  bool timeline_dragging{false};
  int seek_draft{0};
  float ui_scale{1.0f};
  float inspector_width{320.f};
  bool inspector_resizing{false};
  bool app_settings_loaded{false}, open_app_settings{false};
  bool app_settings_dirty{false};
  std::string app_settings_error;
  float font_raster_scale{1.0f};
  float height_color_min{-1.0f};
  float height_color_max{4.0f};
  lidar_viewer::UiLayout ui_layout;
  std::string ui_error;
  bool ground_removal_enabled{false};
  GroundView ground_view{GroundView::All};
  std::size_t ground_removed_count{0};
  double playback_fps{8.0};
  std::optional<double> input_fps_estimate;
  double current_playback_fps{0.0};
  double current_view_fps{0.0};
  double fps_measurement_started_at{0.0};
  int fps_measurement_frames{0};
  double playback_measurement_started_at{0.0};
  int playback_measurement_advances{0};
  double last_frame_advance_time{0.0};
  std::string transient_status;
  double transient_status_until{0.0};
  std::uint64_t ground_sequence{static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())};
  GridRenderer grid;
  ClusterBoxRenderer cluster_boxes;
  lidar_viewer::RoiSettings roi;
  // Persist existing roi_* boxes as ground exclusions. Display cropping owns
  // an independent collection and never changes an algorithm's input.
  lidar_viewer::RoiSettings display_roi, saved_display_roi;
  enum class RoiPurpose {Display, Ground};
  RoiPurpose roi_purpose{RoiPurpose::Ground};
  std::size_t selected_display_roi{0};
  lidar_viewer::RoiHistory display_roi_history;
  bool display_roi_pending{false};
  bool open_roi_export{false};
  lidar_viewer::FileDialog dataset_picker;
  bool dataset_tab_requested{false},open_training{false};
  bool dataset_fit_each{true};std::optional<std::size_t> dataset_fitted_index;
  std::optional<std::size_t> dataset_list_index;
  std::unique_ptr<lidar_viewer::TrainingWorkspace> training_workspace;
  bool export_aligned{false};
  int export_scope{0},export_first{1},export_last{1},export_stride{1};
  std::array<char,4096> export_directory{};
  lidar_viewer::FileDialog export_picker;
  std::unique_ptr<lidar_viewer::BackgroundWorker> export_worker;
  std::future<lidar_viewer::RoiExportResult> export_future;
  std::shared_ptr<lidar_viewer::RoiExportProgress> export_progress;
  std::size_t export_total{0};
  std::size_t export_region_count{0};
  double export_started_at{0};
  std::string export_error,export_message;
  fs::path export_destination;
  lidar_viewer::RoiSettings saved_roi;
  lidar_viewer::RoiSettings analysis_roi;
  lidar_viewer::RoiSettings applied_exclusions;
  bool apply_exclusions_requested{false}, applying_exclusions{false};
  std::size_t exclusion_removed_count{0};
  lidar_viewer::RoiHistory roi_history;
  enum class RoiTool {Navigate, Draw, Move};
  RoiTool roi_tool{RoiTool::Navigate};
  bool roi_tab_requested{false}, roi_tab_active{false}, roi_pending{false}, roi_numeric_edit{false};
  bool roi_reveal_selection{true}, save_requested{false};
  bool roi_details_requested{false};
  float roi_list_height{0};
  float roi_new_z{0}, roi_new_height{1};
  struct RoiGesture {
    bool active{false};
    enum class Kind {Move, Resize, Rotate} kind{Kind::Move};
    int handle_x{0}, handle_y{0};
    bool resume_playback{false}, resume_live{false};
    Vec3 start{};
    lidar_viewer::RoiSettings before;
    lidar_viewer::RoiBox original;
  } roi_gesture;
  std::size_t selected_roi{0};
  lidar_viewer::RoiCoordinateFrame roi_frame;
  struct RoiRenderer {
    GLuint vao{0}, vbo{0};
    std::vector<Vec3> vertices;
    bool dirty{true};
  } roi_renderer;
  GLuint axes_vao{0}, axes_vbo{0};
  float axes_length{-1};
  GridSettings grid_settings;
  CalibrationSettings calibration;
  lidar_viewer::algorithms::PipelineConfig algorithm_config;
  std::vector<lidar_viewer::algorithms::Cluster> clusters;
  std::vector<lidar_viewer::algorithms::Detection> detections;
  std::unique_ptr<lidar_viewer::LearnedDetector> detector;
  std::shared_ptr<const std::vector<KittiPoint>> detector_input;
  std::vector<lidar_viewer::algorithms::Detection> full_detections;
  lidar_viewer::TrackingParams tracking_params;
  std::unique_ptr<lidar_viewer::ObjectTracker> tracker;
  std::vector<lidar_viewer::ObjectTrack> visible_tracks;
  std::uint64_t tracking_session{1},tracking_revision{0},tracking_alignment{0};
  std::string tracking_model,tracking_frame_id,tracking_source;
  std::string tracking_error;
  std::uint64_t detector_revision{1}, detector_submitted{0}, detector_completed{0};
  float detector_yaw{0}, detector_latency_ms{0};
  double runtime_sample_time{0};
  float runtime_latency_ms{0};
  std::size_t runtime_shown{0},runtime_candidates{0};
  std::uint64_t runtime_session{0};
  bool runtime_sample_valid{false};
  std::string runtime_detail;
  double detector_submitted_at{0};
  std::string detector_detail, detector_error, detector_active_model;
  bool detector_model_loaded{false};
  bool detector_retry{false}, detector_cli_override{false};
  int object_method{3}; // Last selected method while object analysis is disabled.
  std::string dataset_profile{"default"};
  std::unordered_map<std::size_t, FrameData> frame_cache;
  std::deque<std::size_t> frame_cache_order;
  std::unique_ptr<lidar_viewer::BackgroundWorker> frame_reader;
  std::future<FrameData> prefetch_future;
  std::optional<std::size_t> requested_frame;
  std::optional<std::size_t> prefetched_index;
  std::unique_ptr<lidar_viewer::BackgroundWorker> geometry_worker;
  std::future<std::unique_ptr<ViewerState>> geometry_future;
  bool async_geometry{false}, geometry_updates_detector{true}, geometry_roi_only{false};
  std::optional<std::size_t> geometry_index;
  std::optional<std::uint64_t> geometry_live_sequence;
  std::uint64_t geometry_config{0}, submitted_geometry_config{0};
  std::uint64_t geometry_ground_config{0}, geometry_input_config{0};
  std::uint64_t submitted_ground_config{0}, submitted_input_config{0};
};

const lidar_viewer::RoiSettings & EditedRoi(const ViewerState & state)
{
  return state.roi_purpose==ViewerState::RoiPurpose::Display ? state.display_roi : state.roi;
}
lidar_viewer::RoiSettings & EditedRoi(ViewerState & state)
{
  return state.roi_purpose==ViewerState::RoiPurpose::Display ? state.display_roi : state.roi;
}
std::size_t SelectedRoi(const ViewerState & state)
{
  return state.roi_purpose==ViewerState::RoiPurpose::Display ? state.selected_display_roi : state.selected_roi;
}
std::size_t & SelectedRoi(ViewerState & state)
{
  return state.roi_purpose==ViewerState::RoiPurpose::Display ? state.selected_display_roi : state.selected_roi;
}
lidar_viewer::RoiHistory & EditedRoiHistory(ViewerState & state)
{
  return state.roi_purpose==ViewerState::RoiPurpose::Display ? state.display_roi_history : state.roi_history;
}
void MarkRoiPending(ViewerState & state)
{
  state.roi_pending=true;
  if (state.roi_purpose==ViewerState::RoiPurpose::Display) state.display_roi_pending=true;
}

void SetTransientStatus(ViewerState & state, const std::string & message);

struct VisiblePointRange {std::size_t first{0}, count{0};};

VisiblePointRange VisiblePoints(const ViewerState & state)
{
  const auto total=state.frame.points.size(),ground=std::min(total,state.ground_removed_count);
  switch (state.ground_view) {
    case ViewerState::GroundView::Ground:return {total-ground,ground};
    case ViewerState::GroundView::Obstacles:return {0,total-ground};
    case ViewerState::GroundView::All:return {0,total};
  }
  return {0,total};
}

const char * GroundViewKey(ViewerState::GroundView mode)
{
  return mode==ViewerState::GroundView::Ground ? "ground" :
    mode==ViewerState::GroundView::Obstacles ? "obstacles" : "all";
}

Vec3 CameraPosition(const CameraState & camera)
{
  const float cp = std::cos(camera.pitch);
  return {
    camera.target.x + camera.distance * cp * std::sin(camera.yaw),
    camera.target.y + camera.distance * cp * std::cos(camera.yaw),
    camera.target.z + camera.distance * std::sin(camera.pitch)
  };
}

void StoreCurrentProjectionView(CameraState & camera)
{
  auto & bookmark =
    (camera.projection_mode == CameraState::ProjectionMode::View2D)
    ? camera.saved_2d_view
    : camera.saved_3d_view;
  bookmark.target = camera.target;
  bookmark.yaw = camera.yaw;
  bookmark.pitch = camera.pitch;
  bookmark.distance = camera.distance;
  bookmark.valid = true;
}

void ResetCameraToFrame(ViewerState & state)
{
  state.camera.target = state.frame.center;
  if (state.camera.projection_mode == CameraState::ProjectionMode::View2D) {
    state.camera.target.z = 0.0f;
    state.camera.distance = std::max(state.frame.radius * 1.15f, 12.0f);
    state.camera.pitch = -1.5707f;
    state.camera.yaw = 0.0f;
  } else {
    state.camera.distance = state.frame.radius * 2.2f;
    state.camera.pitch = 0.55f;
    state.camera.yaw = -0.8f;
  }
  StoreCurrentProjectionView(state.camera);
}

void SetProjectionMode(ViewerState & state, CameraState::ProjectionMode mode)
{
  if (state.camera.projection_mode == mode) {
    return;
  }

  StoreCurrentProjectionView(state.camera);
  state.camera.projection_mode = mode;
  const auto & bookmark =
    (mode == CameraState::ProjectionMode::View2D)
    ? state.camera.saved_2d_view
    : state.camera.saved_3d_view;

  if (bookmark.valid) {
    state.camera.target = bookmark.target;
    state.camera.yaw = bookmark.yaw;
    state.camera.pitch = bookmark.pitch;
    state.camera.distance = mode==CameraState::ProjectionMode::View2D ?
      lidar_viewer::TopViewHalfHeight(bookmark.distance) : std::max(bookmark.distance,1.f);
  } else if (mode == CameraState::ProjectionMode::View2D) {
    state.camera.target.z = 0.0f;
    state.camera.distance = lidar_viewer::TopViewHalfHeight(state.camera.distance);
    state.camera.pitch = -1.5707f;
    state.camera.yaw = 0.0f;
  } else {
    state.camera.distance = std::max(state.camera.distance, 2.0f);
    state.camera.pitch = 0.55f;
    state.camera.yaw = -0.8f;
  }
  StoreCurrentProjectionView(state.camera);
}

void UploadFrame(ViewerState & state)
{
  glBindVertexArray(state.vao);
  glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
  glBufferData(
    GL_ARRAY_BUFFER,
    static_cast<GLsizeiptr>(state.frame.points.size() * sizeof(KittiPoint)),
    state.frame.points.data(),
    GL_STATIC_DRAW);
}

void AppendBoxLine(std::vector<Vec3> & vertices, const Vec3 & a, const Vec3 & b)
{
  vertices.push_back(a);
  vertices.push_back(b);
}

void AppendBoxFace(
  std::vector<Vec3> & vertices,
  const Vec3 & a,
  const Vec3 & b,
  const Vec3 & c,
  const Vec3 & d)
{
  vertices.push_back(a);
  vertices.push_back(b);
  vertices.push_back(c);
  vertices.push_back(a);
  vertices.push_back(c);
  vertices.push_back(d);
}

void AppendClusterBox(
  const lidar_viewer::algorithms::BoundingBox3D & bounds,
  std::vector<Vec3> & fill_vertices,
  std::vector<Vec3> & line_vertices,
  const lidar_viewer::algorithms::OrientedBox3D & oriented = {})
{
  const auto corner=[&](float x,float y,float z) {
    return oriented.valid ? oriented.center+oriented.axis_x*(x*oriented.half_length)+oriented.axis_y*(y*oriented.half_width)+
      Vec3{0,0,(z<0 ? oriented.min_z : oriented.max_z)-oriented.center.z} :
      Vec3{x<0 ? bounds.min.x : bounds.max.x,y<0 ? bounds.min.y : bounds.max.y,z<0 ? bounds.min.z : bounds.max.z};
  };
  const Vec3 p000=corner(-1,-1,-1);
  const Vec3 p100=corner(1,-1,-1);
  const Vec3 p110=corner(1,1,-1);
  const Vec3 p010=corner(-1,1,-1);
  const Vec3 p001=corner(-1,-1,1);
  const Vec3 p101=corner(1,-1,1);
  const Vec3 p111=corner(1,1,1);
  const Vec3 p011=corner(-1,1,1);

  AppendBoxFace(fill_vertices, p000, p100, p110, p010);
  AppendBoxFace(fill_vertices, p001, p011, p111, p101);
  AppendBoxFace(fill_vertices, p000, p001, p101, p100);
  AppendBoxFace(fill_vertices, p100, p101, p111, p110);
  AppendBoxFace(fill_vertices, p110, p111, p011, p010);
  AppendBoxFace(fill_vertices, p010, p011, p001, p000);

  AppendBoxLine(line_vertices, p000, p100);
  AppendBoxLine(line_vertices, p100, p110);
  AppendBoxLine(line_vertices, p110, p010);
  AppendBoxLine(line_vertices, p010, p000);
  AppendBoxLine(line_vertices, p001, p101);
  AppendBoxLine(line_vertices, p101, p111);
  AppendBoxLine(line_vertices, p111, p011);
  AppendBoxLine(line_vertices, p011, p001);
  AppendBoxLine(line_vertices, p000, p001);
  AppendBoxLine(line_vertices, p100, p101);
  AppendBoxLine(line_vertices, p110, p111);
  AppendBoxLine(line_vertices, p010, p011);
}

std::pair<Color, Color> DetectionClassColors(int class_id)
{
  switch (class_id) {
    case lidar_viewer::algorithms::DetectionClassCar:
      return {{0.12f, 0.72f, 1.00f, 0.14f}, {0.12f, 0.84f, 1.00f, 0.92f}};
    case lidar_viewer::algorithms::DetectionClassBus:
      return {{1.00f, 0.52f, 0.08f, 0.16f}, {1.00f, 0.66f, 0.14f, 0.94f}};
    case lidar_viewer::algorithms::DetectionClassTruck:
      return {{0.84f, 0.24f, 0.18f, 0.16f}, {0.96f, 0.34f, 0.28f, 0.94f}};
    case lidar_viewer::algorithms::DetectionClassPerson:
      return {{0.28f, 0.92f, 0.46f, 0.16f}, {0.38f, 1.00f, 0.54f, 0.94f}};
    case lidar_viewer::algorithms::DetectionClassCyclist:
      return {{0.98f, 0.90f, 0.20f, 0.16f}, {1.00f, 0.96f, 0.32f, 0.94f}};
    case lidar_viewer::algorithms::DetectionClassTree:
      return {{0.16f, 0.58f, 0.22f, 0.16f}, {0.24f, 0.78f, 0.30f, 0.94f}};
    default:
      return {{0.86f, 0.30f, 0.82f, 0.14f}, {0.94f, 0.42f, 0.90f, 0.92f}};
  }
}

void RebuildClusterBoxes(ViewerState & state)
{
  state.cluster_boxes.batches.clear();

  if (!state.algorithm_config.clustering.enabled && !state.algorithm_config.detection.enabled) {
    state.cluster_boxes.dirty = true;
    return;
  }

  if (state.algorithm_config.detection.enabled) {
    state.cluster_boxes.batches.reserve(5);
    for (const auto & detection : state.detections) {
      auto [fill_color, line_color] = DetectionClassColors(detection.class_id);
      const auto index=static_cast<std::size_t>(&detection-state.detections.data());
      if (index<state.visible_tracks.size() && state.visible_tracks[index].id) {
        if (state.visible_tracks[index].predicted) {fill_color.a*=.15f;line_color.a*=.45f;}
        else if (!state.visible_tracks[index].confirmed) {fill_color.a*=.4f;line_color.a*=.75f;}
      }
      auto batch_it = std::find_if(
        state.cluster_boxes.batches.begin(),
        state.cluster_boxes.batches.end(),
        [&](const ClusterBoxRenderer::Batch & batch) {
          return
            batch.line_color.r == line_color.r &&
            batch.line_color.g == line_color.g &&
            batch.line_color.b == line_color.b &&
            batch.line_color.a == line_color.a;
        });
      if (batch_it == state.cluster_boxes.batches.end()) {
        state.cluster_boxes.batches.push_back({fill_color, line_color, {}, {}});
        batch_it = std::prev(state.cluster_boxes.batches.end());
      }
      AppendClusterBox(detection.bounds, batch_it->fill_vertices, batch_it->line_vertices, detection.oriented_bounds);
    }
  }
  if (state.algorithm_config.clustering.enabled) {
    state.cluster_boxes.batches.push_back({
      {0.08f, 0.72f, 1.0f, 0.12f},
      {0.08f, 0.86f, 1.0f, 0.86f},
      {},
      {}
    });
    auto & batch = state.cluster_boxes.batches.back();
    batch.fill_vertices.reserve(state.clusters.size() * 36);
    batch.line_vertices.reserve(state.clusters.size() * 24);
    for (const auto & cluster : state.clusters) {
      AppendClusterBox(cluster.bounds, batch.fill_vertices, batch.line_vertices, cluster.oriented_bounds);
    }
  }

  state.cluster_boxes.dirty = true;
}

void FilterDetections(ViewerState & state)
{
  state.detections.clear();state.visible_tracks.clear();
  if (state.algorithm_config.detection.enabled && state.detector_completed==state.detector_revision) {
    std::vector<unsigned char> assigned(state.full_detections.size());
    if (state.tracking_params.enabled && state.tracker && state.tracking_revision==state.detector_revision) {
      for (const auto & track:state.tracker->Tracks()) {
        if (track.measurement_index && *track.measurement_index<assigned.size()) assigned[*track.measurement_index]=1;
        if (state.algorithm_config.detection.Accepts(track.detection)) {
          state.detections.push_back(track.detection);state.visible_tracks.push_back(track);
        }
      }
    }
    for (std::size_t i=0;i<state.full_detections.size();++i)
      if (!assigned[i] && state.algorithm_config.detection.Accepts(state.full_detections[i])) {
        state.detections.push_back(state.full_detections[i]);state.visible_tracks.emplace_back();
      }
  }
  RebuildClusterBoxes(state);
}

void ResetTracking(ViewerState & state)
{
  if (state.tracker) state.tracker->Reset();
  ++state.tracking_session;state.tracking_revision=0;state.visible_tracks.clear();
}

void TrackCurrentDetections(ViewerState & state)
{
  if (!state.tracking_params.enabled || state.detector_completed!=state.detector_revision) return;
  std::string source;
  if (state.live_mode) source="live:"+state.bag_topic;
  else if (!state.frames.empty()) {
    const auto & handle=state.frames[state.current_index];
    source=(handle.format==InputFormat::KittiXyzi ? handle.path.parent_path() : handle.path).u8string();
  }
  if (state.tracking_model!=state.algorithm_config.detection.model || state.tracking_frame_id!=state.base_frame.frame_id ||
      state.tracking_alignment!=state.geometry_input_config || state.tracking_source!=source) {
    ResetTracking(state);state.tracking_model=state.algorithm_config.detection.model;
    state.tracking_frame_id=state.base_frame.frame_id;state.tracking_alignment=state.geometry_input_config;
    state.tracking_source=source;
  }
  if (state.tracking_revision==state.detector_revision) return;
  if (!state.tracker) state.tracker=std::make_unique<lidar_viewer::ObjectTracker>();
  lidar_viewer::TrackingFrame frame;frame.session=state.tracking_session;
  frame.sequence=state.live_mode ? state.live_display_sequence : state.current_index;
  frame.allow_skipped_frames=state.live_mode;
  frame.timestamp_ns=state.base_frame.timestamp_ns && *state.base_frame.timestamp_ns>=0 ? *state.base_frame.timestamp_ns :
    static_cast<std::int64_t>(static_cast<long double>(frame.sequence)*1000000000.L/state.input_fps_estimate.value_or(10.));
  try {
    state.tracker->Update(state.full_detections,frame,state.tracking_params,state.algorithm_config.detection.min_confidence);
    state.tracking_error.clear();
  } catch (const std::exception & error) {state.tracker->Reset();state.tracking_error=error.what();}
  state.tracking_revision=state.detector_revision;
  if (std::getenv("LIDAR_VIEWER_TRACKING_TRACE")) {
    for (const auto & track:state.tracker->Tracks())
      std::cout<<"Track frame "<<frame.sequence<<" id "<<track.id<<" class "<<track.detection.class_id
        <<" confirmed "<<track.confirmed<<" predicted "<<track.predicted<<" speed "
        <<std::hypot(track.velocity.x,track.velocity.y)<<" x "<<track.detection.oriented_bounds.center.x<<std::endl;
    if (!state.tracker->Stats().reset_reason.empty()) std::cout<<"Track reset: "<<state.tracker->Stats().reset_reason<<std::endl;
  }
}

double DetectorClock()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool DetectorFramePending(const ViewerState & state)
{
  if (!state.algorithm_config.detection.enabled || !state.detector ||
      state.detector_submitted!=state.detector_revision || state.detector_completed==state.detector_revision) return false;
  const auto status=state.detector->Status();
  // Let playback continue during startup or failure. Once ready, keep cloud
  // and boxes on the same revision. A slow backend never blocks UI or recording.
  return status.ready && !status.failed && DetectorClock()-state.detector_submitted_at<2.;
}

bool UpdateDetector(ViewerState & state)
{
  if (!state.algorithm_config.detection.enabled) return false;
  if (!state.detector_input || state.detector_input->empty()) {
    if (state.detector_completed!=state.detector_revision) {
      state.detector_completed=state.detector_revision;state.full_detections.clear();TrackCurrentDetections(state);FilterDetections(state);
    }
    return false;
  }
  if (!state.detector || state.detector_retry || state.detector_active_model!=state.algorithm_config.detection.model) {
    if (!state.detector) state.detector=std::make_unique<lidar_viewer::LearnedDetector>();
    state.detector_active_model=state.algorithm_config.detection.model;
    state.detector->Start(lidar_viewer::ProjectResourcePath("python"),state.detector_active_model);
    state.detector_retry=false;state.detector_submitted=0;state.detector_error.clear();
  }
  if (state.detector_submitted!=state.detector_revision) {
    // Lower candidates never suppress a higher-scoring NMS survivor. Cache
    // down to the UI minimum once; changing score is then an immediate filter.
    try {state.detector->Submit(state.detector_revision,state.detector_input,.1f,state.detector_yaw);}
    catch (const std::exception & error) {
      state.detector_error=error.what();state.detector_completed=state.detector_revision;
      state.full_detections.clear();FilterDetections(state);return false;
    }
    state.detector_submitted=state.detector_revision;state.detector_submitted_at=DetectorClock();
  }
  if (auto result=state.detector->Poll();result && result->revision==state.detector_revision) {
    state.detector_completed=result->revision;state.full_detections=std::move(result->detections);
    state.detector_latency_ms=result->latency_ms;state.detector_detail=result->detail;state.detector_error=result->error;
    if (state.detector_error.empty()) TrackCurrentDetections(state);else ResetTracking(state);
    FilterDetections(state);
    if (std::getenv("LIDAR_VIEWER_DETECTOR_TRACE"))
      std::cout<<"Detector frame "<<state.current_index<<" revision "<<state.detector_revision<<" boxes "<<state.detections.size()<<" full "<<state.full_detections.size()<<std::endl;
    return true;
  }
  return false;
}

void UpdateSourceDurations(ViewerState & state)
{
  if (state.duration_future.valid()) {
    if (state.duration_future.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return;
    for (auto & item:state.duration_future.get()) {
      if (std::none_of(state.bag_sources.begin(),state.bag_sources.end(),[&](const BagSource & source) {
          return source.path==item.path;
        })) continue;
      if (item.available) state.unavailable_sources.erase(item.path);
      else state.unavailable_sources.insert(item.path);
      if (!item.read_duration) continue;
      auto formatted=FormatBagDuration(item.duration);
      if (std::getenv("LIDAR_VIEWER_CATALOG_TRACE"))
        std::cout<<"Bag duration: "<<item.path<<" | "<<formatted<<std::endl;
      state.bag_durations.insert_or_assign(item.path,std::move(formatted));
    }
  }
  std::vector<std::pair<fs::path,bool>> missing;
  for (const auto & source:state.bag_sources) {
    const bool read_duration=!state.bag_durations.count(source.path);
    if (read_duration || state.check_source_availability) missing.emplace_back(source.path,read_duration);
  }
  state.check_source_availability=false;
  if (missing.empty()) return;
  if (!state.duration_worker) {
    state.duration_worker=std::make_unique<lidar_viewer::BackgroundWorker>();
    state.duration_cancel=std::make_shared<std::atomic_bool>(false);
  }
  state.duration_future=state.duration_worker->Submit([paths=std::move(missing),cancel=state.duration_cancel] {
    ViewerState::BagDurations durations;
    durations.reserve(paths.size());
    for (const auto & item:paths) {
      if (cancel->load()) break;
      std::error_code error;
      const bool available=fs::exists(item.first,error) && !error;
      durations.push_back({item.first,item.second && available ? ReadBagDurationNs(item.first) : std::nullopt,
        item.second,available});
    }
    return durations;
  });
}

const std::string & SourceDuration(const ViewerState & state,const BagSource & source)
{
  static const std::string pending="...";
  const auto found=state.bag_durations.find(source.path);
  return found==state.bag_durations.end() ? pending : found->second;
}

void BuildFileGroups(ViewerState & state)
{
  state.file_groups.clear();
  if (state.frames.empty()) {
    state.current_file_group = 0;
    return;
  }

  std::size_t first_index = 0;
  while (first_index < state.frames.size()) {
    const fs::path current_path = state.frames[first_index].path;
    std::size_t last_index = first_index;
    while ((last_index + 1) < state.frames.size() && state.frames[last_index + 1].path == current_path) {
      ++last_index;
    }
    if (state.frames[first_index].format==InputFormat::KittiXyzi) {
      while (last_index+1<state.frames.size() && state.frames[last_index+1].format==InputFormat::KittiXyzi &&
          state.frames[last_index+1].path.parent_path()==current_path.parent_path()) ++last_index;
    }
    if(state.frames[first_index].dataset_sample) {
      while(last_index+1<state.frames.size() && state.frames[last_index+1].dataset_sample) ++last_index;
    }
    const auto display_path=state.frames[first_index].dataset_sample ? state.frames[first_index].dataset_sample->root :
      state.frames[first_index].format==InputFormat::KittiXyzi && last_index>first_index ? current_path.parent_path() : current_path;

    state.file_groups.push_back({
      display_path,
      FileDisplayName(display_path),
      first_index,
      last_index
    });
    first_index = last_index + 1;
  }

  state.current_file_group = 0;
  if (std::getenv("LIDAR_VIEWER_UI_TRACE")) {
    for (const auto & group:state.file_groups)
      std::cout<<"Source frames "<<group.first_index<<' '<<group.last_index<<std::endl;
  }
}

void UpdateCurrentFileGroup(ViewerState & state)
{
  for (std::size_t i = 0; i < state.file_groups.size(); ++i) {
    const auto & group = state.file_groups[i];
    if (state.current_index >= group.first_index && state.current_index <= group.last_index) {
      state.current_file_group = i;
      return;
    }
  }
  state.current_file_group = 0;
}

void RecomputeFrameBounds(FrameData & frame)
{
  if (frame.points.empty()) {
    frame.center = {0.0f, 0.0f, 0.0f};
    frame.radius = 10.0f;
    frame.min_z = 0.0f;
    frame.max_z = 0.0f;
    return;
  }

  Vec3 min_corner{
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()
  };
  Vec3 max_corner{
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()
  };

  for (const auto & p : frame.points) {
    min_corner.x = std::min(min_corner.x, p.x);
    min_corner.y = std::min(min_corner.y, p.y);
    min_corner.z = std::min(min_corner.z, p.z);
    max_corner.x = std::max(max_corner.x, p.x);
    max_corner.y = std::max(max_corner.y, p.y);
    max_corner.z = std::max(max_corner.z, p.z);
  }

  frame.center = {
    0.5f * (min_corner.x + max_corner.x),
    0.5f * (min_corner.y + max_corner.y),
    0.5f * (min_corner.z + max_corner.z)
  };
  const Vec3 extent = max_corner - min_corner;
  frame.radius = std::max({extent.x, extent.y, extent.z}) * 0.7f;
  frame.radius = std::max(frame.radius, 10.0f);
  frame.min_z = min_corner.z;
  frame.max_z = max_corner.z;
}

struct CalibrationTransform
{
  std::array<float, 9> rotation{};
  float z_offset{0.0f};
};

CalibrationTransform BuildCalibrationTransform(const CalibrationSettings & calibration)
{
  const float yaw = calibration.yaw_deg * kPi / 180.0f;
  const float pitch = calibration.pitch_deg * kPi / 180.0f;
  const float roll = calibration.roll_deg * kPi / 180.0f;

  const float cy = std::cos(yaw);
  const float sy = std::sin(yaw);
  const float cp = std::cos(pitch);
  const float sp = std::sin(pitch);
  const float cr = std::cos(roll);
  const float sr = std::sin(roll);

  return {{
    cp * cy, (sr * sp * cy - cr * sy), (cr * sp * cy + sr * sy),
    cp * sy, (sr * sp * sy + cr * cy), (cr * sp * sy - sr * cy),
    -sp,     sr * cp,                  cr * cp
  }, calibration.z_m};
}

Vec3 ApplyCalibrationTransform(const Vec3 & v, const CalibrationTransform & transform)
{
  return {
    transform.rotation[0] * v.x + transform.rotation[1] * v.y + transform.rotation[2] * v.z,
    transform.rotation[3] * v.x + transform.rotation[4] * v.y + transform.rotation[5] * v.z,
    transform.rotation[6] * v.x + transform.rotation[7] * v.y + transform.rotation[8] * v.z +
      transform.z_offset
  };
}

void ApplyGlobalAlignment(FrameData & frame, float ground_offset)
{
  if (frame.points.empty() || ground_offset == 0.0f) {
    return;
  }
  for (auto & p : frame.points) {
    p.z -= ground_offset;
  }
  // Translation changes neither XY bounds nor extents/radius.
  frame.center.z -= ground_offset;
  frame.min_z -= ground_offset;
  frame.max_z -= ground_offset;
  frame.ground_z -= ground_offset;
}

void RebuildRoiGeometry(ViewerState & state)
{
  auto & vertices=state.roi_renderer.vertices;
  vertices.clear(); vertices.reserve(EditedRoi(state).boxes.size()*24);
  for (const auto & box : EditedRoi(state).boxes) {
    auto corners=lidar_viewer::RoiCorners(box);
    for (auto & corner : corners) corner=state.roi_frame.Transform(corner);
    for (unsigned i=0;i<8;++i) for (unsigned bit : {1u,2u,4u}) if (!(i&bit)) {
      vertices.push_back(corners[i]); vertices.push_back(corners[i|bit]);
    }
  }
  state.roi_renderer.dirty=true;
}

void ApplyRoiResult(ViewerState & state, const FrameData & processed, std::size_t obstacles,
  const lidar_viewer::RoiCoordinateFrame & coordinates)
{
  const auto started=std::chrono::steady_clock::now();
  FrameData frame;
  frame.frame_id=processed.frame_id; frame.imu=processed.imu; frame.ground_z=processed.ground_z;
  frame.timestamp_ns=processed.timestamp_ns;
  frame.points.reserve(processed.points.size());
  frame.points.assign(processed.points.begin(),processed.points.begin()+static_cast<std::ptrdiff_t>(obstacles));
  auto clusters=state.ground_removal_enabled ? lidar_viewer::algorithms::RunClustering(frame.points,state.algorithm_config.clustering) :
    std::vector<lidar_viewer::algorithms::Cluster>{};
  const auto retained_obstacles=frame.points.size();
  frame.points.insert(frame.points.end(),processed.points.begin()+static_cast<std::ptrdiff_t>(obstacles),processed.points.end());
  // Classify/cluster the full scan first. Stable compaction preserves the two
  // ground display ranges, including points skipped by the exclusion boxes.
  auto visible_obstacles=retained_obstacles;
  if (state.display_roi.enabled) {
    const lidar_viewer::PreparedRoi display(state.display_roi,coordinates);
    std::size_t retained=0;visible_obstacles=0;
    for (std::size_t i=0;i<frame.points.size();++i) {
      const auto & p=frame.points[i];
      if (!display.Contains({p.x,p.y,p.z})) continue;
      if (i<retained_obstacles) ++visible_obstacles;
      frame.points[retained++]=p;
    }
    frame.points.resize(retained);
  }
  RecomputeFrameBounds(frame);
  state.ground_removed_count=frame.points.size()-visible_obstacles;
  state.frame=std::move(frame); state.clusters=std::move(clusters);
  state.roi_frame=coordinates;
  RebuildRoiGeometry(state); FilterDetections(state);
  state.frame_dirty=true; state.roi_pending=false;
  state.roi_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

void PrepareDetectorInput(ViewerState & state, const FrameData & aligned, const FrameData & raw, float yaw,
  float ground_offset)
{
  auto points=std::make_shared<std::vector<KittiPoint>>();points->reserve(aligned.points.size());
  for (std::size_t i=0;i<aligned.points.size();++i) {
    const auto & p=aligned.points[i];
    // The base scan already has the dataset floor offset subtracted. Identify
    // missing returns at the original sensor origin, not the shifted origin.
    if (raw.points[i].x!=0 || raw.points[i].y!=0 || raw.points[i].z!=-ground_offset) points->push_back(p);
  }
  state.detector_input=std::move(points);state.detector_yaw=yaw;
  ++state.detector_revision;state.full_detections.clear();state.detections.clear();state.detector_error.clear();
}

void RefreshDetectorInput(ViewerState & state)
{
  // Enabling a detector does not require running ground segmentation again.
  if (state.detector_input) {FilterDetections(state);return;}
  const auto leveling=state.imu_level_enabled ? lidar_viewer::ComputeImuLeveling(state.base_frame.imu,state.imu_mount) :
    lidar_viewer::ImuLevelingResult{};
  const auto resolved=lidar_viewer::ResolveDisplayAlignment(state.calibration,state.imu_level_enabled,leveling);
  FrameData aligned=state.base_frame;
  const auto transform=BuildCalibrationTransform(resolved.effective);
  for (auto & point:aligned.points) {
    const auto p=ApplyCalibrationTransform({point.x,point.y,point.z},transform);
    point.x=p.x;point.y=p.y;point.z=p.z;
  }
  PrepareDetectorInput(state,aligned,state.base_frame,resolved.effective.yaw_deg*kPi/180.f,state.global_ground_offset);
}

void ApplyCalibration(ViewerState & state, const FrameData * source = nullptr,
  std::optional<float> source_ground_offset = std::nullopt, bool update_detector=true)
{
  const auto started=std::chrono::steady_clock::now();
  FrameData frame=source ? *source : state.base_frame;
  frame.source_intensities.reset();
  lidar_viewer::ImuLevelingResult leveling;
  leveling.status="Off";
  if (state.imu_level_enabled) {
    leveling=lidar_viewer::ComputeImuLeveling(frame.imu,state.imu_mount);
  }
  const auto resolved=lidar_viewer::ResolveDisplayAlignment(state.calibration,state.imu_level_enabled,leveling);
  const auto transform=BuildCalibrationTransform(resolved.effective);
  lidar_viewer::RoiCoordinateFrame coordinates;
  coordinates.rotation=transform.rotation;
  coordinates.translation=coordinates.Rotate({0,0,-source_ground_offset.value_or(state.global_ground_offset)})+
    Vec3{0,0,transform.z_offset};
  const lidar_viewer::PreparedRoi exclusions(state.applied_exclusions,coordinates,lidar_viewer::RoiFilterStage::Preprocessing);
  for (auto & point : frame.points) {
    const auto rotated=ApplyCalibrationTransform({point.x,point.y,point.z},transform);
    point.x=rotated.x; point.y=rotated.y; point.z=rotated.z;
  }
  if (update_detector) {
    state.detector_input.reset();
    if (state.algorithm_config.detection.enabled) {
    // Organized XT32 scans contain zero XYZ for missing returns. Do not turn
    // those into a dense artificial pillar when height alignment translates Z.
      PrepareDetectorInput(state,frame,source ? *source : state.base_frame,resolved.effective.yaw_deg*kPi/180.f,
        source_ground_offset.value_or(state.global_ground_offset));
    } else {
      ++state.detector_revision;state.full_detections.clear();state.detections.clear();state.detector_error.clear();
    }
  }
  state.exclusion_removed_count=0;
  std::size_t obstacles=frame.points.size();
  if (state.ground_removal_enabled && !frame.points.empty()) {
    auto skipped=exclusions.ExtractRejected(frame.points);
    state.exclusion_removed_count=skipped.size();
    auto params=state.algorithm_config.ground_removal; params.enabled=true;
    params.sensor_height_m=std::max(.05f,transform.z_offset-source_ground_offset.value_or(state.global_ground_offset)-params.gseg3d_ground_height_m);
    params.sequence_id=state.ground_sequence;
    auto result=lidar_viewer::algorithms::RunGroundSegmentation(frame.points,params);
    result.non_ground_points.insert(result.non_ground_points.end(),skipped.begin(),skipped.end());
    obstacles=result.non_ground_points.size();
    frame.points=std::move(result.non_ground_points);
    frame.points.reserve(frame.points.size()+result.ground_points.size());
    for (auto point : result.ground_points) {point.intensity=-1; frame.points.push_back(point);}
    ++state.ground_passes;
  } else state.exclusion_removed_count=exclusions.CountRejected(frame.points);
  // Skipped points remain visible without a ground label. Only the remaining
  // points enter segmentation; the learned detector always receives the scan.
  ApplyRoiResult(state,frame,obstacles,coordinates);
  state.processed_frame=std::move(frame); state.processed_obstacles=obstacles;
  if (state.imu_level_enabled && leveling.applied!=state.imu_level_result.applied)
    std::cout << "IMU leveling " << (leveling.applied ? "active: " : "unavailable: ") << leveling.status << std::endl;
  state.imu_level_result=std::move(leveling);
  state.processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

void CancelGeometry(ViewerState & state)
{
  if (state.geometry_worker) state.geometry_worker->CancelPending();
  state.geometry_future={};state.geometry_index.reset();state.geometry_live_sequence.reset();
  state.applying_exclusions=false;
}

void QueueGeometry(ViewerState & state, FrameData base, std::optional<std::size_t> index={},
  std::optional<std::uint64_t> sequence={}, bool update_detector=true, bool roi_only=false)
{
  if (state.loading_bag_source) return;
  if (roi_only && !lidar_viewer::SameRoiExclusions(state.analysis_roi,state.applied_exclusions)) {
    roi_only=false;
  }
  update_detector=update_detector || (state.algorithm_config.detection.enabled && !state.detector_input);
  // The worker owns a snapshot; it never reads mutable UI state or uses OpenGL.
  auto work=std::make_unique<ViewerState>();
  work->base_frame=std::move(base);work->global_ground_offset=state.global_ground_offset;
  work->calibration=state.calibration;work->imu_level_enabled=state.imu_level_enabled;
  work->imu_mount=state.imu_mount;work->algorithm_config=state.algorithm_config;
  work->imu_level_result=state.imu_level_result;
  work->ground_removal_enabled=state.ground_removal_enabled;work->ground_sequence=state.ground_sequence;
  work->roi=state.roi;work->display_roi=state.display_roi;work->roi_frame=state.roi_frame;
  work->analysis_roi=state.analysis_roi;work->applied_exclusions=state.analysis_roi;
  if (roi_only) {
    work->processed_frame=state.processed_frame;work->processed_obstacles=state.processed_obstacles;
    work->processing_ms=state.processing_ms;
  }
  if (!state.geometry_worker) state.geometry_worker=std::make_unique<lidar_viewer::BackgroundWorker>();
  state.geometry_future=state.geometry_worker->Submit([work=std::move(work),update_detector,roi_only]() mutable {
    if (roi_only) {
      ApplyRoiResult(*work,work->processed_frame,work->processed_obstacles,work->roi_frame);
      if (update_detector) RefreshDetectorInput(*work);
    }
    else ApplyCalibration(*work,nullptr,std::nullopt,update_detector);
    return std::move(work);
  });
  state.geometry_index=index;state.geometry_live_sequence=sequence;
  state.geometry_updates_detector=update_detector;state.geometry_roi_only=roi_only;
  state.submitted_geometry_config=state.geometry_config;state.roi_pending=false;
  state.submitted_ground_config=state.geometry_ground_config;
  state.submitted_input_config=state.geometry_input_config;
}

void RebuildGrid(ViewerState & state)
{
  state.grid.vertices.clear();

  const int count = std::max(1, state.grid_settings.count);
  const float spacing = std::max(0.1f, state.grid_settings.spacing);
  const float extent = static_cast<float>(count) * spacing;
  const float z = 0.0f;

  state.grid.vertices.reserve(static_cast<std::size_t>((count * 2 + 1) * 4));
  for (int i = -count; i <= count; ++i) {
    const float offset = static_cast<float>(i) * spacing;
    state.grid.vertices.push_back({-extent, offset, z});
    state.grid.vertices.push_back({ extent, offset, z});
    state.grid.vertices.push_back({offset, -extent, z});
    state.grid.vertices.push_back({offset,  extent, z});
  }

  glBindVertexArray(state.grid.vao);
  glBindBuffer(GL_ARRAY_BUFFER, state.grid.vbo);
  glBufferData(
    GL_ARRAY_BUFFER,
    static_cast<GLsizeiptr>(state.grid.vertices.size() * sizeof(Vec3)),
    state.grid.vertices.data(),
    GL_STATIC_DRAW);
  state.grid.dirty = false;
}

constexpr std::size_t kMaxCachedFrames = 3;
constexpr std::size_t kFrameCacheBytes = 64u*1024u*1024u;

void StoreFrameInCache(ViewerState & state, std::size_t index, FrameData frame)
{
  auto cache_it = state.frame_cache.find(index);
  if (cache_it == state.frame_cache.end()) {
    state.frame_cache_order.push_back(index);
  } else {
    cache_it->second = std::move(frame);
    return;
  }

  state.frame_cache.emplace(index, std::move(frame));
  std::size_t bytes=0;
  for (const auto & entry:state.frame_cache) bytes+=entry.second.points.capacity()*sizeof(KittiPoint);
  // Retain one oversized frame so it can still be consumed without reloading.
  while (state.frame_cache_order.size() > 1 &&
      (state.frame_cache_order.size() > kMaxCachedFrames || bytes > kFrameCacheBytes)) {
    const std::size_t evict_index = state.frame_cache_order.front();
    state.frame_cache_order.pop_front();
    bytes-=state.frame_cache.at(evict_index).points.capacity()*sizeof(KittiPoint);
    state.frame_cache.erase(evict_index);
  }
}

void ConsumeReadyPrefetch(ViewerState & state)
{
  if (!state.prefetched_index.has_value() || !state.prefetch_future.valid()) {
    return;
  }

  if (state.prefetch_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    return;
  }

  try {
    StoreFrameInCache(state, *state.prefetched_index, state.prefetch_future.get());
  } catch (const std::exception & error) {
    state.is_playing=false;state.requested_frame.reset();
    state.ui_error=std::string("Frame could not be loaded: ")+error.what();
  }
  state.prefetched_index.reset();
}

void QueueFramePrefetch(ViewerState & state, std::size_t next_index)
{
  if (state.frames.empty()) {
    return;
  }

  ConsumeReadyPrefetch(state);
  if (state.frame_cache.find(next_index) != state.frame_cache.end()) {
    return;
  }
  if (state.prefetched_index.has_value()) {
    if (*state.prefetched_index == next_index) {
      return;
    }
    if (state.prefetch_future.valid() &&
      state.prefetch_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
      return;
    }
    ConsumeReadyPrefetch(state);
  }

  const FrameHandle handle = state.frames[next_index];
  state.prefetched_index = next_index;
  if (!state.frame_reader) state.frame_reader=std::make_unique<lidar_viewer::BackgroundWorker>();
  state.prefetch_future = state.frame_reader->Submit([handle]() {
    return LoadFrame(handle);
  });
}

void QueueNextFramePrefetch(ViewerState & state)
{
  if (state.frames.size()>1) QueueFramePrefetch(state,(state.current_index+1)%state.frames.size());
}

bool CompleteGeometry(ViewerState & state)
{
  if (!state.geometry_future.valid() || state.roi_gesture.active ||
      state.geometry_future.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return false;
  try {
    auto work=state.geometry_future.get();
    // A settings edit supersedes an in-flight result. Recompute the requested
    // source with the newest settings instead of showing boxes in old axes.
    if (state.submitted_geometry_config!=state.geometry_config) {
      if (state.submitted_ground_config!=state.geometry_ground_config) {
        QueueGeometry(state,std::move(work->base_frame),state.geometry_index,state.geometry_live_sequence,
          state.geometry_updates_detector || state.submitted_input_config!=state.geometry_input_config,false);
      } else {
        // ROI/cluster edits reuse the in-flight segmentation too. Preserve the
        // fresh scan's aligned detector snapshot instead of computing it twice.
        work->roi=state.roi;work->display_roi=state.display_roi;work->algorithm_config.clustering=state.algorithm_config.clustering;
        work->algorithm_config.detection=state.algorithm_config.detection;
        const bool prepare_detector=work->algorithm_config.detection.enabled &&
          (state.geometry_updates_detector || !state.detector_input);
        state.geometry_updates_detector|=prepare_detector;
        state.geometry_future=state.geometry_worker->Submit([work=std::move(work),prepare_detector]() mutable {
          ApplyRoiResult(*work,work->processed_frame,work->processed_obstacles,work->roi_frame);
          if (prepare_detector) RefreshDetectorInput(*work);
          return std::move(work);
        });
        state.submitted_geometry_config=state.geometry_config;
      }
      return false;
    }
    state.base_frame=std::move(work->base_frame);state.frame=std::move(work->frame);
    state.applied_exclusions=std::move(work->applied_exclusions);
    state.processed_frame=std::move(work->processed_frame);state.processed_obstacles=work->processed_obstacles;
    state.ground_removed_count=work->ground_removed_count;state.clusters=std::move(work->clusters);
    if (!state.geometry_roi_only) state.exclusion_removed_count=work->exclusion_removed_count;
    state.roi_frame=work->roi_frame;state.roi_ms=work->roi_ms;state.processing_ms=work->processing_ms;
    state.ground_passes+=work->ground_passes;state.imu_level_result=std::move(work->imu_level_result);
    if (state.geometry_updates_detector) {
      state.detector_input=std::move(work->detector_input);state.detector_yaw=work->detector_yaw;
      ++state.detector_revision;state.full_detections.clear();state.detector_error.clear();
    }
    RebuildRoiGeometry(state);FilterDetections(state);state.frame_dirty=true;
    if (state.applying_exclusions) {
      state.applying_exclusions=false;SetTransientStatus(state,"ROI applied to ground segmentation");
    }
    if (state.geometry_roi_only)
      std::cout<<"ROI applied: "<<state.roi.boxes.size()<<" boxes | "<<state.frame.points.size()<<" points | "
        <<state.roi_ms<<" ms | ground passes "<<state.ground_passes<<std::endl;
    if (std::getenv("LIDAR_VIEWER_ROI_TRACE"))
      std::cout<<"Analysis input: excluded "<<state.exclusion_removed_count<<" | detector "
        <<(state.detector_input ? state.detector_input->size() : 0)<<" | ground passes "<<state.ground_passes<<std::endl;
    if (state.geometry_index) {
      state.current_index=*state.geometry_index;UpdateCurrentFileGroup(state);QueueNextFramePrefetch(state);
    }
    if (state.geometry_live_sequence) {
      state.live_display_sequence=*state.geometry_live_sequence;
      if (!state.live_has_frame) {
        state.fit_view_requested=true;
        std::cout<<"Live first frame: "<<state.frame.points.size()<<" points"<<std::endl;
      }
      state.live_has_frame=true;
    }
    state.geometry_index.reset();state.geometry_live_sequence.reset();
    return true;
  } catch (const std::exception & error) {
    if (state.applying_exclusions) {
      state.analysis_roi=state.applied_exclusions;state.applying_exclusions=false;
    }
    state.ui_error=std::string("Frame processing failed: ")+error.what();state.is_playing=false;
    if (state.live_mode) state.live_paused=true;
    state.geometry_index.reset();state.geometry_live_sequence.reset();return false;
  }
}

void LoadFrameByIndex(ViewerState & state, std::size_t index)
{
  if (state.frames.empty()) {
    throw std::runtime_error("No frames are available");
  }
  const auto next_index = index % state.frames.size();
  ConsumeReadyPrefetch(state);
  FrameData next_base;
  const auto cached = state.frame_cache.find(next_index);
  if (cached != state.frame_cache.end()) {
    next_base = cached->second;
  } else if (state.prefetched_index && *state.prefetched_index == next_index && state.prefetch_future.valid()) {
    state.prefetched_index.reset();
    next_base = state.prefetch_future.get();
    StoreFrameInCache(state, next_index, next_base);
  } else {
    next_base = LoadFrame(state.frames[next_index]);
    StoreFrameInCache(state, next_index, next_base);
  }
  float offset = state.global_ground_offset;
  if (!state.alignment_initialized) {
    const auto format = state.frames[next_index].format;
    offset = (format == InputFormat::OusterLegacyContainer || format == InputFormat::Ros2Bag)
      ? 0.0f : next_base.ground_z;
  }
  ApplyGlobalAlignment(next_base, offset);
  if (state.alignment_initialized && next_index!=state.current_index+1) ++state.ground_sequence;
  if (state.async_geometry) {
    state.global_ground_offset=offset;state.alignment_initialized=true;
    QueueGeometry(state,std::move(next_base),next_index);
    if (state.frames.size()>1) QueueFramePrefetch(state,(next_index+1)%state.frames.size());
    return;
  }
  ApplyCalibration(state, &next_base, offset);
  state.base_frame = std::move(next_base);
  state.global_ground_offset = offset;
  state.alignment_initialized = true;
  state.current_index = next_index;
  UpdateCurrentFileGroup(state);
  QueueNextFramePrefetch(state);
}

bool TryLoadFrame(ViewerState & state, std::size_t index)
{
  try {
    if (state.frames.empty()) return false;
    index%=state.frames.size();
    if (state.geometry_future.valid() && state.geometry_index==index) return false;
    ConsumeReadyPrefetch(state);
    if (state.frame_cache.find(index)==state.frame_cache.end()) {
      state.requested_frame=index;
      QueueFramePrefetch(state,index);
      return false;
    }
    state.requested_frame.reset();
    LoadFrameByIndex(state, index);
    state.ui_error.clear();
    state.last_frame_advance_time = glfwGetTime();
    return !state.async_geometry;
  } catch (const std::exception & error) {
    state.is_playing = false;
    state.ui_error = std::string("Frame could not be loaded: ") + error.what();
    std::cerr << state.ui_error << '\n';
    return false;
  }
}

bool CompleteFrameRequest(ViewerState & state)
{
  if (state.loading_bag_source) return false;
  ConsumeReadyPrefetch(state);
  if (!state.requested_frame) return false;
  return TryLoadFrame(state,*state.requested_frame);
}


void MarkGridDirty(ViewerState & state)
{
  state.grid_settings.spacing = std::clamp(state.grid_settings.spacing, 0.1f, 50.0f);
  state.grid_settings.thickness = std::clamp(state.grid_settings.thickness, 1.0f, 6.0f);
  state.grid_settings.count = std::clamp(state.grid_settings.count, 2, 100);
  state.grid.dirty = true;
}

void SetTransientStatus(ViewerState & state, const std::string & message)
{
  state.transient_status = message;
  state.transient_status_until = glfwGetTime() + 2.0;
}

std::optional<double> EstimateInputFps(const ViewerState & state)
{
  if (state.frames.empty()) {
    return std::nullopt;
  }

  if (state.frames.size() <= 1 && state.frames.front().format == InputFormat::KittiXyzi) {
    return std::nullopt;
  }

  switch (state.frames.front().format) {
    case InputFormat::Ros2Bag: {
      std::vector<double> intervals;
      for (std::size_t i = 1; i < state.frames.size(); ++i) {
        if (state.frames[i].path != state.frames[i - 1].path) {
          continue;
        }
        const auto delta = state.frames[i].bag_timestamp_ns - state.frames[i - 1].bag_timestamp_ns;
        if (delta > 0) {
          intervals.push_back(static_cast<double>(delta) * 1e-9);
        }
      }
      if (intervals.empty()) {
        return std::nullopt;
      }
      const auto middle = intervals.begin() + static_cast<std::ptrdiff_t>(intervals.size() / 2);
      std::nth_element(intervals.begin(), middle, intervals.end());
      return 1.0 / *middle;
    }
    case InputFormat::OusterLegacyContainer:
      return 10.0;
    case InputFormat::KittiXyzi:
      return 10.0;
    default:
      return std::nullopt;
  }
}

void SyncPlaybackFpsToInput(ViewerState & state)
{
  if (state.input_fps_estimate.has_value()) {
    state.playback_fps = std::clamp(*state.input_fps_estimate, 0.1, 120.0);
  }
}

void LoadSettings(ViewerState & state);
fs::path AlgorithmsConfigPath();
std::string DetectDatasetProfile(const fs::path & input_path);

void OpenBagSource(ViewerState & state, std::size_t index)
{
  if (state.live.Active() || state.loading_bag_source || index >= state.bag_sources.size() ||
      (state.current_bag_source && *state.current_bag_source == index)) return;
  state.input_load_previous_playing=state.is_playing;
  state.is_playing = false;
  state.requested_frame.reset();
  state.timeline_dragging = false;
  try {
    const auto path = state.bag_sources[index].path;
    const auto topic = state.bag_topic;
    const auto imu_topic = state.bag_imu_topic;
    auto load=std::make_shared<lidar_viewer::InputLoadState>();
    auto prepared=std::make_unique<ViewerState>();
    auto & staged=*prepared;
    const auto profile=DetectDatasetProfile(path);
    const bool change_profile = state.dataset_profile != profile;
    staged.dataset_profile = profile;
    staged.detector_cli_override=state.detector_cli_override;
    if (change_profile) {
      staged.calibration = {0, 0, 0, 0};
      staged.algorithm_config = lidar_viewer::LoadPipelineConfig(AlgorithmsConfigPath());
      LoadSettings(staged);
      SetProjectionMode(staged,state.camera.projection_mode);
      if (state.bag_sensor_height_override) staged.calibration.z_m = *state.bag_sensor_height_override;
      if (state.imu_level_cli_override) staged.imu_level_enabled = true;
    } else {
      staged.calibration = state.calibration;
      staged.algorithm_config = state.algorithm_config;
      staged.ground_removal_enabled = state.ground_removal_enabled;
      staged.ground_view=state.ground_view;
      staged.imu_level_enabled = state.imu_level_enabled;
      staged.imu_mount = state.imu_mount;
      staged.roi = state.roi;
      staged.display_roi=state.display_roi;
      staged.selected_display_roi=state.selected_display_roi;
      staged.analysis_roi=state.analysis_roi;staged.applied_exclusions=state.analysis_roi;
      staged.selected_roi = state.selected_roi;
      staged.object_method=state.object_method;
    }
    CancelGeometry(state);
    if (!state.geometry_worker) state.geometry_worker=std::make_unique<lidar_viewer::BackgroundWorker>();
    state.bag_load_future = state.geometry_worker->Submit([path, topic, imu_topic, load, prepared=std::move(prepared)]() mutable {
      ViewerState::BagLoadResult result;
      result.frames = ResolveInputFrames(path, topic, imu_topic, load.get());
      if (result.frames.empty()) throw std::runtime_error("No point cloud frames in this bag");
      load->Stage(lidar_viewer::InputLoadStage::ReadingFirstFrame);
      result.first_frame = LoadFrame(result.frames.front());
      load->Stage(lidar_viewer::InputLoadStage::PreparingDisplay);
      result.ground_offset=result.frames.front().format==InputFormat::KittiXyzi ? result.first_frame.ground_z : 0.f;
      ApplyGlobalAlignment(result.first_frame,result.ground_offset);
      ApplyCalibration(*prepared,&result.first_frame,result.ground_offset);
      result.prepared=std::move(prepared);
      load->CheckCancelled();
      return result;
    });
    state.input_load=std::move(load);
    state.input_load_started=std::chrono::steady_clock::now();
    state.input_load_autoplay=false;
    state.loading_bag_source = index;
    state.ui_error.clear();
  } catch (const std::exception & error) {
    state.ui_error = std::string("Input could not be opened: ") + error.what();
    state.is_playing=state.input_load_previous_playing;
  }
}

void StartBagPicker(ViewerState & state)
{
  if (state.live.Active() || state.loading_bag_source || state.folder_scan_future.valid() || state.bag_picker.Active() || state.dataset_picker.Active()) return;
  try {
    fs::path initial;
    if (!state.frames.empty()) {
      const auto current = state.frames.front().path;
      initial = fs::is_directory(current) ? current : current.parent_path();
    } else if (!state.data_folders.empty()) {
      initial = state.data_folders.front();
    } else if (const char * home = std::getenv("HOME")) {
      initial = home;
    }
    state.bag_picker.OpenBag(initial);
    state.ui_error.clear();
  } catch (const std::exception & error) {
    state.ui_error = error.what();
  }
}

void UnhideBagSource(ViewerState & state,const fs::path & path)
{
  auto hidden=state.hidden_sources;
  hidden.erase(std::remove(hidden.begin(),hidden.end(),path),hidden.end());
  if (hidden==state.hidden_sources) return;
  SaveHiddenBagSources(state.hidden_sources_config,hidden);
  state.hidden_sources=std::move(hidden);
}

void RemoveBagSource(ViewerState & state,const fs::path & path)
{
  if (state.loading_bag_source) return;
  const auto found=std::find_if(state.bag_sources.begin(),state.bag_sources.end(),[&](const BagSource & source) {return source.path==path;});
  if (found==state.bag_sources.end()) return;
  const auto index=static_cast<std::size_t>(found-state.bag_sources.begin());
  auto hidden=state.hidden_sources;hidden.push_back(path);
  SaveHiddenBagSources(state.hidden_sources_config,hidden);
  state.hidden_sources=std::move(hidden);
  state.bag_sources.erase(state.bag_sources.begin()+static_cast<std::ptrdiff_t>(index));
  if (state.current_bag_source) {
    if (*state.current_bag_source==index) state.current_bag_source.reset();
    else if (*state.current_bag_source>index) --*state.current_bag_source;
  }
  state.bag_durations.erase(path);state.unavailable_sources.erase(path);
  SetTransientStatus(state,"Removed from list. Files are kept.");
  if (std::getenv("LIDAR_VIEWER_CATALOG_TRACE")) std::cout<<"Source removed from list: "<<path<<std::endl;
}

void RestoreBagSources(ViewerState & state)
{
  std::vector<BagSource> sources;
  for (const auto & path:state.hidden_sources) sources.push_back(MakeBagSource(path));
  auto merged=MergeBagSources(state.bag_sources,sources);
  std::optional<fs::path> current;
  if (state.current_bag_source) current=state.bag_sources[*state.current_bag_source].path;
  else if (!state.frames.empty() && state.frames.front().format==InputFormat::Ros2Bag) current=state.frames.front().path;
  SaveHiddenBagSources(state.hidden_sources_config,{});
  state.hidden_sources.clear();state.bag_sources=std::move(merged);state.current_bag_source.reset();
  for (std::size_t i=0;i<state.bag_sources.size();++i)
    if (current && state.bag_sources[i].path==*current) state.current_bag_source=i;
  state.check_source_availability=true;
  SetTransientStatus(state,"Removed recordings restored to the list.");
}

void OpenInputPath(ViewerState & state, const fs::path & input)
{
  if (state.live.Active() || state.loading_bag_source || state.folder_scan_future.valid())
    throw std::runtime_error("Disconnect live input or wait for loading before opening a file");
  const auto path=input.filename()=="metadata.json" ? input.parent_path() : input;
  const bool dataset=lidar_viewer::IsDatasetFile(path) || (fs::is_directory(path) && !IsRos2BagInput(path));
  auto sources=dataset ? std::vector<BagSource>{} : DiscoverBagSources(path);
  if(dataset) {const auto canonical=fs::canonical(path);sources.push_back({canonical,canonical.filename().u8string(),{}, {}});}
  if (sources.empty() && input.extension()==".bin" && fs::is_regular_file(input)) {
    const auto path=fs::canonical(input);
    sources.push_back({path,path.filename().u8string(),{}, {}});
  }
  if (sources.size() != 1)
    throw std::runtime_error("Select a point cloud file, dataset folder or ROS 2 bag");
  const auto selected = sources.front().path;
  UnhideBagSource(state,selected);
  std::optional<fs::path> current;
  if (state.current_bag_source) current = state.bag_sources[*state.current_bag_source].path;
  auto merged = MergeBagSources(state.bag_sources, sources);
  state.bag_sources = std::move(merged);
  state.current_bag_source.reset();
  std::size_t selected_index = 0;
  for (std::size_t i = 0; i < state.bag_sources.size(); ++i) {
    if (current && state.bag_sources[i].path == *current) state.current_bag_source = i;
    if (state.bag_sources[i].path == selected) selected_index = i;
  }
  state.bag_date_filter.clear();
  state.bag_search.fill('\0');
  OpenBagSource(state, selected_index);
}

void DropCallback(GLFWwindow * window, int count, const char ** paths)
{
  auto * state=static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state || count<1) return;
  try {
    if (count!=1) throw std::runtime_error("Drop one point cloud file, dataset folder or bag at a time");
    if (state->bag_picker.Active() || state->dataset_picker.Active()) return;
    OpenInputPath(*state,fs::u8path(paths[0]));
  } catch (const std::exception & error) {state->ui_error=error.what();}
}

void CompleteBagPicker(ViewerState & state, GLFWwindow * window)
{
  try {
    const auto result = state.bag_picker.Poll();
    if (!result) return;
    glfwFocusWindow(window);
    // Native dialogs can hide the cursor without a GLFW motion event. Restore
    // its position so a second click at the same button works immediately.
    double cursor_x = 0, cursor_y = 0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);
    ImGui_ImplGlfw_CursorPosCallback(window, cursor_x, cursor_y);
    if (!result->error.empty()) throw std::runtime_error(result->error);
    if (result->path.empty()) return;  // Cancel leaves the current frame intact.
    OpenInputPath(state,result->path);
  } catch (const std::exception & error) {
    state.ui_error = std::string("Input could not be opened: ") + error.what();
  }
}

void CompleteDatasetPicker(ViewerState & state,GLFWwindow * window)
{
  try {
    const auto result=state.dataset_picker.Poll();if(!result) return;
    glfwFocusWindow(window);
    if(!result->error.empty()) throw std::runtime_error(result->error);
    if(!result->path.empty()) OpenInputPath(state,result->path);
  } catch(const std::exception & error) {state.ui_error=error.what();}
}

void CompleteBagLoad(ViewerState & state)
{
  if (!state.loading_bag_source || !state.bag_load_future.valid() ||
      state.bag_load_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
  const auto selected = *state.loading_bag_source;
  try {
    auto result = state.bag_load_future.get();
    if (state.input_load) state.input_load->CheckCancelled();
    auto & staged=*result.prepared;
    if(!result.frames.empty() && result.frames.front().dataset_sample) state.dataset_tab_requested=true;
    state.dataset_fitted_index.reset();
    state.dataset_list_index.reset();
    const auto & profile=staged.dataset_profile;
    const bool change_profile=state.dataset_profile!=profile;
    const float ground_offset=result.ground_offset;
    // A packaged-task future can be discarded without waiting for old disk I/O.
    // The old job owns its handle and cannot insert into the replacement cache.
    if (state.frame_reader) state.frame_reader->CancelPending();
    state.prefetch_future={};
    CancelGeometry(state);
    state.prefetched_index.reset();
    state.requested_frame.reset();
    state.frame_cache.clear();
    state.frame_cache_order.clear();
    state.frames = std::move(result.frames);
    ResetTracking(state);
    state.base_frame = std::move(result.first_frame);
    state.live_mode = false;
    state.live_has_frame = false;
    state.frame = std::move(staged.frame);
    state.detector_input=std::move(staged.detector_input);state.detector_yaw=staged.detector_yaw;
    ++state.detector_revision;state.full_detections.clear();
    state.processed_frame=std::move(staged.processed_frame);
    state.processed_obstacles=staged.processed_obstacles;
    state.processing_ms=staged.processing_ms; state.roi_ms=staged.roi_ms;
    state.ground_passes+=staged.ground_passes;state.ground_sequence=staged.ground_sequence;
    state.roi_pending=false; state.roi_gesture.active=false; state.roi_tool=ViewerState::RoiTool::Navigate;
    if (change_profile) {
      state.saved_roi=staged.saved_roi;state.saved_display_roi=staged.saved_display_roi;
      state.roi_history.Clear();state.display_roi_history.Clear();
      state.roi_purpose=staged.roi_purpose;
    }
    state.ground_removed_count = staged.ground_removed_count;
    state.clusters = std::move(staged.clusters);
    state.detections = std::move(staged.detections);
    state.cluster_boxes.batches = std::move(staged.cluster_boxes.batches);
    state.cluster_boxes.dirty = true;
    state.frame_dirty = true;
    state.imu_level_result = std::move(staged.imu_level_result);
    state.roi = std::move(staged.roi);
    state.display_roi=std::move(staged.display_roi);state.display_roi_pending=false;
    state.selected_display_roi=staged.selected_display_roi;
    state.analysis_roi=std::move(staged.analysis_roi);state.applied_exclusions=std::move(staged.applied_exclusions);
    state.exclusion_removed_count=staged.exclusion_removed_count;
    state.selected_roi = staged.selected_roi;
    state.roi_frame = staged.roi_frame;
    RebuildRoiGeometry(state);
    if (change_profile) {
      state.calibration = staged.calibration;
      state.imu_level_enabled = staged.imu_level_enabled;
      state.imu_mount = staged.imu_mount;
      state.algorithm_config = staged.algorithm_config;
      state.object_method=staged.object_method;
      state.tracking_params=staged.tracking_params;
      state.ground_removal_enabled = staged.ground_removal_enabled;
      state.ground_view=staged.ground_view;
      state.camera = staged.camera;
      state.grid_settings = staged.grid_settings;
      state.grid.dirty = true;
      state.color_mode = staged.color_mode;
      state.height_color_min = staged.height_color_min;
      state.height_color_max = staged.height_color_max;
      state.loop_playback = staged.loop_playback;
    }
    state.current_index = 0;
    state.current_bag_source = selected;
    state.global_ground_offset = ground_offset;
    state.alignment_initialized = true;
    state.dataset_profile = profile;
    BuildFileGroups(state);
    state.input_fps_estimate = EstimateInputFps(state);
    SyncPlaybackFpsToInput(state);
    state.current_playback_fps = 0;
    state.playback_measurement_advances = 0;
    state.playback_measurement_started_at = glfwGetTime();
    state.seek_draft = 0;
    state.fit_view_requested = true;
    state.ui_error.clear();
    SetTransientStatus(state, "Input opened - press Play to start");
    std::cout << (state.frames.front().format==InputFormat::Ros2Bag ? "Opened bag: " : "Opened point cloud: ")
              << state.bag_sources[selected].path << " | "
              << state.frames.size() << " frames" << std::endl;
  } catch (const lidar_viewer::InputLoadCancelled &) {
    SetTransientStatus(state,"Opening canceled");
    std::cout << "Opening canceled; current input preserved" << std::endl;
    state.input_load_autoplay=state.input_load_previous_playing && !state.frames.empty();
  } catch (const std::exception & error) {
    state.ui_error = std::string("Input could not be opened: ") + error.what();
    std::cerr << state.ui_error << std::endl;
    state.input_load_autoplay=state.input_load_previous_playing && !state.frames.empty();
  }
  state.loading_bag_source.reset();
  state.input_load.reset();
  state.is_playing = state.input_load_autoplay;
  state.input_load_autoplay=false;
}

fs::path DataFoldersConfigPath()
{
  return lidar_viewer::UserConfigurationDirectory()/"data_folders.cfg";
}

fs::path ExpandFolderPath(std::string path)
{
  if (path == "~" || path.rfind("~/", 0) == 0) {
    return lidar_viewer::UserHome()/fs::u8path(path.size()>2 ? path.substr(2) : "");
  }
  return fs::u8path(path);
}

void SetLivePaused(ViewerState & state, bool paused)
{
  if (!state.live.Active() || state.live_paused == paused) return;
  if (paused && !state.live_has_frame) return;
  state.live_paused = paused;
  // A newer live frame may already be on the geometry worker. Dropping its
  // future keeps the exact displayed scan frozen; the receiver still retains
  // the latest scan for resume. Same-frame settings jobs have no live sequence.
  if (paused && state.geometry_live_sequence) CancelGeometry(state);
  std::cout << "Live display " << (paused ? "paused" : "resumed") << " at frame "
            << state.live_display_sequence << " | receiver remains active" << std::endl;
}

void ConnectLive(ViewerState & state)
{
  if (state.live.Active() || state.loading_bag_source || state.bag_picker.Active() || state.folder_scan_future.valid()) return;
  state.live.Connect({state.live_points_topic.data(), state.live_imu_topic.data()});
  if (state.dataset_profile != "rosbag") {
    state.dataset_profile = "rosbag";
    state.calibration = {};
    LoadSettings(state);
    if (state.bag_sensor_height_override) state.calibration.z_m = *state.bag_sensor_height_override;
    if (state.imu_level_cli_override) state.imu_level_enabled = true;
  }
  if (state.frame_reader) state.frame_reader->CancelPending();
  state.prefetch_future={};
  CancelGeometry(state);
  state.prefetched_index.reset(); state.requested_frame.reset(); state.frame_cache.clear(); state.frame_cache_order.clear();
  state.frames.clear(); state.file_groups.clear(); state.current_bag_source.reset();
  ResetTracking(state);
  state.base_frame = {}; state.frame = {}; state.processed_frame={}; state.processed_obstacles=0;
  state.roi_gesture.active=false;state.roi_pending=false;state.display_roi_pending=false;
  state.roi_tool=ViewerState::RoiTool::Navigate;
  state.exclusion_removed_count=0;
  state.clusters.clear(); state.detections.clear();state.full_detections.clear();state.detector_input.reset();++state.detector_revision;
  RebuildClusterBoxes(state);
  state.roi_renderer.vertices.clear(); state.roi_renderer.dirty = true;
  state.global_ground_offset = 0; state.alignment_initialized = true;
  state.current_index = 0; state.current_file_group = 0; state.seek_draft = 0;
  state.is_playing = false; state.timeline_dragging = false; state.ground_removed_count = 0;
  state.live_mode = true; state.live_paused = false; state.live_has_frame = false; state.live_display_sequence = 0;
  state.live_stats = state.live.Stats(); state.frame_dirty = true;
  state.ui_error.clear();
  std::cout << "Live connected: " << state.live_points_topic.data() << " | IMU "
            << state.live_imu_topic.data() << std::endl;
}

void UpdateLiveFrame(ViewerState & state)
{
  if (!state.live_mode) return;
  state.live_stats = state.live.Stats();
  if (!state.live_stats.connected || state.live_paused || state.geometry_future.valid() || DetectorFramePending(state)) return;
  auto next = state.live.TakeLatest();
  if (!next) return;
  try {
    if (state.async_geometry) {QueueGeometry(state,std::move(next->frame),{},next->sequence);return;}
    ApplyCalibration(state, &next->frame, 0);
    state.base_frame = std::move(next->frame);
    state.live_display_sequence = next->sequence;
    if (!state.live_has_frame) {
      state.fit_view_requested = true;
      std::cout << "Live first frame: " << state.frame.points.size() << " points" << std::endl;
    }
    state.live_has_frame = true;
  } catch (const std::exception & error) {
    state.ui_error = std::string("Live frame could not be processed: ") + error.what();
    SetLivePaused(state, true);
  }
}

fs::path NewLiveRecordingPath(const ViewerState & state)
{
  const auto folder = ExpandFolderPath(state.live_record_folder.data());
  if (folder.empty()) throw std::runtime_error("Choose a recording folder");
  const auto now = std::chrono::system_clock::now();
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  const std::time_t korea_time = std::chrono::system_clock::to_time_t(now) + 9 * 3600;
  const std::tm date = lidar_viewer::UtcTime(korea_time);
  std::ostringstream name;
  name << "sensors_" << std::put_time(&date, "%Y%m%d_%H%M%S_") << std::setfill('0') << std::setw(6) << micros % 1000000;
  return folder / name.str();
}

void ToggleLiveRecording(ViewerState & state)
{
  if (state.live_stats.record_busy) {
    state.live.StopRecording();
    std::cout << "Live recording: finalizing" << std::endl;
  } else {
    const auto destination = NewLiveRecordingPath(state);
    state.live.StartRecording(destination);
    std::cout << "Live recording: " << destination << std::endl;
  }
  state.live_stats = state.live.Stats();
}

void BrowseFolder(ViewerState & state, const fs::path & folder)
{
  try {
    const auto canonical = fs::canonical(folder);
    if (!fs::is_directory(canonical)) throw std::runtime_error("Choose a directory");
    if (canonical.string().size() >= state.folder_path_input.size())
      throw std::runtime_error("Folder path is too long");
    std::vector<fs::path> children;
    std::error_code error;
    for (fs::directory_iterator it(canonical, fs::directory_options::skip_permission_denied, error), end;
         it != end; it.increment(error)) {
      if (error) break;
      if (it->is_directory(error) && it->path().filename().string().front() != '.') children.push_back(it->path());
      error.clear();
    }
    std::sort(children.begin(), children.end());
    state.browsing_folder = canonical;
    std::fill(state.folder_path_input.begin(), state.folder_path_input.end(), '\0');
    const auto text = canonical.string();
    std::copy(text.begin(), text.end(), state.folder_path_input.begin());
    state.folder_children = std::move(children);
    state.folder_error.clear();
  } catch (const std::exception & error) {
    state.folder_error = std::string("Folder could not be opened: ") + error.what();
  }
}

fs::path SettingsPath()
{
  return lidar_viewer::ProjectResourcePath(kSettingsFileName);
}

std::string DetectDatasetProfile(const fs::path & input_path)
{
  if(lidar_viewer::IsDatasetFile(input_path) || input_path.filename()=="metadata.json") return "dataset";
  if (IsRos2BagInput(input_path)) {
    return "rosbag";
  }
  const std::string path = input_path.string();
  if (path.find("gangnam") != std::string::npos ||
    path.find("2019_10_04") != std::string::npos)
  {
    return "gangnam";
  }
  if (path.find("2011_09_26") != std::string::npos ||
    path.find("kitti") != std::string::npos ||
    path.find("KITTI") != std::string::npos)
  {
    return "kitti";
  }
  if(fs::is_directory(input_path)) {
    for(const auto & entry:fs::directory_iterator(input_path,fs::directory_options::skip_permission_denied)) {
      if(lidar_viewer::IsDatasetFile(entry.path()) || entry.path().filename()=="metadata.json") return "dataset";
      if(entry.is_directory() && (fs::is_regular_file(entry.path()/"metadata.json") || fs::is_regular_file(entry.path()/"coord.npy"))) return "dataset";
    }
  }
  return "default";
}

fs::path ProfileSettingsPath(const std::string & profile)
{
#ifdef LIDAR_VIEWER_USER_SETTINGS
  return lidar_viewer::UserConfigurationDirectory()/"profiles"/(profile+".cfg");
#else
  return lidar_viewer::ProjectResourcePath(kProfilesDir) / (profile + ".cfg");
#endif
}

fs::path AlgorithmsConfigPath()
{
  return lidar_viewer::ProjectResourcePath(kAlgorithmsConfigPath);
}

void SaveSettings(ViewerState & state)
{
  if (state.applying_exclusions || !lidar_viewer::SameRoiExclusions(state.roi,state.applied_exclusions))
    throw std::runtime_error("Apply the edited ground ROI before saving the profile");
  for (const float value : {state.camera.point_size, state.camera.distance,
      state.calibration.yaw_deg, state.calibration.pitch_deg, state.calibration.roll_deg,
      state.calibration.z_m, state.grid_settings.spacing, state.grid_settings.thickness,
      state.height_color_min, state.height_color_max, state.ui_scale,
      state.imu_mount.roll_deg, state.imu_mount.pitch_deg, state.imu_mount.yaw_deg,state.tracking_params.max_missed_s}) {
    if (!std::isfinite(value)) throw std::runtime_error("Cannot save a non-finite setting");
  }
  for (const auto & value : {state.imu_mount.imu_frame, state.imu_mount.lidar_frame}) {
    if (value.find_first_of("\r\n") != std::string::npos || value.find('\0') != std::string::npos)
      throw std::runtime_error("Invalid IMU mount frame name");
  }
  const fs::path destination = ProfileSettingsPath(state.dataset_profile);
  std::error_code ec;
  fs::create_directories(destination.parent_path(), ec);
  if (ec) {
    throw std::system_error(ec, "Failed to create settings directory");
  }

  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<float>::max_digits10);

  out << "profile=" << state.dataset_profile << '\n';
  out << "grid_enabled=" << (state.grid_settings.enabled ? 1 : 0) << '\n';
  out << "grid_spacing=" << state.grid_settings.spacing << '\n';
  out << "grid_thickness=" << state.grid_settings.thickness << '\n';
  out << "grid_count=" << state.grid_settings.count << '\n';
  out << "color_mode=" << state.color_mode << '\n';
  out << "height_color_min=" << state.height_color_min << '\n';
  out << "height_color_max=" << state.height_color_max << '\n';
  out << "loop_playback=" << (state.loop_playback ? 1 : 0) << '\n';
  CameraState camera_snapshot = state.camera;
  StoreCurrentProjectionView(camera_snapshot);

  out << "point_size=" << camera_snapshot.point_size << '\n';
  out << "ground_cell_size_m=" << state.algorithm_config.ground_removal.candidate_grid_cell_m << '\n';
  out << "ground_mode=" << state.algorithm_config.ground_removal.mode << '\n';
  out << "fugseg_num_segments=" << state.algorithm_config.ground_removal.fugseg_num_segments << '\n';
  out << "fugseg_bin_size_m=" << state.algorithm_config.ground_removal.fugseg_bin_size_m << '\n';
  out << "fugseg_inlier_threshold_m=" << state.algorithm_config.ground_removal.fugseg_inlier_threshold_m << '\n';
  out << "fugseg_max_slope_change_deg=" << state.algorithm_config.ground_removal.fugseg_max_slope_change_deg << '\n';
  out << "fugseg_max_gap_m=" << state.algorithm_config.ground_removal.fugseg_max_gap_m << '\n';
  out << "fugseg_horizontal_noise_deg=" << state.algorithm_config.ground_removal.fugseg_horizontal_noise_deg << '\n';
  out << "fugseg_vertical_noise_deg=" << state.algorithm_config.ground_removal.fugseg_vertical_noise_deg << '\n';
  out << "fugseg_range_noise_m=" << state.algorithm_config.ground_removal.fugseg_range_noise_m << '\n';
  out << "ground_use_cuda=" << state.algorithm_config.ground_removal.use_cuda << '\n';
  out << "gseg3d_ground_inlier_threshold_m=" << state.algorithm_config.ground_removal.gseg3d_ground_inlier_threshold_m << '\n';
  out << "gseg3d_cell_size_x_m=" << state.algorithm_config.ground_removal.gseg3d_cell_size_x_m << '\n';
  out << "gseg3d_cell_size_y_m=" << state.algorithm_config.ground_removal.gseg3d_cell_size_y_m << '\n';
  out << "gseg3d_cuda_cell_size_m=" << state.algorithm_config.ground_removal.gseg3d_cuda_cell_size_m << '\n';
  out << "gseg3d_ground_height_m=" << state.algorithm_config.ground_removal.gseg3d_ground_height_m << '\n';
  out << "gseg3d_max_ground_height_deviation_m=" << state.algorithm_config.ground_removal.gseg3d_max_ground_height_deviation_m << '\n';
  out << "patchwork_distance_threshold_m=" << state.algorithm_config.ground_removal.patchwork_distance_threshold_m << '\n';
  out << "patchwork_seed_threshold_m=" << state.algorithm_config.ground_removal.patchwork_seed_threshold_m << '\n';
  out << "max_ground_angle_deg=" << state.algorithm_config.ground_removal.max_ground_angle_deg << '\n';
  out << "final_distance_threshold_m=" << state.algorithm_config.ground_removal.final_distance_threshold_m << '\n';
  out << "projection_mode=" <<
    (camera_snapshot.projection_mode == CameraState::ProjectionMode::View2D ? "2d" : "3d") << '\n';
  out << "camera_target_x=" << camera_snapshot.target.x << '\n';
  out << "camera_target_y=" << camera_snapshot.target.y << '\n';
  out << "camera_target_z=" << camera_snapshot.target.z << '\n';
  out << "camera_yaw_rad=" << camera_snapshot.yaw << '\n';
  out << "camera_pitch_rad=" << camera_snapshot.pitch << '\n';
  out << "camera_distance=" << camera_snapshot.distance << '\n';
  out << "camera_3d_target_x=" << camera_snapshot.saved_3d_view.target.x << '\n';
  out << "camera_3d_target_y=" << camera_snapshot.saved_3d_view.target.y << '\n';
  out << "camera_3d_target_z=" << camera_snapshot.saved_3d_view.target.z << '\n';
  out << "camera_3d_yaw_rad=" << camera_snapshot.saved_3d_view.yaw << '\n';
  out << "camera_3d_pitch_rad=" << camera_snapshot.saved_3d_view.pitch << '\n';
  out << "camera_3d_distance=" << camera_snapshot.saved_3d_view.distance << '\n';
  out << "camera_3d_valid=" << (camera_snapshot.saved_3d_view.valid ? 1 : 0) << '\n';
  out << "camera_2d_target_x=" << camera_snapshot.saved_2d_view.target.x << '\n';
  out << "camera_2d_target_y=" << camera_snapshot.saved_2d_view.target.y << '\n';
  out << "camera_2d_target_z=" << camera_snapshot.saved_2d_view.target.z << '\n';
  out << "camera_2d_yaw_rad=" << camera_snapshot.saved_2d_view.yaw << '\n';
  out << "camera_2d_pitch_rad=" << camera_snapshot.saved_2d_view.pitch << '\n';
  out << "camera_2d_distance=" << camera_snapshot.saved_2d_view.distance << '\n';
  out << "camera_2d_valid=" << (camera_snapshot.saved_2d_view.valid ? 1 : 0) << '\n';
  out << "ground_removal_enabled=" << (state.ground_removal_enabled ? 1 : 0) << '\n';
  out << "ground_view=" << GroundViewKey(state.ground_view) << '\n';
  out << "clustering_enabled=" << (state.algorithm_config.clustering.enabled ? 1 : 0) << '\n';
  out << "clustering_mode=" << state.algorithm_config.clustering.mode << '\n';
  out << "clustering_tolerance_m=" << state.algorithm_config.clustering.tolerance_m << '\n';
  out << "clustering_vertical_gap_m=" << state.algorithm_config.clustering.max_vertical_gap_m << '\n';
  out << "detection_enabled=" << (state.algorithm_config.detection.enabled ? 1 : 0) << '\n';
  out << "detection_model=" << state.algorithm_config.detection.model << '\n';
  out << "detection_min_score=" << state.algorithm_config.detection.min_confidence << '\n';
  out << "detection_classes=" << state.algorithm_config.detection.classes << '\n';
  out << "object_method=" << state.object_method << '\n';
  out << "calibration_yaw_deg=" << state.calibration.yaw_deg << '\n';
  out << "calibration_pitch_deg=" << state.calibration.pitch_deg << '\n';
  out << "calibration_roll_deg=" << state.calibration.roll_deg << '\n';
  out << "calibration_z_m=" << state.calibration.z_m << '\n';
  out << "imu_level_enabled=" << state.imu_level_enabled << '\n';
  out << "imu_mount_confirmed=" << state.imu_mount.confirmed << '\n';
  out << "imu_mount_imu_frame=" << state.imu_mount.imu_frame << '\n';
  out << "imu_mount_lidar_frame=" << state.imu_mount.lidar_frame << '\n';
  out << "imu_mount_roll_deg=" << state.imu_mount.roll_deg << '\n';
  out << "imu_mount_pitch_deg=" << state.imu_mount.pitch_deg << '\n';
  out << "imu_mount_yaw_deg=" << state.imu_mount.yaw_deg << '\n';
  out << "imu_mount_auto_axes=" << state.imu_mount.auto_axes << '\n';
  out << "tracking_enabled=" << state.tracking_params.enabled << '\n';
  out << "tracking_max_missed_s=" << state.tracking_params.max_missed_s << '\n';
  out << "tracking_confirmation_hits=" << state.tracking_params.confirmation_hits << '\n';
  out << "tracking_show_trails=" << state.tracking_params.show_trails << '\n';
  out << "tracking_show_speed=" << state.tracking_params.show_speed << '\n';
  out << lidar_viewer::SerializeRoiSettings(state.roi);
  out << lidar_viewer::SerializeRoiSettings(state.display_roi,"display_roi_");
  if (!out) {
    throw std::runtime_error("Failed to serialize settings");
  }
  lidar_viewer::settings::AtomicWriteText(destination, out.str());
  state.saved_roi=state.roi;
  state.saved_display_roi=state.display_roi;
}

void LoadSettingsFile(ViewerState & state, const fs::path & path)
{
  std::ifstream in(path);
  if (!in) {
    return;
  }

  using lidar_viewer::settings::ParseFiniteFloat;
  using lidar_viewer::settings::ParseStrictInt;
  float height_min = state.height_color_min;
  float height_max = state.height_color_max;
  std::size_t line_number = 0;
  std::string line;
  std::string settings_text;
  while (std::getline(in, line)) {
    settings_text += line + '\n';
    ++line_number;
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    try {
      if (key == "grid_enabled") {
        state.grid_settings.enabled = (ParseStrictInt(value) != 0);
      } else if (key == "grid_spacing") {
        state.grid_settings.spacing = ParseFiniteFloat(value);
      } else if (key == "grid_thickness") {
        state.grid_settings.thickness = ParseFiniteFloat(value);
      } else if (key == "grid_count") {
        state.grid_settings.count = ParseStrictInt(value);
      } else if (key == "color_mode") {
        state.color_mode = std::clamp(ParseStrictInt(value), 0, 2);
      } else if (key == "height_color_min") {
        height_min = std::clamp(ParseFiniteFloat(value), -100.0f, 100.0f);
      } else if (key == "height_color_max") {
        height_max = std::clamp(ParseFiniteFloat(value), -100.0f, 100.0f);
      } else if (key == "ui_scale") {
        state.ui_scale = std::clamp(ParseFiniteFloat(value), 0.75f, 1.5f);
      } else if (key == "loop_playback") {
        state.loop_playback = ParseStrictInt(value) != 0;
      } else if (key == "point_size") {
        state.camera.point_size = std::clamp(ParseFiniteFloat(value), 1.0f, 8.0f);
      } else if (key == "ground_cell_size_m") {
        state.algorithm_config.ground_removal.candidate_grid_cell_m =
          std::clamp(ParseFiniteFloat(value), 0.1f, 5.0f);
      } else if (key == "ground_mode") {
        if (value == "fast_grid" || value == "gseg3d" || value == "patchworkpp" || value == "fugseg" || value == "grid" || value == "ransac")
          state.algorithm_config.ground_removal.mode = value;
        else throw std::invalid_argument("Unknown ground segmentation method");
      } else if (key == "fugseg_num_segments") {
        state.algorithm_config.ground_removal.fugseg_num_segments=std::clamp(ParseStrictInt(value),30,720);
      } else if (key == "fugseg_bin_size_m") {
        state.algorithm_config.ground_removal.fugseg_bin_size_m=std::clamp(ParseFiniteFloat(value),0.25f,3.f);
      } else if (key == "fugseg_inlier_threshold_m") {
        state.algorithm_config.ground_removal.fugseg_inlier_threshold_m=std::clamp(ParseFiniteFloat(value),0.02f,0.5f);
      } else if (key == "fugseg_max_slope_change_deg") {
        state.algorithm_config.ground_removal.fugseg_max_slope_change_deg=std::clamp(ParseFiniteFloat(value),1.f,45.f);
      } else if (key == "fugseg_max_gap_m") {
        state.algorithm_config.ground_removal.fugseg_max_gap_m=std::clamp(ParseFiniteFloat(value),1.f,30.f);
      } else if (key == "fugseg_horizontal_noise_deg") {
        state.algorithm_config.ground_removal.fugseg_horizontal_noise_deg=std::clamp(ParseFiniteFloat(value),0.001f,1.f);
      } else if (key == "fugseg_vertical_noise_deg") {
        state.algorithm_config.ground_removal.fugseg_vertical_noise_deg=std::clamp(ParseFiniteFloat(value),0.001f,1.f);
      } else if (key == "fugseg_range_noise_m") {
        state.algorithm_config.ground_removal.fugseg_range_noise_m=std::clamp(ParseFiniteFloat(value),0.001f,0.5f);
      } else if (key == "ground_use_cuda") {
        state.algorithm_config.ground_removal.use_cuda = ParseStrictInt(value) != 0;
      } else if (key == "gseg3d_ground_inlier_threshold_m") {
        state.algorithm_config.ground_removal.gseg3d_ground_inlier_threshold_m =
          std::clamp(ParseFiniteFloat(value), .02f, 1.0f);
      } else if (key == "gseg3d_cell_size_x_m") {
        state.algorithm_config.ground_removal.gseg3d_cell_size_x_m =
          std::clamp(ParseFiniteFloat(value), 0.1f, 5.0f);
      } else if (key == "gseg3d_cell_size_y_m") {
        state.algorithm_config.ground_removal.gseg3d_cell_size_y_m =
          std::clamp(ParseFiniteFloat(value), 0.1f, 5.0f);
      } else if (key == "gseg3d_cuda_cell_size_m") {
        state.algorithm_config.ground_removal.gseg3d_cuda_cell_size_m =
          std::clamp(ParseFiniteFloat(value), 0.1f, 5.0f);
      } else if (key=="gseg3d_ground_height_m") {
        state.algorithm_config.ground_removal.gseg3d_ground_height_m=std::clamp(ParseFiniteFloat(value),-5.f,5.f);
      } else if (key=="gseg3d_max_ground_height_deviation_m") {
        state.algorithm_config.ground_removal.gseg3d_max_ground_height_deviation_m=std::clamp(ParseFiniteFloat(value),.01f,1.f);
      } else if (key=="patchwork_distance_threshold_m") {
        state.algorithm_config.ground_removal.patchwork_distance_threshold_m=std::clamp(ParseFiniteFloat(value),.02f,.5f);
      } else if (key=="patchwork_seed_threshold_m") {
        state.algorithm_config.ground_removal.patchwork_seed_threshold_m=std::clamp(ParseFiniteFloat(value),.02f,.5f);
      } else if (key=="max_ground_angle_deg") {
        state.algorithm_config.ground_removal.max_ground_angle_deg=std::clamp(ParseFiniteFloat(value),1.f,45.f);
      } else if (key=="final_distance_threshold_m") {
        state.algorithm_config.ground_removal.final_distance_threshold_m=std::clamp(ParseFiniteFloat(value),.02f,.5f);
      } else if (key == "projection_mode") {
        state.camera.projection_mode =
          (value == "2d") ? CameraState::ProjectionMode::View2D : CameraState::ProjectionMode::View3D;
      } else if (key == "camera_target_x") {
        state.camera.target.x = ParseFiniteFloat(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_target_y") {
        state.camera.target.y = ParseFiniteFloat(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_target_z") {
        state.camera.target.z = ParseFiniteFloat(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_yaw_rad") {
        state.camera.yaw = ParseFiniteFloat(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_pitch_rad") {
        state.camera.pitch = std::clamp(ParseFiniteFloat(value), -1.5707f, 1.5707f);
        state.camera.has_saved_view = true;
      } else if (key == "camera_distance") {
        state.camera.distance = lidar_viewer::TopViewHalfHeight(ParseFiniteFloat(value));
        state.camera.has_saved_view = true;
      } else if (key == "camera_3d_target_x") {
        state.camera.saved_3d_view.target.x = ParseFiniteFloat(value);
      } else if (key == "camera_3d_target_y") {
        state.camera.saved_3d_view.target.y = ParseFiniteFloat(value);
      } else if (key == "camera_3d_target_z") {
        state.camera.saved_3d_view.target.z = ParseFiniteFloat(value);
      } else if (key == "camera_3d_yaw_rad") {
        state.camera.saved_3d_view.yaw = ParseFiniteFloat(value);
      } else if (key == "camera_3d_pitch_rad") {
        state.camera.saved_3d_view.pitch = std::clamp(ParseFiniteFloat(value), -1.5707f, 1.5707f);
      } else if (key == "camera_3d_distance") {
        state.camera.saved_3d_view.distance = std::clamp(ParseFiniteFloat(value), 1.0f, 5000.0f);
      } else if (key == "camera_3d_valid") {
        state.camera.saved_3d_view.valid = (ParseStrictInt(value) != 0);
      } else if (key == "camera_2d_target_x") {
        state.camera.saved_2d_view.target.x = ParseFiniteFloat(value);
      } else if (key == "camera_2d_target_y") {
        state.camera.saved_2d_view.target.y = ParseFiniteFloat(value);
      } else if (key == "camera_2d_target_z") {
        state.camera.saved_2d_view.target.z = ParseFiniteFloat(value);
      } else if (key == "camera_2d_yaw_rad") {
        state.camera.saved_2d_view.yaw = ParseFiniteFloat(value);
      } else if (key == "camera_2d_pitch_rad") {
        state.camera.saved_2d_view.pitch = std::clamp(ParseFiniteFloat(value), -1.5707f, 1.5707f);
      } else if (key == "camera_2d_distance") {
        state.camera.saved_2d_view.distance = lidar_viewer::TopViewHalfHeight(ParseFiniteFloat(value));
      } else if (key == "camera_2d_valid") {
        state.camera.saved_2d_view.valid = (ParseStrictInt(value) != 0);
      } else if (key == "ground_removal_enabled") {
        state.ground_removal_enabled = (ParseStrictInt(value) != 0);
      } else if (key=="ground_view") {
        if (value=="all") state.ground_view=ViewerState::GroundView::All;
        else if (value=="ground") state.ground_view=ViewerState::GroundView::Ground;
        else if (value=="obstacles") state.ground_view=ViewerState::GroundView::Obstacles;
        else throw std::invalid_argument("Unknown ground display mode");
      } else if (key == "clustering_enabled") {
        state.algorithm_config.clustering.enabled = (ParseStrictInt(value) != 0);
      } else if (key == "clustering_mode") {
        if (value!="euclidean" && value!="euclidean_3d") throw std::invalid_argument("Unknown cluster mode");
        state.algorithm_config.clustering.mode=value;
      } else if (key == "clustering_tolerance_m") {
        state.algorithm_config.clustering.tolerance_m=std::clamp(ParseFiniteFloat(value),.1f,1.5f);
      } else if (key == "clustering_vertical_gap_m") {
        state.algorithm_config.clustering.max_vertical_gap_m=std::clamp(ParseFiniteFloat(value),.1f,2.5f);
      } else if (key == "detection_enabled") {
        state.algorithm_config.detection.enabled = (ParseStrictInt(value) != 0);
      } else if (key == "detection_model") {
        if (value!="tiny" && value!="short_range") throw std::invalid_argument("Unknown detector model");
        state.algorithm_config.detection.model=value;state.detector_model_loaded=true;
      } else if (key == "detection_min_score") {
        state.algorithm_config.detection.min_confidence=std::clamp(ParseFiniteFloat(value),.1f,.9f);
      } else if (key == "detection_classes") {
        const int mask=ParseStrictInt(value);
        if (mask<0 || mask>31) throw std::invalid_argument("Invalid detection classes");
        state.algorithm_config.detection.classes=static_cast<std::uint32_t>(mask);
      } else if (key == "object_method") {
        const int method=ParseStrictInt(value);
        if (method<0 || method>3) throw std::invalid_argument("Unknown object method");
        state.object_method=method;
      } else if (key == "calibration_yaw_deg") {
        state.calibration.yaw_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      } else if (key == "calibration_pitch_deg") {
        state.calibration.pitch_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      } else if (key == "calibration_roll_deg") {
        state.calibration.roll_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      } else if (key == "calibration_z_m") {
        state.calibration.z_m = std::clamp(ParseFiniteFloat(value), -5.0f, 5.0f);
      } else if (key == "imu_level_enabled") {
        state.imu_level_enabled = ParseStrictInt(value) != 0;
      } else if (key=="tracking_enabled") {
        state.tracking_params.enabled=ParseStrictInt(value)!=0;
      } else if (key=="tracking_max_missed_s") {
        state.tracking_params.max_missed_s=std::clamp(ParseFiniteFloat(value),.1f,2.f);
      } else if (key=="tracking_confirmation_hits") {
        state.tracking_params.confirmation_hits=std::clamp(ParseStrictInt(value),1,10);
      } else if (key=="tracking_show_trails") {
        state.tracking_params.show_trails=ParseStrictInt(value)!=0;
      } else if (key=="tracking_show_speed") {
        state.tracking_params.show_speed=ParseStrictInt(value)!=0;
      } else if (key=="imu_mount_auto_axes") {
        state.imu_mount.auto_axes=ParseStrictInt(value)!=0;
      } else if (key == "imu_mount_confirmed") {
        state.imu_mount.confirmed = ParseStrictInt(value) != 0;
      } else if (key == "imu_mount_imu_frame") {
        state.imu_mount.imu_frame = value;
      } else if (key == "imu_mount_lidar_frame") {
        state.imu_mount.lidar_frame = value;
      } else if (key == "imu_mount_roll_deg") {
        state.imu_mount.roll_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      } else if (key == "imu_mount_pitch_deg") {
        state.imu_mount.pitch_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      } else if (key == "imu_mount_yaw_deg") {
        state.imu_mount.yaw_deg = std::clamp(ParseFiniteFloat(value), -180.0f, 180.0f);
      }
    } catch (const std::exception & error) {
      std::cerr << "Ignoring invalid setting '" << key << "' at " << path << ':'
                << line_number << ": " << error.what() << '\n';
    }
  }

  if (height_max - height_min > 1e-3f) {
    state.height_color_min = height_min;
    state.height_color_max = height_max;
  } else {
    std::cerr << "Ignoring invalid height color range in " << path << '\n';
  }
  if (in.bad()) {
    std::cerr << "Failed to completely read settings file " << path << '\n';
  }

  if (!in.bad()) {
    try { state.roi = lidar_viewer::ParseRoiSettings(settings_text, state.roi); }
    catch (const std::exception & error) {
      std::cerr << "Ignoring invalid ROI settings in " << path << ": " << error.what() << '\n';
    }
    try {state.display_roi=lidar_viewer::ParseRoiSettings(settings_text,state.display_roi,"display_roi_");}
    catch (const std::exception & error) {
      std::cerr << "Ignoring invalid display ROI settings in " << path << ": " << error.what() << '\n';
    }
  }
  MarkGridDirty(state);
}

void LoadSettings(ViewerState & state)
{
  const auto app_scale=state.ui_scale;
  state.display_roi={};state.selected_display_roi=0;state.display_roi_pending=false;
  state.detector_model_loaded=false;
  state.ground_view=ViewerState::GroundView::All;
#ifdef LIDAR_VIEWER_USER_SETTINGS
  LoadSettingsFile(state,lidar_viewer::ProjectResourcePath(kProfilesDir)/(state.dataset_profile+".cfg"));
#endif
  LoadSettingsFile(state, ProfileSettingsPath(state.dataset_profile));
  if (state.dataset_profile == "default") {
    LoadSettingsFile(state, SettingsPath());
  }
  state.camera.distance=std::clamp(state.camera.distance,
    state.camera.projection_mode==CameraState::ProjectionMode::View2D ? lidar_viewer::kMinTopViewHalfHeight : 1.f,5000.f);
  // Old tiny profiles stored uncalibrated scores without a model identity.
  // Preserve explicitly saved model thresholds; migrate legacy defaults only.
  if (!state.detector_model_loaded && state.algorithm_config.detection.model=="short_range")
    state.algorithm_config.detection.min_confidence=std::max(.5f,state.algorithm_config.detection.min_confidence);
  if (state.object_method==0 || state.object_method==3)
    state.object_method=state.algorithm_config.detection.model=="short_range" ? 3 : 0;
  state.roi=lidar_viewer::GroundExclusionRoi(std::move(state.roi));
  for (auto & box:state.display_roi.boxes) box.mode=lidar_viewer::RoiBoxMode::Include;
  state.saved_display_roi=state.display_roi;state.display_roi_history.Clear();
  state.roi_purpose=state.display_roi.enabled ? ViewerState::RoiPurpose::Display : ViewerState::RoiPurpose::Ground;
  state.saved_roi=state.roi; state.roi_history.Clear();
  state.analysis_roi=state.roi;
  state.applied_exclusions=state.roi;
  if (state.app_settings_loaded) state.ui_scale=app_scale;
  if (state.detector_cli_override) state.algorithm_config.detection.enabled=true;
  if (state.ground_view!=ViewerState::GroundView::All) state.ground_removal_enabled=true;
}


void LoadAppSettings(ViewerState & state)
{
  try {
    std::ifstream file(lidar_viewer::UserConfigurationDirectory()/"app_settings.cfg");
    std::string line;
    while (std::getline(file,line)) {
      if (line.rfind("ui_scale=",0)==0)
        state.ui_scale=std::clamp(lidar_viewer::settings::ParseFiniteFloat(line.substr(9)),.75f,1.5f);
      else if (line.rfind("inspector_width=",0)==0)
        state.inspector_width=std::clamp(lidar_viewer::settings::ParseFiniteFloat(line.substr(16)),280.f,520.f);
    }
  } catch (const std::exception & error) {state.app_settings_error=error.what();}
  state.app_settings_loaded=true;
}

void SaveAppSettings(ViewerState & state)
{
  try {
    const auto folder=lidar_viewer::UserConfigurationDirectory();fs::create_directories(folder);
    lidar_viewer::settings::AtomicWriteText(folder/"app_settings.cfg","ui_scale="+std::to_string(state.ui_scale)+
      "\ninspector_width="+std::to_string(state.inspector_width)+"\n");
    state.app_settings_dirty=false;state.app_settings_error.clear();
  } catch (const std::exception & error) {state.app_settings_error=error.what();}
}

void AdvanceFrame(ViewerState & state, int direction)
{
  if (state.live_mode || state.frames.empty() || state.loading_bag_source) {
    return;
  }
  const auto index=state.requested_frame.value_or(state.geometry_index.value_or(state.current_index));
  if (direction > 0) {
    if (index + 1 >= state.frames.size() && !state.loop_playback) {
      state.is_playing = false;
      return;
    }
    TryLoadFrame(state, (index + 1) % state.frames.size());
  } else {
    if (index == 0 && !state.loop_playback) {
      return;
    }
    TryLoadFrame(state, index == 0 ? state.frames.size() - 1 : index - 1);
  }
}

void TogglePlayback(ViewerState & state)
{
  if (state.roi_gesture.active) {
    if (state.live_mode) state.roi_gesture.resume_live=!state.roi_gesture.resume_live;
    else state.roi_gesture.resume_playback=!state.roi_gesture.resume_playback;
    return;
  }
  if (state.live_mode) {SetLivePaused(state,!state.live_paused);return;}
  if (state.frames.empty() || state.loading_bag_source) return;
  state.is_playing=!state.is_playing;
  if (!state.is_playing) state.requested_frame.reset();
  state.last_frame_advance_time=glfwGetTime();
}

void HandlePlaybackShortcuts(ViewerState & state)
{
  // Window focus alone must not swallow transport keys. ImGui routing owns
  // them so a focused button/checkbox cannot also activate on the same Space.
  // Active text editors and popup navigation keep their normal keyboard input.
  if (state.loading_bag_source || state.bag_picker.Active() || ImGui::GetIO().WantTextInput ||
      ImGui::IsPopupOpen(nullptr,ImGuiPopupFlags_AnyPopup)) return;
  constexpr auto flags=ImGuiInputFlags_RouteGlobal|ImGuiInputFlags_RouteOverFocused;
  if (ImGui::Shortcut(ImGuiKey_Space,flags)) TogglePlayback(state);
  for (const auto & step:std::array<std::pair<ImGuiKey,int>,4>{{
      {ImGuiKey_RightArrow,1},{ImGuiKey_N,1},{ImGuiKey_LeftArrow,-1},{ImGuiKey_P,-1}}})
    if (ImGui::Shortcut(step.first,flags|ImGuiInputFlags_Repeat)) AdvanceFrame(state,step.second);
}

bool PointerInScene(GLFWwindow *, const ViewerState & state)
{
  // Use the position from the most recent motion event. A native pointer query
  // can already be ahead of a queued mouse-down when processing a busy frame.
  const double x=state.camera.last_x, y=state.camera.last_y;
  const auto & rect = state.ui_layout.viewport;
  return x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h;
}

void FreezeForRoi(ViewerState & state)
{
  state.is_playing=false;state.requested_frame.reset();SetLivePaused(state,true);
}

void FinishRoiGesture(ViewerState & state, bool cancel);
void SelectRoiTool(ViewerState & state, ViewerState::RoiTool tool)
{
  if (state.roi_gesture.active) FinishRoiGesture(state,true);
  state.roi_tool=tool;
  if (tool!=ViewerState::RoiTool::Navigate) {
    SetProjectionMode(state,CameraState::ProjectionMode::View2D);
    state.roi_tab_requested=true;
    if (tool==ViewerState::RoiTool::Draw && !EditedRoi(state).boxes.empty()) {
      const auto & selected=EditedRoi(state).boxes[SelectedRoi(state)];
      state.roi_new_z=selected.center.z;state.roi_new_height=selected.size.z;
    }
  }
}

void FinishRoiGesture(ViewerState & state, bool cancel)
{
  if (!state.roi_gesture.active) return;
  if (state.roi_tool==ViewerState::RoiTool::Draw && !EditedRoi(state).boxes.empty()) {
    const auto & box=EditedRoi(state).boxes[SelectedRoi(state)];
    if (box.size.x<.1f || box.size.y<.1f) cancel=true;
  }
  if (cancel) EditedRoi(state)=state.roi_gesture.before;
  else if (!lidar_viewer::SameRoi(state.roi_gesture.before,EditedRoi(state))) EditedRoiHistory(state).Record(state.roi_gesture.before);
  state.roi_gesture.active=false; MarkRoiPending(state); state.roi_reveal_selection=true;
  state.is_playing=state.roi_gesture.resume_playback;
  if (state.roi_gesture.resume_live) SetLivePaused(state,false);
  if (SelectedRoi(state)>=EditedRoi(state).boxes.size()) SelectedRoi(state)=EditedRoi(state).boxes.empty() ? 0 : EditedRoi(state).boxes.size()-1;
  RebuildRoiGeometry(state);
  if (state.roi_tool==ViewerState::RoiTool::Draw) state.roi_tool=ViewerState::RoiTool::Move;
}

std::optional<Vec3> RoiPointerPoint(const ViewerState & state, double x, double y, float z)
{
  const auto & view=state.ui_layout.viewport;
  if (view.w<1 || view.h<1) return {};
  return lidar_viewer::RoiPointFromTopView(static_cast<float>(2*(x-view.x)/view.w-1),
    static_cast<float>(1-2*(y-view.y)/view.h),view.w/view.h,lidar_viewer::TopViewHalfHeight(state.camera.distance),
    state.camera.target,state.camera.yaw,state.roi_frame,z);
}

void DuplicateSelectedRoi(ViewerState & state)
{
  if (EditedRoi(state).boxes.empty() || EditedRoi(state).boxes.size()>=lidar_viewer::kMaxRoiBoxes) return;
  auto copy=lidar_viewer::DuplicateRoi(EditedRoi(state).boxes[SelectedRoi(state)],EditedRoi(state));
  EditedRoiHistory(state).Record(EditedRoi(state)); EditedRoi(state).boxes.push_back(std::move(copy));
  SelectedRoi(state)=EditedRoi(state).boxes.size()-1; MarkRoiPending(state); state.roi_reveal_selection=true;
}

void DeleteSelectedRoi(ViewerState & state)
{
  auto & boxes=EditedRoi(state).boxes;
  if (SelectedRoi(state)>=boxes.size()) return;
  EditedRoiHistory(state).Record(EditedRoi(state));
  boxes.erase(boxes.begin()+static_cast<std::ptrdiff_t>(SelectedRoi(state)));
  // Keep the next row selected; only step back when deleting the last row.
  SelectedRoi(state)=boxes.empty() ? 0 : std::min(SelectedRoi(state),boxes.size()-1);
  state.roi_numeric_edit=false;state.roi_reveal_selection=true;
  MarkRoiPending(state);
}

void RestoreRoiHistory(ViewerState & state, bool redo)
{
  if (state.roi_gesture.active) {FinishRoiGesture(state,true); return;}
  if (redo ? EditedRoiHistory(state).Redo(EditedRoi(state)) : EditedRoiHistory(state).Undo(EditedRoi(state))) {
    SelectedRoi(state)=EditedRoi(state).boxes.empty() ? 0 : std::min(SelectedRoi(state),EditedRoi(state).boxes.size()-1);
    MarkRoiPending(state); state.roi_numeric_edit=false; state.roi_reveal_selection=true;
    RebuildRoiGeometry(state);
  }
}

struct RoiHandlePoint {ImVec2 screen; int x, y; bool rotate{false};};

ImVec2 RoiScreenPoint(const ViewerState & state, Vec3 raw)
{
  const auto p=state.roi_frame.Transform(raw)-state.camera.target;
  const auto & v=state.ui_layout.viewport;
  const float pixels=v.h/(2*lidar_viewer::TopViewHalfHeight(state.camera.distance));
  const float c=std::cos(state.camera.yaw),s=std::sin(state.camera.yaw);
  return {v.x+v.w*.5f+(p.x*c+p.y*s)*pixels,v.y+v.h*.5f-(-p.x*s+p.y*c)*pixels};
}

std::vector<RoiHandlePoint> RoiHandlePoints(const ViewerState & state)
{
  if (state.roi_tool!=ViewerState::RoiTool::Move || EditedRoi(state).boxes.empty() ||
      state.camera.projection_mode!=CameraState::ProjectionMode::View2D) return {};
  const auto & box=EditedRoi(state).boxes[SelectedRoi(state)];
  if(!box.enabled || !EditedRoi(state).show_boxes) return {};
  const float a=box.yaw_deg*kPi/180.f,c=std::cos(a),s=std::sin(a);
  std::vector<RoiHandlePoint> result;
  for (const auto axes : {std::array<int,2>{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}}) {
    const float x=axes[0]*box.size.x*.5f,y=axes[1]*box.size.y*.5f;
    result.push_back({RoiScreenPoint(state,box.center+Vec3{x*c-y*s,x*s+y*c,0}),axes[0],axes[1]});
  }
  const auto center=RoiScreenPoint(state,box.center),edge=result[5].screen;
  const float dx=edge.x-center.x,dy=edge.y-center.y,length=std::max(1.f,std::hypot(dx,dy));
  result.push_back({{edge.x+dx/length*30*state.ui_layout.scale,edge.y+dy/length*30*state.ui_layout.scale},0,0,true});
  return result;
}

void MouseButtonCallback(GLFWwindow * window, int button, int action, int)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) return;
  if (action == GLFW_RELEASE) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      state->camera.dragging_left=false;
      if (state->roi_gesture.active) FinishRoiGesture(*state,false);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) state->camera.dragging_right = false;
    return;
  }
  if (!PointerInScene(window, *state) ||
      (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)) return;
  if (button==GLFW_MOUSE_BUTTON_LEFT && state->roi_tool!=ViewerState::RoiTool::Navigate) {
    if (state->camera.projection_mode!=CameraState::ProjectionMode::View2D) return;
    const double x=state->camera.last_x,y=state->camera.last_y;
    auto & gesture=state->roi_gesture;
    gesture.before=EditedRoi(*state);
    gesture.kind=ViewerState::RoiGesture::Kind::Move;
    if (state->roi_tool==ViewerState::RoiTool::Draw) {
      if (EditedRoi(*state).boxes.size()>=lidar_viewer::kMaxRoiBoxes) return;
      const auto point=RoiPointerPoint(*state,x,y,state->roi_new_z);
      if (!point) {state->ui_error="Cannot draw on the LiDAR XY plane at this tilt."; return;}
      gesture.start=*point;
      const auto right_point=RoiPointerPoint(*state,x+10,y,state->roi_new_z);
      if (!right_point) return;
      const auto right=*right_point-*point;
      const float yaw=std::atan2(right.y,right.x)*180.f/kPi;
      const auto mode=state->roi_purpose==ViewerState::RoiPurpose::Display ?
        lidar_viewer::RoiBoxMode::Include : lidar_viewer::RoiBoxMode::Exclude;
      auto box=lidar_viewer::RoiFromDrag(*point,*point,state->roi_new_z,state->roi_new_height,
        lidar_viewer::NextRoiName(EditedRoi(*state),mode),yaw,mode);
      EditedRoi(*state).boxes.push_back(box); SelectedRoi(*state)=EditedRoi(*state).boxes.size()-1;
      EditedRoi(*state).enabled=true; EditedRoi(*state).show_boxes=true; gesture.original=box;
    } else {
      bool found=false;
      for (const auto & handle : RoiHandlePoints(*state)) {
        if (std::hypot(handle.screen.x-x,handle.screen.y-y)>11*state->ui_layout.scale) continue;
        const auto & box=EditedRoi(*state).boxes[SelectedRoi(*state)];
        const auto point=RoiPointerPoint(*state,x,y,box.center.z);
        if (!point) return;
        gesture.original=box;gesture.start=*point;
        gesture.kind=handle.rotate ? ViewerState::RoiGesture::Kind::Rotate : ViewerState::RoiGesture::Kind::Resize;
        gesture.handle_x=handle.x;gesture.handle_y=handle.y;found=true;break;
      }
      for (std::size_t n=0;n<EditedRoi(*state).boxes.size();++n) {
        if (found) break;
        const auto i=n==0 ? SelectedRoi(*state) : (n<=SelectedRoi(*state) ? n-1 : n);
        const auto & box=EditedRoi(*state).boxes[i];
        if(!box.enabled || !EditedRoi(*state).show_boxes) continue;
        const auto point=RoiPointerPoint(*state,x,y,box.center.z);
        auto hit_box=box; hit_box.enabled=true;hit_box.mode=lidar_viewer::RoiBoxMode::Include;
        if (point && lidar_viewer::PreparedRoi({true,true,{hit_box}}).Contains(*point)) {
          SelectedRoi(*state)=i; gesture.start=*point; gesture.original=box; found=true; break;
        }
      }
      if (!found) {state->camera.dragging_left=true;return;}
    }
    gesture.resume_playback=state->is_playing;
    gesture.resume_live=state->live_mode && !state->live_paused;
    gesture.active=true; FreezeForRoi(*state); RebuildRoiGeometry(*state);
    return;
  }
  if (button == GLFW_MOUSE_BUTTON_LEFT) state->camera.dragging_left = true;
  if (button == GLFW_MOUSE_BUTTON_RIGHT) state->camera.dragging_right = true;
}

void CursorCallback(GLFWwindow * window, double x, double y)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) return;
  const double dx = x - state->camera.last_x, dy = y - state->camera.last_y;
  state->camera.last_x = x;
  state->camera.last_y = y;
  if (state->roi_gesture.active) {
    const auto point=RoiPointerPoint(*state,x,y,state->roi_gesture.original.center.z);
    if (point) {
      auto box=EditedRoi(*state).boxes[SelectedRoi(*state)];
      try {
        if (state->roi_tool==ViewerState::RoiTool::Draw)
          box=lidar_viewer::RoiFromDrag(state->roi_gesture.start,*point,box.center.z,box.size.z,box.name,
            state->roi_gesture.original.yaw_deg,box.mode);
        else if (state->roi_gesture.kind==ViewerState::RoiGesture::Kind::Resize) {
          const auto & original=state->roi_gesture.original;
          const float angle=original.yaw_deg*kPi/180.f,c=std::cos(angle),s=std::sin(angle);
          const float hx=state->roi_gesture.handle_x*original.size.x*.5f,hy=state->roi_gesture.handle_y*original.size.y*.5f;
          const auto handle=original.center+Vec3{hx*c-hy*s,hx*s+hy*c,0};
          box=lidar_viewer::ResizeRoi(original,handle+(*point-state->roi_gesture.start),state->roi_gesture.handle_x,state->roi_gesture.handle_y);
        }
        else if (state->roi_gesture.kind==ViewerState::RoiGesture::Kind::Rotate)
          box=lidar_viewer::RotateRoi(state->roi_gesture.original,state->roi_gesture.start,*point,
            glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS || glfwGetKey(window,GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);
        else box.center=state->roi_gesture.original.center+(*point-state->roi_gesture.start);
        lidar_viewer::ValidateRoiBox(box);
        EditedRoi(*state).boxes[SelectedRoi(*state)]=box; RebuildRoiGeometry(*state);
      } catch (const std::exception & error) {state->ui_error=error.what();}
    }
    return;
  }
  if (!PointerInScene(window, *state) ||
      (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)) return;
  const float scale = state->camera.projection_mode==CameraState::ProjectionMode::View2D ?
    2*lidar_viewer::TopViewHalfHeight(state->camera.distance)/std::max(state->ui_layout.viewport.h,1.f) :
    std::max(kPanBaseScale * state->camera.distance,0.01f);
  if (state->camera.dragging_left) {
    if (state->camera.projection_mode == CameraState::ProjectionMode::View2D) {
      state->camera.target = lidar_viewer::PanTopView(
        state->camera.target, state->camera.yaw, static_cast<float>(dx), static_cast<float>(dy), scale);
    } else {
      state->camera.yaw += static_cast<float>(dx) * kOrbitSensitivity;
      state->camera.pitch = std::clamp(state->camera.pitch + static_cast<float>(dy) * kOrbitSensitivity, -1.5f, 1.5f);
    }
  }
  if (state->camera.dragging_right) {
    if (state->camera.projection_mode == CameraState::ProjectionMode::View2D) {
      state->camera.yaw = lidar_viewer::RotateTopView(state->camera.yaw, static_cast<float>(dx) * kOrbitSensitivity);
    } else {
      const Vec3 forward = Normalize(state->camera.target - CameraPosition(state->camera));
      const Vec3 right = Normalize(Cross(forward, {0., 0., 1.}));
      const Vec3 up = Normalize(Cross(right, forward));
      state->camera.target = state->camera.target - right * static_cast<float>(dx) * scale +
        up * static_cast<float>(dy) * scale;
    }
  }
}

void ScrollCallback(GLFWwindow * window, double, double yoffset)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state || !PointerInScene(window, *state) ||
      (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)) return;
  const auto distance=state->camera.distance*static_cast<float>(std::pow(0.9,yoffset));
  if (state->camera.projection_mode==CameraState::ProjectionMode::View2D) {
    const auto & view=state->ui_layout.viewport;
    if (view.w<1 || view.h<1 || state->roi_gesture.active) return;
    const float before=lidar_viewer::TopViewHalfHeight(state->camera.distance);
    state->camera.distance=lidar_viewer::TopViewHalfHeight(distance);
    // Keep the point under the cursor stationary while magnifying the view.
    const float nx=static_cast<float>(2*(state->camera.last_x-view.x)/view.w-1);
    const float ny=static_cast<float>(1-2*(state->camera.last_y-view.y)/view.h);
    state->camera.target=lidar_viewer::PanTopView(state->camera.target,state->camera.yaw,
      -nx*view.w/view.h,ny,before-state->camera.distance);
  } else state->camera.distance=std::clamp(distance,1.f,5000.f);
}

void KeyCallback(GLFWwindow * window, int key, int, int action, int mods)
{
  if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) return;
  if (key == GLFW_KEY_O && (mods & GLFW_MOD_CONTROL) && action == GLFW_PRESS) {
    StartBagPicker(*state);
    return;
  }
  if ((mods&GLFW_MOD_CONTROL) && key==GLFW_KEY_S && action==GLFW_PRESS) {
    state->save_requested=true; return;
  }
  const bool typing=ImGui::GetCurrentContext() && ImGui::GetIO().WantTextInput;
  const bool popup=ImGui::GetCurrentContext() && ImGui::IsPopupOpen(nullptr,ImGuiPopupFlags_AnyPopup);
  if (!typing && !popup && action==GLFW_PRESS) {
    if (key==GLFW_KEY_ESCAPE && state->open_app_settings) {
      state->open_app_settings=false;if (state->app_settings_dirty) SaveAppSettings(*state);return;
    }
    if (key==GLFW_KEY_ESCAPE && state->input_load) {state->input_load->Cancel();return;}
    if (state->loading_bag_source) return;
    if (key==GLFW_KEY_ESCAPE && (state->roi_gesture.active || state->roi_tool!=ViewerState::RoiTool::Navigate)) {
      FinishRoiGesture(*state,true); state->roi_tool=ViewerState::RoiTool::Navigate; return;
    }
    if (key==GLFW_KEY_R && !(mods&GLFW_MOD_CONTROL)) {SelectRoiTool(*state,ViewerState::RoiTool::Draw); return;}
    if ((mods&GLFW_MOD_CONTROL) && key==GLFW_KEY_Z) {RestoreRoiHistory(*state,(mods&GLFW_MOD_SHIFT)!=0); return;}
    if ((mods&GLFW_MOD_CONTROL) && key==GLFW_KEY_Y) {RestoreRoiHistory(*state,true); return;}
    if ((mods&GLFW_MOD_CONTROL) && key==GLFW_KEY_D && state->roi_tab_active) {
      try {DuplicateSelectedRoi(*state);} catch (const std::exception & error) {state->ui_error=error.what();} return;
    }
    if (key==GLFW_KEY_DELETE && state->roi_tab_active) {DeleteSelectedRoi(*state); return;}
  }
  if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard) return;
  switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    case GLFW_KEY_F: state->fit_view_requested = true; break;
    case GLFW_KEY_C: state->color_mode = (state->color_mode + 1) % 3; break;
    case GLFW_KEY_Q:
    case GLFW_KEY_E:
      if (state->camera.projection_mode == CameraState::ProjectionMode::View2D)
        state->camera.yaw = lidar_viewer::RotateTopView(state->camera.yaw, (key == GLFW_KEY_Q ? -1.f : 1.f) * kPi / 36);
      break;
    case GLFW_KEY_LEFT_BRACKET: state->camera.point_size = std::max(1.f, state->camera.point_size - 1.f); break;
    case GLFW_KEY_RIGHT_BRACKET: state->camera.point_size = std::min(8.f, state->camera.point_size + 1.f); break;
    default: break;
  }
}

void ConfigureInterface(GLFWwindow * window, ViewerState & state)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto & io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  float sx = 1.f, sy = 1.f;
  glfwGetWindowContentScale(window, &sx, &sy);
  state.font_raster_scale = std::clamp(std::max(sx, sy), 1.f, 3.f);
  bool font_loaded=false;
  for (const auto * font : {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf",
      "/System/Library/Fonts/Supplemental/Arial.ttf"}) {
    if (fs::is_regular_file(font)) {font_loaded=io.Fonts->AddFontFromFileTTF(font,17.f*state.font_raster_scale)!=nullptr; if(font_loaded) break;}
  }
  if (!font_loaded) io.Fonts->AddFontDefault();
  const char * cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
  if (fs::exists(cjk)) {
    ImFontConfig merged;
    merged.MergeMode = true;
    io.Fonts->AddFontFromFileTTF(cjk, 17.f * state.font_raster_scale, &merged, io.Fonts->GetGlyphRangesKorean());
  }
  auto & style = ImGui::GetStyle();
  ImGui::StyleColorsDark(&style);
  style.WindowPadding = {16, 16};
  style.FramePadding = {10, 7};
  style.ItemSpacing = {10, 10};
  style.ItemInnerSpacing = {8, 6};
  style.WindowRounding = 10;
  style.FrameRounding = 6;
  style.GrabRounding = 5;
  style.PopupRounding = 8;
  style.TabRounding = 6;
  style.WindowBorderSize = 1;
  style.FrameBorderSize = 0;
  style.ScrollbarSize = 10;
  style.Colors[ImGuiCol_Text] = {0.90f, 0.93f, 0.97f, 1};
  style.Colors[ImGuiCol_TextDisabled] = {0.58f, 0.65f, 0.74f, 1};
  style.Colors[ImGuiCol_WindowBg] = {0.075f, 0.092f, 0.12f, 1};
  style.Colors[ImGuiCol_PopupBg] = {0.09f, 0.11f, 0.145f, 1};
  style.Colors[ImGuiCol_Border] = {0.16f, 0.20f, 0.26f, 1};
  style.Colors[ImGuiCol_FrameBg] = {0.12f, 0.15f, 0.195f, 1};
  style.Colors[ImGuiCol_FrameBgHovered] = {0.17f, 0.22f, 0.285f, 1};
  style.Colors[ImGuiCol_FrameBgActive] = {0.19f, 0.27f, 0.35f, 1};
  style.Colors[ImGuiCol_Button] = {0.13f, 0.18f, 0.24f, 1};
  style.Colors[ImGuiCol_ButtonHovered] = {0.19f, 0.29f, 0.39f, 1};
  style.Colors[ImGuiCol_ButtonActive] = {0.23f, 0.37f, 0.49f, 1};
  style.Colors[ImGuiCol_CheckMark] = {0.31f, 0.77f, 0.91f, 1};
  style.Colors[ImGuiCol_SliderGrab] = {0.31f, 0.77f, 0.91f, 1};
  style.Colors[ImGuiCol_SliderGrabActive] = {0.49f, 0.88f, 1.f, 1};
  style.Colors[ImGuiCol_Header] = {0.12f, 0.16f, 0.21f, 1};
  style.Colors[ImGuiCol_HeaderHovered] = {0.18f, 0.35f, 0.46f, 1};
  style.Colors[ImGuiCol_Tab] = {0.10f, 0.135f, 0.18f, 1};
  style.Colors[ImGuiCol_TabHovered] = {0.19f, 0.31f, 0.41f, 1};
  style.Colors[ImGuiCol_TabSelected] = {0.16f, 0.29f, 0.39f, 1};
  if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init("#version 330")) {
    throw std::runtime_error("Failed to initialize the interface");
  }
}

void BeginPanel(const char * name, const lidar_viewer::UiRect & rect, bool scroll = true)
{
  ImGui::SetNextWindowPos({rect.x, rect.y});
  ImGui::SetNextWindowSize({rect.w, rect.h});
  ImGui::Begin(name, nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
    (scroll ? 0 : ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse));
}

void SectionTitle(const char * title)
{
  ImGui::Spacing();
  ImGui::TextColored({0.62f, 0.73f, 0.85f, 1.f}, "%s", title);
  ImGui::Separator();
}

void Metric(const char * label, const std::string & value)
{
  if (ImGui::BeginTable(label, 2, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, .52f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, .48f);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0); ImGui::TextDisabled("%s", label);
    ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(value.c_str());
    ImGui::EndTable();
  }
}

void Tip(const char * value)
{
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort|ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(std::min(22*ImGui::GetFontSize(),ImGui::GetIO().DisplaySize.x-32));
    ImGui::TextUnformatted(value);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

std::string BagLabel(const ViewerState & state,const BagSource & source)
{
  return (source.recording_date.empty() ? "Unknown date" :
    source.recording_date + "  " + source.recording_time) + "  |  " + SourceDuration(state,source) + "  |  " + source.display_name;
}

void BuildSourceSelector(ViewerState & state, float width)
{
  const auto * group = state.file_groups.empty() ? nullptr : &state.file_groups[state.current_file_group];
  const bool dataset=!state.live_mode && !state.frames.empty() && state.frames.front().dataset_sample!=nullptr;
  const auto preview = state.live_mode ? std::string("LIVE | ") + state.live_points_topic.data() :
    dataset ? std::string("Dataset | ")+std::to_string(state.frames.size())+" samples | "+
      (state.current_bag_source ? state.bag_sources[*state.current_bag_source].path.filename().u8string() : (group ? group->display_name : "")) :
    state.current_bag_source ? BagLabel(state,state.bag_sources[*state.current_bag_source]) :
    (group ? group->display_name : "No input loaded");
  ImGui::SetNextItemWidth(width);
  const float popup_width = std::min(720.f * state.ui_layout.scale, ImGui::GetIO().DisplaySize.x - 32.f);
  ImGui::SetNextWindowSizeConstraints({popup_width, 0}, {popup_width, 600.f * state.ui_layout.scale});
  ImGui::BeginDisabled(state.loading_bag_source.has_value() || state.folder_scan_future.valid() ||
    state.bag_picker.Active() || state.live.Active() || (!group && state.bag_sources.empty() && state.hidden_sources.empty()));
  std::optional<fs::path> remove_source;
  if (ImGui::BeginCombo("##Source", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
    if (ImGui::IsWindowAppearing()) state.check_source_availability=true;
    if (!state.bag_sources.empty()) {
      ImGui::TextUnformatted("Data sources");
      ImGui::TextDisabled("%zu sources  |  Dates from recording names", state.bag_sources.size());
      std::vector<std::string> dates;
      for (const auto & source : state.bag_sources) {
        const auto date = source.recording_date.empty() ? "Unknown date" : source.recording_date;
        if (std::find(dates.begin(), dates.end(), date) == dates.end()) dates.push_back(date);
      }
      ImGui::SetNextItemWidth(175.f * state.ui_layout.scale);
      if (ImGui::BeginCombo("##BagDate", state.bag_date_filter.empty() ? "All dates" : state.bag_date_filter.c_str())) {
        if (ImGui::Selectable("All dates", state.bag_date_filter.empty())) state.bag_date_filter.clear();
        for (const auto & date : dates)
          if (ImGui::Selectable(date.c_str(), state.bag_date_filter == date)) state.bag_date_filter = date;
        ImGui::EndCombo();
      }
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::InputTextWithHint("##BagSearch", "Find bag name...", state.bag_search.data(), state.bag_search.size());
      ImGui::Separator();
      std::string needle = state.bag_search.data();
      auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
      std::transform(needle.begin(), needle.end(), needle.begin(), lower);
      std::size_t shown = 0;
      if (ImGui::BeginTable("##BagRecordings",4,ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Recorded",ImGuiTableColumnFlags_WidthFixed,175.f*state.ui_layout.scale);
        ImGui::TableSetupColumn("Duration",ImGuiTableColumnFlags_WidthFixed,85.f*state.ui_layout.scale);
        ImGui::TableSetupColumn("Recording",ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##Actions",ImGuiTableColumnFlags_WidthFixed,24.f*state.ui_layout.scale);
        ImGui::TableHeadersRow();
        for (std::size_t i = 0; i < state.bag_sources.size(); ++i) {
          const auto & source = state.bag_sources[i];
          const auto date = source.recording_date.empty() ? "Unknown date" : source.recording_date;
          if (!state.bag_date_filter.empty() && date != state.bag_date_filter) continue;
          std::string searchable = source.display_name;
          std::transform(searchable.begin(), searchable.end(), searchable.begin(), lower);
          if (!needle.empty() && searchable.find(needle) == std::string::npos) continue;
          ++shown;
          const bool selected = state.current_bag_source && *state.current_bag_source == i;
          const bool unavailable=state.unavailable_sources.count(source.path)!=0;
          ImGui::PushID(source.path.u8string().c_str());
          ImGui::TableNextRow();ImGui::TableSetColumnIndex(3);
          const float action_x=ImGui::GetCursorScreenPos().x;
          ImGui::TableSetColumnIndex(0);
          const float row_width=action_x-ImGui::GetCursorScreenPos().x-ImGui::GetStyle().CellPadding.x;
          const auto recorded=date+(source.recording_time.empty() ? "" : "  "+source.recording_time);
          if (unavailable) ImGui::PushStyleColor(ImGuiCol_Text,ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
          ImGui::BeginDisabled(unavailable);
          if (ImGui::Selectable(recorded.c_str(),selected,ImGuiSelectableFlags_SpanAllColumns,{row_width,0})) OpenBagSource(state,i);
          ImGui::EndDisabled();
          Tip(source.path.string().c_str());
          ImGui::TableSetColumnIndex(1);ImGui::TextUnformatted(SourceDuration(state,source).c_str());
          Tip("Whole recording duration from bag metadata. --:--:-- means timing is unavailable.");
          ImGui::TableSetColumnIndex(2);
          ImGui::TextUnformatted((unavailable ? "[Unavailable] "+source.display_name : source.display_name).c_str());
          Tip(source.path.string().c_str());
          if (unavailable) ImGui::PopStyleColor();
          ImGui::TableSetColumnIndex(3);
          const float height=ImGui::GetTextLineHeight();const float icon=std::min(height,16.f*state.ui_layout.scale);
          ImGui::PushStyleColor(ImGuiCol_Button,{0,0,0,0});
          const bool removed=ImGui::Button("##Remove",{22.f*state.ui_layout.scale,height});
          ImGui::PopStyleColor();
          const auto origin=ImGui::GetItemRectMin();
          const bool hovered=ImGui::IsItemHovered();
          if (hovered) ImGui::GetWindowDrawList()->AddRectFilled(origin,ImGui::GetItemRectMax(),IM_COL32(100,40,43,180),3);
          lidar_viewer::DrawUiIcon(ImGui::GetWindowDrawList(),lidar_viewer::UiIcon::Trash,
            {origin.x+3.f*state.ui_layout.scale,origin.y+(height-icon)*.5f},icon,
            hovered ? IM_COL32(255,150,140,255) : ImGui::GetColorU32(ImGuiCol_TextDisabled));
          Tip("Remove from list. Original files are kept. Restore removed to undo.");
          if (removed) remove_source=source.path;
          if (std::getenv("LIDAR_VIEWER_CATALOG_TRACE") && ImGui::GetFrameCount()%8==0)
            std::cout<<"Source row: "<<source.path<<" | select "<<action_x-row_width+8<<' '<<origin.y+height*.5f
              <<" | remove "<<origin.x+11.f*state.ui_layout.scale<<' '<<origin.y+height*.5f
              <<" | available "<<!unavailable<<std::endl;
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
      if (!shown) ImGui::TextDisabled("No bags match this date or name.");
    }
    if (!state.current_bag_source && !state.frames.empty() && state.frames.front().format != InputFormat::Ros2Bag) {
      if (!state.bag_sources.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled(state.frames.front().dataset_sample ? "Current dataset" : "Current BIN sequence");
      }
      for (std::size_t i = 0; i < state.file_groups.size(); ++i) {
        const bool selected = i == state.current_file_group;
        if (ImGui::Selectable(state.file_groups[i].display_name.c_str(), selected)) {
          state.is_playing = false;
          TryLoadFrame(state, state.file_groups[i].first_index);
        }
      }
    }
    ImGui::Separator();
    if (!state.hidden_sources.empty()) {
      if (ImGui::SmallButton(("Restore removed ("+std::to_string(state.hidden_sources.size())+")").c_str())) {
        try {RestoreBagSources(state);state.ui_error.clear();}
        catch (const std::exception & error) {state.ui_error=error.what();}
      }
      if (std::getenv("LIDAR_VIEWER_CATALOG_TRACE") && ImGui::GetFrameCount()%8==0) {
        const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
        std::cout<<"Restore removed UI: "<<(a.x+b.x)*.5f<<' '<<(a.y+b.y)*.5f<<std::endl;
      }
      Tip("Restore recordings removed from this list. You can also open a removed recording explicitly with Open file.");
    }
    ImGui::TextDisabled("Open file: select a recording. Add folder: register a collection.");
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (remove_source) {
    try {RemoveBagSource(state,*remove_source);state.ui_error.clear();}
    catch (const std::exception & error) {state.ui_error=std::string("Could not remove recording: ")+error.what();}
  }
  if (group) Tip(group->path.string().c_str());
}

void BuildFolderPicker(ViewerState & state)
{
  if (state.open_folder_picker) {
    state.open_folder_picker = false;
    fs::path initial = !state.frames.empty() ? state.frames.front().path.parent_path() :
      (!state.data_folders.empty() ? state.data_folders.front() : ExpandFolderPath("~"));
    BrowseFolder(state, initial);
    ImGui::OpenPopup("Add data folder");
  }
  const auto display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos({display.x * .5f, display.y * .5f}, ImGuiCond_Appearing, {.5f, .5f});
  ImGui::SetNextWindowSize({std::min(700.f * state.ui_layout.scale, display.x - 32.f),
    std::min(560.f * state.ui_layout.scale, display.y - 32.f)}, ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Add data folder", nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
  bool scanning = state.folder_scan_future.valid();
  if (scanning && state.folder_scan_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    try {
      auto found = state.folder_scan_future.get();
      if (found.sources.empty()) throw std::runtime_error("No ROS 2 bags found in this folder or its subfolders");
      auto folders = state.data_folders;
      if (std::find(folders.begin(), folders.end(), found.folder) == folders.end()) folders.push_back(found.folder);
      auto merged = FilterBagSources(MergeBagSources(state.bag_sources, found.sources),state.hidden_sources);
      std::optional<fs::path> current;
      if (state.current_bag_source) current = state.bag_sources[*state.current_bag_source].path;
      fs::create_directories(state.data_folders_config.parent_path());
      SaveBagFolders(state.data_folders_config, folders);
      const auto added = merged.size() - state.bag_sources.size();
      state.data_folders = std::move(folders);
      state.bag_sources = std::move(merged);
      if (current) {
        for (std::size_t i = 0; i < state.bag_sources.size(); ++i)
          if (state.bag_sources[i].path == *current) state.current_bag_source = i;
      }
      state.bag_date_filter.clear();
      state.bag_search.fill('\0');
      SetTransientStatus(state, std::to_string(added) + " bags added to the list");
      std::cout << "Data folder added: " << found.folder << " | " << state.bag_sources.size()
                << " bags in library" << std::endl;
      ImGui::CloseCurrentPopup();
    } catch (const std::exception & error) {
      state.folder_error = error.what();
    }
    scanning = false;
  }
  ImGui::TextWrapped("Choose a folder containing ROS 2 bags. Subfolders are included; recordings stay in place.");
  ImGui::BeginDisabled(scanning);
  ImGui::SetNextItemWidth(-85.f * state.ui_layout.scale);
  const bool entered = ImGui::InputText("##FolderPath", state.folder_path_input.data(),
    state.folder_path_input.size(), ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  if (ImGui::Button("Open") || entered) BrowseFolder(state, ExpandFolderPath(state.folder_path_input.data()));
  if (ImGui::Button("Up")) BrowseFolder(state, state.browsing_folder.parent_path());
  ImGui::SameLine();
  if (ImGui::Button("Home")) BrowseFolder(state, ExpandFolderPath("~"));
  const float reserve = 118.f * state.ui_layout.scale;
  ImGui::BeginChild("Folders", {0, std::max(40.f, ImGui::GetContentRegionAvail().y - reserve)}, true);
  fs::path navigate;
  for (const auto & folder : state.folder_children) {
    if (ImGui::Selectable((folder.filename().string() + " /").c_str(), false,
      ImGuiSelectableFlags_AllowDoubleClick) && ImGui::IsMouseDoubleClicked(0)) navigate = folder;
  }
  if (state.folder_children.empty()) ImGui::TextDisabled("No subfolders");
  ImGui::EndChild();
  if (!navigate.empty()) BrowseFolder(state, navigate);
  ImGui::TextDisabled("Double-click a folder to open it.");
  ImGui::EndDisabled();
  if (!state.folder_error.empty()) ImGui::TextColored({1.f, .56f, .48f, 1}, "%s", state.folder_error.c_str());
  if (scanning) ImGui::TextUnformatted("Searching for recordings...");
  ImGui::BeginDisabled(scanning);
  if (ImGui::Button("Add this folder")) {
    try {
      const auto folder = fs::canonical(ExpandFolderPath(state.folder_path_input.data()));
      if (!fs::is_directory(folder)) throw std::runtime_error("Choose an existing directory");
      state.folder_error.clear();
      state.folder_scan_future = std::async(std::launch::async, [folder]() {
        return ViewerState::FolderScanResult{folder, DiscoverBagSources(folder)};
      });
    } catch (const std::exception & error) { state.folder_error = error.what(); }
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  ImGui::EndDisabled();
  ImGui::EndPopup();
}

void BuildLiveDialog(ViewerState & state)
{
  const auto display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos({display.x * .5f, display.y * .5f}, ImGuiCond_Appearing, {.5f, .5f});
  ImGui::SetNextWindowSize({std::min(650.f * state.ui_layout.scale, display.x - 32.f), 0}, ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Live sensors", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) return;
  const bool available = lidar_viewer::LiveSession::Available();
  const bool connected = state.live.Active();
  ImGui::TextWrapped("Subscribe to existing ROS 2 sensor topics. Sensor drivers are started separately.");
  if (!available) ImGui::TextWrapped("This build has no live ROS 2 backend. Reconfigure after sourcing ROS 2.");
  ImGui::BeginDisabled(connected || !available);
  ImGui::TextUnformatted("LiDAR PointCloud2 topic");
  ImGui::SetNextItemWidth(-1); ImGui::InputText("##LivePointsTopic", state.live_points_topic.data(), state.live_points_topic.size());
  ImGui::TextUnformatted("IMU topic (optional)");
  ImGui::SetNextItemWidth(-1); ImGui::InputText("##LiveImuTopic", state.live_imu_topic.data(), state.live_imu_topic.size());
  ImGui::EndDisabled();
  ImGui::BeginDisabled(state.live_stats.record_busy);
  ImGui::TextUnformatted("Recording parent folder");
  ImGui::SetNextItemWidth(-1); ImGui::InputText("##LiveRecordingFolder", state.live_record_folder.data(), state.live_record_folder.size());
  ImGui::EndDisabled();
  ImGui::TextWrapped("Pause freezes the display only. Receiving and raw recording continue. Play returns to the latest frame.");
  if (state.live_mode) {
    ImGui::Text("Received: %llu clouds / %llu IMU samples", static_cast<unsigned long long>(state.live_stats.clouds),
      static_cast<unsigned long long>(state.live_stats.imu_samples));
    ImGui::Text("LiDAR: %.1f Hz / age %.0f ms", state.live_stats.cloud_hz, state.live_stats.cloud_age_ms);
    if (!state.live_stats.error.empty()) ImGui::TextWrapped("Receive error: %s", state.live_stats.error.c_str());
    if (!state.live_stats.recording_path.empty()) ImGui::TextWrapped("Bag: %s", state.live_stats.recording_path.c_str());
    ImGui::Text("Recorded: %llu messages / %.1f MiB / %llu dropped",
      static_cast<unsigned long long>(state.live_stats.recorded_messages), state.live_stats.recorded_bytes / 1048576.,
      static_cast<unsigned long long>(state.live_stats.record_dropped));
    if (!state.live_stats.record_error.empty()) ImGui::TextColored({1,.56f,.48f,1}, "%s", state.live_stats.record_error.c_str());
  }
  ImGui::Separator();
  ImGui::BeginDisabled(!available);
  if (ImGui::Button(connected ? "Disconnect" : "Connect", {130.f * state.ui_layout.scale, 0})) {
    try {
      if (connected) {
        state.live.Disconnect(); state.live_stats=state.live.Stats();
        std::cout << "Live disconnected" << std::endl;
      } else ConnectLive(state);
      state.ui_error.clear();
      ImGui::CloseCurrentPopup();
    } catch (const std::exception & error) {state.ui_error=error.what();}
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
  if (!state.ui_error.empty()) ImGui::TextWrapped("%s", state.ui_error.c_str());
  ImGui::EndPopup();
}

void FocusSelectedRoi(ViewerState & state)
{
  if (EditedRoi(state).boxes.empty()) return;
  const auto & box=EditedRoi(state).boxes[SelectedRoi(state)];
  state.camera.target=state.roi_frame.Transform(box.center);
  state.camera.distance=std::clamp(1.25f*std::max({box.size.x,box.size.y,box.size.z}),
    state.camera.projection_mode==CameraState::ProjectionMode::View2D ? lidar_viewer::kMinTopViewHalfHeight : 2.f,5000.f);
}

std::string CountText(std::uint64_t count)
{
  auto text=std::to_string(count);
  for (std::ptrdiff_t i=static_cast<std::ptrdiff_t>(text.size())-3;i>0;i-=3)
    text.insert(static_cast<std::size_t>(i),",");
  return text;
}

std::string LoadElapsed(const ViewerState & state)
{
  const auto seconds=std::chrono::duration_cast<std::chrono::seconds>(
    std::chrono::steady_clock::now()-state.input_load_started).count();
  std::ostringstream out;out<<seconds/60<<':'<<std::setw(2)<<std::setfill('0')<<seconds%60;
  return out.str();
}

void BuildInputLoading(ViewerState & state)
{
  if (!state.loading_bag_source || !state.input_load) return;
  const auto & view=state.ui_layout.viewport;
  const float s=state.ui_layout.scale;
  if (view.w<280*s || view.h<170*s) return;
  const auto progress=state.input_load->Snapshot();
  const bool canceled=state.input_load->Cancelled();
  const float width=std::min(520*s,view.w-24*s), height=std::min(264*s,view.h-20*s);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{20*s,16*s});
  BeginPanel("Input loading",{view.x+(view.w-width)*.5f,view.y+(view.h-height)*.5f,width,height});
  ImGui::TextColored({.48f,.82f,.94f,1},"%s",canceled ? "Canceling..." : lidar_viewer::InputLoadStageName(progress.stage));
  ImGui::Spacing();
  const auto & source=state.bag_sources[*state.loading_bag_source];
  // Long paths stay available on hover without expanding the loading card.
  ImGui::TextUnformatted(source.display_name.c_str());
  Tip(source.path.u8string().c_str());
  ImGui::Spacing();
  const bool measured=progress.stage==lidar_viewer::InputLoadStage::Indexing && progress.total_messages>0;
  char overlay[80];
  if (measured) std::snprintf(overlay,sizeof(overlay),"%.0f%% of messages scanned",
    std::min(100.,100.*progress.messages/progress.total_messages));
  else std::snprintf(overlay,sizeof(overlay),"%s",canceled ? "Stopping reader..." : "Working...");
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram,{.25f,.68f,.78f,1});
  ImGui::ProgressBar(measured ? std::min(1.f,static_cast<float>(progress.messages)/progress.total_messages) :
    -static_cast<float>(ImGui::GetTime()*.6),{-1,12*s},"");
  ImGui::PopStyleColor();
  ImGui::TextDisabled("%s",overlay);
  if (progress.total_frames) ImGui::Text("%s / %s LiDAR frames",CountText(progress.frames).c_str(),CountText(progress.total_frames).c_str());
  else ImGui::Text("%s LiDAR frames",CountText(progress.frames).c_str());
  ImGui::TextDisabled("%s IMU samples  |  elapsed %s",CountText(progress.imu_samples).c_str(),LoadElapsed(state).c_str());
  ImGui::Spacing();
  ImGui::BeginDisabled(canceled);
  if (ImGui::Button("Cancel opening",{140*s,0})) state.input_load->Cancel();
  ImGui::EndDisabled();ImGui::SameLine();ImGui::TextDisabled("Esc");
  ImGui::End();ImGui::PopStyleVar();
}

void TraceUi(const char * id)
{
  if (!std::getenv("LIDAR_VIEWER_UI_TRACE") || ImGui::GetFrameCount()%2 || !ImGui::IsItemVisible()) return;
  const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
  std::cout<<"UI "<<id<<' '<<a.x<<' '<<a.y<<' '<<b.x<<' '<<b.y<<std::endl;
}

void UpdateRoiExport(ViewerState & state)
{
  try {
    if (const auto selected=state.export_picker.Poll()) {
      if (!selected->error.empty()) state.export_error=selected->error;
      else if (!selected->path.empty()) {
        const auto path=selected->path.u8string();
        if (path.size()>=state.export_directory.size()) state.export_error="The selected path is too long";
        else {std::fill(state.export_directory.begin(),state.export_directory.end(),0);
          std::copy(path.begin(),path.end(),state.export_directory.begin());}
      }
    }
  } catch (const std::exception & error) {state.export_error=error.what();}
  if (!state.export_future.valid() ||
      state.export_future.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return;
  try {
    const auto result=state.export_future.get();
    state.export_message=result.canceled ? "Export canceled" :
      "Saved "+CountText(result.points)+" points in "+std::to_string(result.frames)+" frames";
    SetTransientStatus(state,state.export_message);
    std::cout<<"ROI export: "<<state.export_message<<" | "<<result.destination.u8string()<<std::endl;
  } catch (const std::exception & error) {
    state.export_error=error.what();state.export_message.clear();
    SetTransientStatus(state,"ROI export failed: "+state.export_error);
    std::cerr<<"ROI export failed: "<<state.export_error<<std::endl;
  }
}

void StartRoiExport(ViewerState & state)
{
  if (state.export_future.valid() || state.export_picker.Active()) return;
  try {
    lidar_viewer::RoiExportRequest request;
    request.regions=state.display_roi;request.aligned_coordinates=state.export_aligned;
    request.calibration=state.calibration;request.imu_level_enabled=state.imu_level_enabled;
    request.imu_mount=state.imu_mount;request.global_ground_offset=state.global_ground_offset;
    request.profile=state.dataset_profile;
    const auto parent=ExpandFolderPath(state.export_directory.data());
    if (parent.empty()) throw std::runtime_error("Choose an export folder");
    if (state.live_mode) {
      if (!state.live_has_frame) throw std::runtime_error("Waiting for the first live frame");
      request.live_snapshot=state.base_frame;request.live_sequence=state.live_display_sequence;
      request.live_topic=state.live_points_topic.data();
    } else {
      if (state.frames.empty()) throw std::runtime_error("Open a recording first");
      if (state.export_scope==0) request.sources.push_back({state.frames[state.current_index],state.current_index});
      else {
        if (state.export_first<1 || state.export_last<state.export_first || state.export_stride<1 ||
            static_cast<std::size_t>(state.export_last)>state.frames.size())
          throw std::runtime_error("Choose a valid frame range and sampling step");
        for (std::size_t i=static_cast<std::size_t>(state.export_first-1);
            i<static_cast<std::size_t>(state.export_last);i+=static_cast<std::size_t>(state.export_stride))
          request.sources.push_back({state.frames[i],i});
      }
    }
    request.destination=lidar_viewer::AvailableRoiExportDestination(parent,request);
    state.export_total=request.live_snapshot ? 1 : request.sources.size();
    state.export_region_count=static_cast<std::size_t>(std::count_if(request.regions.boxes.begin(),request.regions.boxes.end(),
      [](const auto & b){return b.enabled;}));
    state.export_destination=request.destination;state.export_progress=std::make_shared<lidar_viewer::RoiExportProgress>();
    state.export_error.clear();state.export_message.clear();state.export_started_at=glfwGetTime();
    if (!state.export_worker) state.export_worker=std::make_unique<lidar_viewer::BackgroundWorker>();
    state.export_future=state.export_worker->Submit([request=std::move(request),progress=state.export_progress] {
      return lidar_viewer::ExportRoiDataset(request,progress);
    });
  } catch (const std::exception & error) {state.export_error=error.what();}
}

void BuildRoiExportDialog(ViewerState & state)
{
  const float s=state.ui_layout.scale;
  if (state.open_roi_export) {
    ImGui::OpenPopup("Export ROI dataset");state.open_roi_export=false;
    if (!state.export_directory[0]) {
      auto parent=lidar_viewer::UserHome()/"Documents";
      if (!fs::is_directory(parent)) parent=lidar_viewer::UserHome();
      const auto path=parent.u8string();
      if (path.size()<state.export_directory.size()) std::copy(path.begin(),path.end(),state.export_directory.begin());
    }
    state.export_last=static_cast<int>(std::min<std::size_t>(state.frames.size(),std::numeric_limits<int>::max()));
  }
  const auto display=ImGui::GetIO().DisplaySize;
  const float width=std::min(600*s,display.x-32*s);
  ImGui::SetNextWindowPos({display.x*.5f,display.y*.5f},ImGuiCond_Appearing,{.5f,.5f});
  ImGui::SetNextWindowSizeConstraints({width,0},{width,display.y-32*s});
  ImGui::SetNextWindowSize({width,0},ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Export ROI dataset",nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings)) return;
  const bool running=state.export_future.valid(),picking=state.export_picker.Active();
  const auto regions=std::count_if(state.display_roi.boxes.begin(),state.display_roi.boxes.end(),[](const auto & b){return b.enabled;});
  ImGui::TextWrapped("Export points inside %zu checked display regions.",
    running ? state.export_region_count : static_cast<std::size_t>(regions));
  ImGui::TextDisabled("NumPy XYZI + PCD + metadata  |  Original intensity");
  if (!running) {
    ImGui::Spacing();ImGui::BeginDisabled(picking);
    ImGui::TextUnformatted("Frames");
    if (state.live_mode) ImGui::Text("Current live frame #%llu",static_cast<unsigned long long>(state.live_display_sequence));
    else {
      ImGui::RadioButton("Current frame",&state.export_scope,0);TraceUi("export_current");
      ImGui::SameLine();ImGui::TextDisabled("%zu / %zu",state.current_index+1,state.frames.size());
      ImGui::RadioButton("Frame range",&state.export_scope,1);TraceUi("export_range");
      if (state.export_scope==1) {
        if (ImGui::BeginTable("##ExportRange",3,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_NoSavedSettings)) {
          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);ImGui::TextDisabled("First frame");
          ImGui::TableSetColumnIndex(1);ImGui::TextDisabled("Last frame");
          ImGui::TableSetColumnIndex(2);ImGui::TextDisabled("Every N frames");
          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);ImGui::SetNextItemWidth(-1);ImGui::InputInt("##ExportFirst",&state.export_first,0,0);TraceUi("export_first");
          ImGui::TableSetColumnIndex(1);ImGui::SetNextItemWidth(-1);ImGui::InputInt("##ExportLast",&state.export_last,0,0);TraceUi("export_last");
          ImGui::TableSetColumnIndex(2);ImGui::SetNextItemWidth(-1);ImGui::InputInt("##ExportStride",&state.export_stride,0,0);TraceUi("export_stride");
          ImGui::EndTable();
        }
      }
    }
    ImGui::Spacing();ImGui::TextUnformatted("Coordinates");
    if (ImGui::RadioButton("Sensor coordinates (original)",!state.export_aligned)) state.export_aligned=false;
    TraceUi("export_sensor");
    if (ImGui::RadioButton("Aligned coordinates (as in viewer)",state.export_aligned)) state.export_aligned=true;
    TraceUi("export_aligned");
    ImGui::TextDisabled("Frame time and coordinate transforms are saved with each sample.");
    ImGui::Spacing();ImGui::TextUnformatted("Destination folder");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x-96*s);
    ImGui::InputText("##ExportDirectory",state.export_directory.data(),state.export_directory.size());TraceUi("export_directory");
    ImGui::SameLine();
    if (ImGui::Button("Browse...",{88*s,0})) {
      try {state.export_picker.OpenDirectory(ExpandFolderPath(state.export_directory.data()));}
      catch (const std::exception & error) {state.export_error=error.what();}
    }
    TraceUi("export_browse");ImGui::EndDisabled();
    ImGui::TextDisabled("Names: recording, frame and ROI. Full details are saved in JSON.");
  }
  ImGui::Spacing();
  if (running) {
    ImGui::TextWrapped("%s",state.export_destination.u8string().c_str());
    const auto completed=state.export_progress->completed.load();
    ImGui::ProgressBar(static_cast<float>(completed)/std::max<std::size_t>(1,state.export_total),{-1,0});
    ImGui::Text("%zu / %zu frames  |  %s points  |  %.1f s",completed,state.export_total,
      CountText(state.export_progress->points.load()).c_str(),glfwGetTime()-state.export_started_at);
    ImGui::BeginDisabled(state.export_progress->cancel.load());
    if (ImGui::Button("Cancel export")) state.export_progress->cancel.store(true);
    TraceUi("export_cancel");ImGui::EndDisabled();ImGui::SameLine();
  } else {
    if (!state.export_message.empty()) {
      ImGui::TextColored({.42f,.85f,.65f,1},"%s",state.export_message.c_str());
      if (state.export_message!="Export canceled") {
        ImGui::TextWrapped("%s",state.export_destination.u8string().c_str());
        if (ImGui::SmallButton("Copy output path")) ImGui::SetClipboardText(state.export_destination.u8string().c_str());
        TraceUi("export_copy_path");
      }
    }
    if (!state.export_error.empty()) ImGui::TextWrapped("Export failed: %s",state.export_error.c_str());
    ImGui::TextWrapped("Exported samples are unlabeled. Add object boxes or point labels in your annotation tool.");
    ImGui::BeginDisabled(picking || regions==0 || state.loading_bag_source.has_value() || state.roi_gesture.active);
    if (lidar_viewer::ActionButton("##StartExport",lidar_viewer::UiIcon::Export,"Export dataset",{160*s,32*s},s)) StartRoiExport(state);
    TraceUi("export_start");ImGui::EndDisabled();ImGui::SameLine();
  }
  if (ImGui::Button(running ? "Hide" : "Close",{90*s,32*s})) ImGui::CloseCurrentPopup();
  TraceUi("export_close");
  ImGui::EndPopup();
}

void BuildRoiEditor(ViewerState & state)
{
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Tab)) state.roi_numeric_edit=false;
  const float s=state.ui_layout.scale;
  using lidar_viewer::IconButton;using lidar_viewer::UiIcon;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{8*s,8*s});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*s,5*s});
  const float width=ImGui::GetContentRegionAvail().x;
  ImGui::BeginDisabled(state.roi_gesture.active);
  for (const auto purpose:{ViewerState::RoiPurpose::Display,ViewerState::RoiPurpose::Ground}) {
    const bool display=purpose==ViewerState::RoiPurpose::Display,active=purpose==state.roi_purpose;
    if (!display) ImGui::SameLine();
    if (active) ImGui::PushStyleColor(ImGuiCol_Button,{.15f,.37f,.46f,1});
    if (ImGui::Button(display ? "Display ROI" : "Ground exclusion",{(width-8*s)*.5f,32*s}) && !active) {
      state.roi_purpose=purpose;state.roi_tool=ViewerState::RoiTool::Navigate;
      state.roi_numeric_edit=false;state.roi_reveal_selection=true;state.roi_details_requested=false;
      RebuildRoiGeometry(state);
    }
    TraceUi(display ? "roi_purpose_display" : "roi_purpose_ground");
    if (active) ImGui::PopStyleColor();
  }
  ImGui::EndDisabled();
  const bool display=state.roi_purpose==ViewerState::RoiPurpose::Display;
  ImGui::PushID(display ? "DisplayRegions" : "GroundRegions");
  const auto before=EditedRoi(state);bool history_restored=false;
  const bool pending=!lidar_viewer::SameRoiExclusions(state.roi,state.applied_exclusions);
  SectionTitle(display ? "Points to show" : "Ignore areas for ground");
  if (display) {
    ImGui::Checkbox("Show only points inside",&state.display_roi.enabled);TraceUi("display_roi_enabled");
    Tip("Only changes the view. Ground detection and object detection still receive the full scan. Multiple checked regions are combined.");
  }
  ImGui::BeginDisabled(state.roi_gesture.active || EditedRoi(state).boxes.size()>=lidar_viewer::kMaxRoiBoxes);
  if (ImGui::Button("+ Add region",{display ? width : width-86*s,32*s})) SelectRoiTool(state,ViewerState::RoiTool::Draw);
  TraceUi("roi_add");Tip(display ? "Draw a box to view the points inside. Changes appear immediately." :
    "Draw a box around points that should skip ground detection, then Apply. Points remain visible.");
  ImGui::EndDisabled();
  if (display) {
    ImGui::TextDisabled("%s / %s points shown",CountText(state.frame.points.size()).c_str(),CountText(state.processed_frame.points.size()).c_str());
  } else {
    ImGui::SameLine();ImGui::BeginDisabled(!pending || state.roi_gesture.active || state.applying_exclusions);
    ImGui::PushStyleColor(ImGuiCol_Button,{.15f,.37f,.46f,1});
    if (ImGui::Button("Apply",{78*s,32*s})) state.apply_exclusions_requested=true;
    TraceUi("roi_apply");ImGui::PopStyleColor();ImGui::EndDisabled();
    if (state.applying_exclusions) ImGui::TextColored({.48f,.82f,.94f,1},"Applying regions...");
    else if (pending) ImGui::TextColored({1,.74f,.4f,1},"Changes ready to apply");
    else ImGui::TextDisabled("%s points skip ground detection",CountText(state.exclusion_removed_count).c_str());
  }
  ImGui::Spacing();ImGui::AlignTextToFramePadding();ImGui::Text("Regions (%zu)",EditedRoi(state).boxes.size());
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x-106*s);
  ImGui::BeginDisabled(!EditedRoiHistory(state).CanUndo() || state.roi_gesture.active);
  if (IconButton("##UndoRoi",UiIcon::Undo,"Undo region edit. Ctrl+Z",s)) {RestoreRoiHistory(state,false);history_restored=true;}
  TraceUi("roi_undo");ImGui::EndDisabled();ImGui::SameLine(0,4*s);
  ImGui::BeginDisabled(!EditedRoiHistory(state).CanRedo() || state.roi_gesture.active);
  if (IconButton("##RedoRoi",UiIcon::Redo,"Redo region edit. Ctrl+Y",s)) {RestoreRoiHistory(state,true);history_restored=true;}
  TraceUi("roi_redo");ImGui::EndDisabled();ImGui::SameLine(0,4*s);
  if (IconButton("##Outlines",UiIcon::Eye,"Show region outlines",s,EditedRoi(state).show_boxes)) EditedRoi(state).show_boxes=!EditedRoi(state).show_boxes;
  TraceUi("roi_outlines");
  const float rows=static_cast<float>(std::clamp<std::size_t>(EditedRoi(state).boxes.size(),3,8));
  const float preferred_height=rows*38*s+12*s;
  const float list_height=std::min(preferred_height,std::clamp(ImGui::GetContentRegionAvail().y-(display ? 270 : 230)*s,110*s,285*s));
  if (std::abs(list_height-state.roi_list_height)>1) {state.roi_reveal_selection=true;state.roi_list_height=list_height;}
  if (ImGui::BeginListBox("##RoiList",{-1,list_height})) {
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,{0,.5f});
    ImGui::PushStyleColor(ImGuiCol_Header,{.12f,.30f,.39f,1});
    for (std::size_t i=0;i<EditedRoi(state).boxes.size();++i) {
      ImGui::PushID(static_cast<int>(i));auto & box=EditedRoi(state).boxes[i];
      ImGui::BeginDisabled(state.roi_gesture.active);
      const std::string label=(box.name.empty() ? "Unnamed region" : box.name)+"###Region";
      if (ImGui::Selectable(label.c_str(),i==SelectedRoi(state),0,{ImGui::GetContentRegionAvail().x-38*s,30*s})) {
        SelectedRoi(state)=i;state.roi_numeric_edit=false;SelectRoiTool(state,ViewerState::RoiTool::Move);
      }
      TraceUi(("roi_row_"+std::to_string(i)).c_str());
      if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) FocusSelectedRoi(state);
      if (i==SelectedRoi(state) && state.roi_reveal_selection) ImGui::SetScrollHereY(.5f);
      ImGui::SameLine();
      const std::string toggle_tip=std::string(box.enabled ? "Disable " : "Enable ")+
        (display ? "display crop: " : "ground exclusion: ")+box.name+
        ". Click the region name to select it for editing or deletion.";
      const bool inactive=!box.enabled;
      if (inactive) ImGui::PushStyleColor(ImGuiCol_Text,ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      if (IconButton("##Enabled",UiIcon::Power,toggle_tip.c_str(),s,box.enabled)) box.enabled=!box.enabled;
      TraceUi(("roi_enabled_"+std::to_string(i)).c_str());
      if (inactive) ImGui::PopStyleColor();
      ImGui::EndDisabled();ImGui::PopID();
    }
    ImGui::PopStyleColor();ImGui::PopStyleVar();
    state.roi_reveal_selection=false;ImGui::EndListBox();TraceUi("roi_list");
  }
  if (EditedRoi(state).boxes.empty()) ImGui::TextWrapped("Add a region, then drag a rectangle in the 2D view.");
  else {
    SelectedRoi(state)=std::min(SelectedRoi(state),EditedRoi(state).boxes.size()-1);
    ImGui::Text("Selected: %s",EditedRoi(state).boxes[SelectedRoi(state)].name.c_str());
    ImGui::BeginDisabled(state.roi_gesture.active);
    if (IconButton("##FocusRoi",UiIcon::Focus,"Find this region in the view",s)) {FocusSelectedRoi(state);SelectRoiTool(state,ViewerState::RoiTool::Move);}
    TraceUi("roi_focus");ImGui::SameLine(0,4*s);
    ImGui::BeginDisabled(EditedRoi(state).boxes.size()>=lidar_viewer::kMaxRoiBoxes);
    if (IconButton("##CopyRoi",UiIcon::Copy,"Duplicate this region. Ctrl+D",s)) DuplicateSelectedRoi(state);
    TraceUi("roi_duplicate");ImGui::EndDisabled();ImGui::SameLine(0,4*s);
    const auto delete_tip="Delete "+EditedRoi(state).boxes[SelectedRoi(state)].name+". Delete / Ctrl+Z to undo";
    if (IconButton("##DeleteRoi",UiIcon::Trash,delete_tip.c_str(),s)) DeleteSelectedRoi(state);
    TraceUi("roi_delete");ImGui::SameLine(0,12*s);
    if (ImGui::Button("Details...",{-1,30*s})) state.roi_details_requested=true;
    TraceUi("roi_details");ImGui::EndDisabled();
    if (!EditedRoi(state).boxes.empty()) {
      auto box=EditedRoi(state).boxes[SelectedRoi(state)];
      ImGui::TextDisabled("Height range (sensor Z, m)");
      float bottom=box.center.z-box.size.z*.5f,top=box.center.z+box.size.z*.5f;bool changed=false;
      const float input_x=ImGui::GetCursorPosX()+68*s;
      ImGui::AlignTextToFramePadding();ImGui::TextUnformatted("Bottom");ImGui::SameLine(input_x);
      ImGui::SetNextItemWidth(-1);
      changed|=ImGui::InputFloat("##RoiBottom",&bottom,.05f,.25f,"%.2f");
      TraceUi("roi_bottom");
      ImGui::AlignTextToFramePadding();ImGui::TextUnformatted("Top");ImGui::SameLine(input_x);
      ImGui::SetNextItemWidth(-1);
      changed|=ImGui::InputFloat("##RoiTop",&top,.05f,.25f,"%.2f");
      TraceUi("roi_top");
      if (changed) {
        if (std::isfinite(bottom) && std::isfinite(top) && top-bottom>=.05f-1e-6f) {
          box.center.z=(bottom+top)*.5f;box.size.z=top-bottom;
          try {lidar_viewer::ValidateRoiBox(box);EditedRoi(state).boxes[SelectedRoi(state)]=box;state.ui_error.clear();}
          catch (const std::exception & error) {state.ui_error=error.what();}
        } else state.ui_error="Top must be at least 0.05 m above Bottom";
      }
    }
  }
  if (state.roi_details_requested && !EditedRoi(state).boxes.empty()) {ImGui::OpenPopup("Region details");state.roi_details_requested=false;}
  ImGui::SetNextWindowSize({440*s,0},ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal("Region details",nullptr,ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings)) {
    if (EditedRoi(state).boxes.empty()) ImGui::CloseCurrentPopup();
    else {
      auto box=EditedRoi(state).boxes[SelectedRoi(state)];bool edited=false;
      std::array<char,128> name{};std::copy(box.name.begin(),box.name.end(),name.begin());
      ImGui::TextUnformatted("Name");ImGui::SetNextItemWidth(-1);
      if (ImGui::InputText("##RoiName",name.data(),name.size())) {box.name=name.data();edited=true;}
      TraceUi("roi_name");
      float center[]{box.center.x,box.center.y,box.center.z},size[]{box.size.x,box.size.y,box.size.z};
      ImGui::TextUnformatted("Position X / Y / Z (sensor, m)");ImGui::SetNextItemWidth(-1);
      edited|=ImGui::InputFloat3("##RoiCenter",center,"%.3f");TraceUi("roi_center");
      ImGui::TextUnformatted("Size X / Y / Z (m)");ImGui::SetNextItemWidth(-1);
      edited|=ImGui::InputFloat3("##RoiSize",size,"%.3f");TraceUi("roi_size");
      ImGui::TextUnformatted("Rotation (degrees)");ImGui::SetNextItemWidth(-1);
      edited|=ImGui::InputFloat("##RoiYaw",&box.yaw_deg,0,0,"%.2f");TraceUi("roi_yaw");
      box.center={center[0],center[1],center[2]};box.size={size[0],size[1],size[2]};
      if (edited) {
        try {lidar_viewer::ValidateRoiBox(box);box.yaw_deg=std::remainder(box.yaw_deg,360.f);EditedRoi(state).boxes[SelectedRoi(state)]=box;state.ui_error.clear();}
        catch (const std::exception & error) {state.ui_error=error.what();}
      }
      if (ImGui::CollapsingHeader("Coordinates for export")) {
        if (ImGui::Button("Copy 8 corners")) {
          std::ostringstream out;out<<std::setprecision(std::numeric_limits<float>::max_digits10);
          for (const auto & point:lidar_viewer::RoiCorners(EditedRoi(state).boxes[SelectedRoi(state)])) out<<point.x<<", "<<point.y<<", "<<point.z<<'\n';
          ImGui::SetClipboardText(out.str().c_str());
        }
      }
      if (ImGui::Button("Done",{-1,0})) {state.roi_reveal_selection=true;ImGui::CloseCurrentPopup();}
      TraceUi("roi_done");
    }
    ImGui::EndPopup();
  }
  if (!lidar_viewer::SameRoi(before,EditedRoi(state))) {
    if (!history_restored && !state.roi_numeric_edit) EditedRoiHistory(state).Record(before);
    state.roi_numeric_edit=ImGui::IsAnyItemActive();MarkRoiPending(state);RebuildRoiGeometry(state);
  }
  if (!ImGui::IsAnyItemActive()) state.roi_numeric_edit=false;
  if (display) {
    ImGui::Spacing();
    const bool can_export=(state.live_mode ? state.live_has_frame : !state.frames.empty()) &&
      std::any_of(state.display_roi.boxes.begin(),state.display_roi.boxes.end(),[](const auto & b){return b.enabled;});
    ImGui::BeginDisabled(!can_export || state.roi_gesture.active);
    if (lidar_viewer::ActionButton("##ExportRoi",UiIcon::Export,
        state.export_future.valid() ? "Export progress..." : "Export points...",{-1,32*s},s)) state.open_roi_export=true;
    TraceUi("roi_export");ImGui::EndDisabled();
  }
  ImGui::PopID();
  ImGui::PopStyleVar(2);
}

void BuildAppSettings(ViewerState & state)
{
  if (!state.open_app_settings) return;
  const float s=state.ui_layout.scale;
  ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x-16*s,70*s},ImGuiCond_Always,{1,0});
  ImGui::SetNextWindowSize({350*s,0});
  ImGui::Begin("Application settings",&state.open_app_settings,
    ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::TextDisabled("Shared by every recording");ImGui::Spacing();
  ImGui::TextUnformatted("Interface scale");ImGui::SetNextItemWidth(-1);
  if (ImGui::SliderFloat("##AppUiScale",&state.ui_scale,.75f,1.5f,"%.2fx",ImGuiSliderFlags_AlwaysClamp)) state.app_settings_dirty=true;
  const bool editing=ImGui::IsItemActive();
  if (ImGui::Button("Reset to 100%")) {state.ui_scale=1;state.app_settings_dirty=true;}
  ImGui::Spacing();ImGui::TextUnformatted("Inspector width");ImGui::SetNextItemWidth(-1);
  if (ImGui::SliderFloat("##InspectorWidth",&state.inspector_width,280,520,"%.0f px",ImGuiSliderFlags_AlwaysClamp)) state.app_settings_dirty=true;
  const bool width_editing=ImGui::IsItemActive();
  Tip("You can also drag the divider beside the inspector. Double-click it to restore the default width.");
  if (ImGui::Button("Reset panel width")) {state.inspector_width=320;state.app_settings_dirty=true;}
  if (state.app_settings_dirty && !editing && !width_editing && !state.inspector_resizing) SaveAppSettings(state);
  if (!state.app_settings_error.empty()) ImGui::TextWrapped("Could not save settings: %s",state.app_settings_error.c_str());
  ImGui::End();
}

fs::path CurrentDatasetRoot(const ViewerState & state)
{
  if(state.frames.empty() || !state.frames.front().dataset_sample) return {};
  if(state.current_bag_source && *state.current_bag_source<state.bag_sources.size()) {
    const auto path=state.bag_sources[*state.current_bag_source].path;return fs::is_directory(path) ? path : path.parent_path();
  }
  return state.frames.front().dataset_sample->root;
}

void BuildDatasetPanel(ViewerState & state)
{
  const float s=state.ui_layout.scale;
  const auto root=CurrentDatasetRoot(state);
  SectionTitle("Dataset samples");ImGui::TextWrapped("%s",root.filename().u8string().c_str());
  ImGui::TextDisabled("%zu samples  |  Left / Right to browse",state.frames.size());
  if(ImGui::Checkbox("Fit each sample",&state.dataset_fit_each)) state.dataset_fitted_index.reset();
  TraceUi("dataset_fit_each");
  const auto selected=state.current_index;
  if(ImGui::BeginListBox("##DatasetSamples",{-1,std::min(260*s,std::max(100*s,ImGui::GetContentRegionAvail().y-250*s))})) {
    ImGuiListClipper clip;clip.Begin(static_cast<int>(state.frames.size()));
    clip.IncludeItemByIndex(static_cast<int>(selected));
    while(clip.Step()) for(int i=clip.DisplayStart;i<clip.DisplayEnd;++i) {
      const auto & frame=state.frames[static_cast<std::size_t>(i)];ImGui::PushID(i);
      if(ImGui::Selectable(frame.label.c_str(),static_cast<std::size_t>(i)==selected)) TryLoadFrame(state,static_cast<std::size_t>(i));
      if(static_cast<std::size_t>(i)==selected && state.dataset_list_index!=selected) ImGui::SetScrollHereY(.5f);
      TraceUi(("dataset_sample_"+std::to_string(i)).c_str());ImGui::PopID();
    }
    ImGui::EndListBox();
    state.dataset_list_index=selected;
  }
  if(selected>=state.frames.size() || !state.frames[selected].dataset_sample) return;
  const auto & sample=*state.frames[selected].dataset_sample;
  ImGui::Text("Sample %zu / %zu",selected+1,state.frames.size());
  TraceUi(("dataset_current_"+std::to_string(selected)).c_str());
  ImGui::Text("%zu points",state.base_frame.points.size());
  ImGui::Text("Coordinates: %s",sample.coordinate_space.c_str());
  if(sample.source_frame) ImGui::Text("Source frame: %zu",sample.source_frame);
  if(!sample.category.empty()) ImGui::Text("Category: %s",sample.category.c_str());
  ImGui::TextWrapped("Labels: %s",sample.annotation_status.c_str());
  if(!sample.labels.empty()) ImGui::TextDisabled("Point label file present; validate before training.");
  if(ImGui::Button("Training settings",{-1,30*s})) state.open_training=true;
  TraceUi("dataset_train");
  if(ImGui::CollapsingHeader("Sample details")) {
    ImGui::TextWrapped("Points: %s",state.frames[selected].path.u8string().c_str());
    if(!sample.source.empty()) ImGui::TextWrapped("Source: %s",sample.source.c_str());
    if(!sample.metadata.empty()) ImGui::TextWrapped("Metadata: %s",sample.metadata.u8string().c_str());
  }
}

void BuildViewerInterface(ViewerState & state, GLFWwindow *)
{
  const auto layout = state.ui_layout;
  const float s = layout.scale;
  bool reprocess = false, alignment_changed=false, refresh_objects=false, refresh_detector=false;
  const bool has_input = state.live_mode ? state.live_has_frame : !state.frames.empty();
  const bool dataset_input=!state.live_mode && !state.frames.empty() && state.frames.front().dataset_sample!=nullptr;
  BeginPanel("Header", layout.header, false);
  ImGui::SetCursorPosY(15.f * s);
  ImGui::TextColored({.35f, .82f, .96f, 1.f}, "LiDAR");
  ImGui::SameLine();
  ImGui::TextUnformatted("Studio");
  ImGui::SameLine(190.f*s);
  ImGui::BeginDisabled(state.live.Active() || state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active() || state.dataset_picker.Active());
  if (lidar_viewer::ActionButton("##OpenFile",lidar_viewer::UiIcon::Folder,"Open file",{110*s,0},s)) StartBagPicker(state);
  ImGui::EndDisabled(); Tip("Open BIN, NPY, PCD or ROS bag. Ctrl+O.");
  ImGui::SameLine();
  ImGui::BeginDisabled(state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active());
  if (lidar_viewer::ActionButton("##Live",lidar_viewer::UiIcon::Live,"Live",{85*s,0},s)) state.open_live_dialog=true;
  ImGui::EndDisabled();
  ImGui::SameLine();ImGui::BeginDisabled(state.live.Active() || state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active() || state.dataset_picker.Active());
  const bool wide_header=layout.header.w>=1050*s;
  if(lidar_viewer::ActionButton("##OpenDataset",lidar_viewer::UiIcon::Folder,wide_header ? "Datasets" : nullptr,{(wide_header ? 110.f : 32.f)*s,0},s)) {
    try {auto root=CurrentDatasetRoot(state);if(root.empty()) root=lidar_viewer::UserHome();state.dataset_picker.OpenDirectory(root);}
    catch(const std::exception & error) {state.ui_error=error.what();}
  }
  TraceUi("open_dataset");Tip("Open a dataset folder and browse saved ROI samples.");ImGui::EndDisabled();ImGui::SameLine();
  if(lidar_viewer::ActionButton("##Training",lidar_viewer::UiIcon::Play,wide_header ? "Train" : nullptr,{(wide_header ? 86.f : 32.f)*s,0},s)) state.open_training=true;
  TraceUi("open_training");Tip("Configure, validate and run model training.");
  const float header_right = std::max(ImGui::GetItemRectMax().x+16*s, layout.header.w - 360.f * s);
  ImGui::SameLine(header_right);
  ImGui::TextColored(!state.ui_error.empty() ? ImVec4(1.f, .56f, .48f, 1) :
    (state.loading_bag_source ? ImVec4(.44f, .76f, 1, 1) :
      (state.is_playing ? ImVec4(.42f, .85f, .65f, 1) : ImVec4(.76f, .81f, .88f, 1))),
    "%s", !state.ui_error.empty() ? "ERROR" :
      (state.live_mode ? (state.live_stats.connected ? (state.live_paused ? "FROZEN" :
        (state.live_stats.clouds==0 ? "WAITING" : (state.live_stats.cloud_age_ms>1000 ? "STALE" : "LIVE"))) : "DISCONNECTED") :
      (state.loading_bag_source ? "OPENING INPUT" :
        (has_input ? (state.is_playing ? "PLAYING" : "PAUSED") : "READY"))));
  ImGui::SameLine(layout.header.w-115*s);
  if (ImGui::SmallButton("Help")) ImGui::OpenPopup("Shortcuts");
  if (ImGui::BeginPopup("Shortcuts")) {
    ImGui::TextUnformatted("Keyboard & mouse");
    ImGui::Separator();
    ImGui::TextUnformatted("Ctrl+O       Open file chooser\nCtrl+S       Save profile\nR            Draw ROI in top view\nCtrl+Z / Y   Undo / redo ROI\nCtrl+D       Duplicate ROI (ROI tab)\nDelete       Delete ROI (ROI tab)\nEsc          Cancel ROI tool / quit\nSpace        Play / pause\nN / P        Next / previous frame\nF            Fit all points\nC            Cycle point colors\n[ / ]        Point size\nQ / E        Rotate 2D view\nLeft drag    Orbit (3D) / pan (2D)\nRight drag   Pan (3D) / rotate (2D)\nWheel        Zoom");
    ImGui::EndPopup();
  }
  ImGui::SameLine(layout.header.w-48*s);
  if (lidar_viewer::IconButton("##ApplicationSettings",lidar_viewer::UiIcon::Settings,"Application settings",s,state.open_app_settings))
    state.open_app_settings=!state.open_app_settings;
  ImGui::End();

  BeginPanel("Inspector", layout.sidebar, false);
  ImGui::TextUnformatted("Inspector");
  ImGui::SameLine();
  ImGui::TextDisabled("  %s", state.live_mode ? "LIVE ROS 2" : has_input ?
    (dataset_input ? "DATASET" : state.frames.front().format == InputFormat::Ros2Bag ? "ROS 2 BAG" : "BIN") : "NO DATA");
  ImGui::Spacing();
  ImGui::BeginChild("InspectorBody", {0, std::max(1.f, ImGui::GetContentRegionAvail().y - 84.f * s)},
    ImGuiChildFlags_None,ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {1.f * s, 6.f * s});
  const bool tabs_open = ImGui::BeginTabBar("InspectorTabs");
  ImGui::PopStyleVar();
  state.roi_tab_active=false;
  const auto begin_tab = [s,&state](const char * label) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {1.f * s, 6.f * s});
    const bool is_roi=std::string(label)=="ROI";
    const bool is_data=std::string(label)=="Data";
    const bool open=ImGui::BeginTabItem(label,nullptr,(is_roi && state.roi_tab_requested) || (is_data && state.dataset_tab_requested) ? ImGuiTabItemFlags_SetSelected : 0);
    TraceUi((std::string("tab_")+label).c_str());
    ImGui::PopStyleVar();
    if (open) {
      ImGui::BeginChild(label,{0,std::max(1.f,ImGui::GetContentRegionAvail().y-(state.ui_error.empty() ? 0 : 90*s))});
      if (is_roi) {if (state.roi_tab_requested) ImGui::SetScrollY(0);state.roi_tab_requested=false;state.roi_tab_active=true;}
      if(is_data) state.dataset_tab_requested=false;
    }
    return open;
  };
  const auto end_tab=[] {ImGui::EndChild();ImGui::EndTabItem();};
  ImGui::BeginDisabled(state.loading_bag_source.has_value());
  if (tabs_open) {
    if(dataset_input && begin_tab("Data")) {BuildDatasetPanel(state);end_tab();}
    if (begin_tab("View")) {
      SectionTitle("Point appearance");
      ImGui::TextDisabled("Color by");ImGui::SetNextItemWidth(-1);
      ImGui::Combo("##Color", &state.color_mode, "Intensity\0Height\0Uniform\0");
      ImGui::TextDisabled("Point size");ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##PointSize", &state.camera.point_size, 1, 8, "%.1f px", ImGuiSliderFlags_AlwaysClamp);
      if (state.color_mode==1) {
        ImGui::TextDisabled("Height range (m)");ImGui::SetNextItemWidth(-1);
        ImGui::DragFloatRange2("##HeightRange",&state.height_color_min,&state.height_color_max,.05f,-100,100,"%.2f","%.2f");
        if (!std::isfinite(state.height_color_min)) state.height_color_min=-1;
        if (!std::isfinite(state.height_color_max)) state.height_color_max=4;
        state.height_color_max=std::max(state.height_color_min+.01f,state.height_color_max);
      }
      ImGui::Spacing();ImGui::Spacing();ImGui::Checkbox("Reference grid",&state.grid_settings.enabled);
      if (ImGui::CollapsingHeader("Grid settings")) {
        bool dirty=false;ImGui::SetNextItemWidth(-1);
        dirty|=ImGui::SliderFloat("##Spacing",&state.grid_settings.spacing,.1f,20,"Spacing: %.1f m",ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetNextItemWidth(-1);dirty|=ImGui::SliderInt("##Extent",&state.grid_settings.count,2,100,"Extent: %d cells",ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetNextItemWidth(-1);dirty|=ImGui::SliderFloat("##GridLines",&state.grid_settings.thickness,1,4,"Line width: %.1f",ImGuiSliderFlags_AlwaysClamp);
        if (dirty) MarkGridDirty(state);
      }
      if (ImGui::CollapsingHeader("Camera")) {
        if (state.camera.projection_mode==CameraState::ProjectionMode::View2D) {
          float degrees=state.camera.yaw*180.f/kPi;ImGui::TextDisabled("Top view rotation");ImGui::SetNextItemWidth(-1);
          if (ImGui::DragFloat("##TopRotation",&degrees,.5f,-180,180,"%.1f deg")) state.camera.yaw=std::remainder(degrees,360.f)*kPi/180.f;
          Tip("Right drag or Q/E rotates the view. Ctrl+click for an exact angle.");
        }
        if (ImGui::Button("Sensor origin view",{-1,0})) {
          const bool top=state.camera.projection_mode==CameraState::ProjectionMode::View2D;
          state.camera.target={0,0,0};state.camera.distance=35;
          state.camera.yaw=top ? 0 : -.8f;state.camera.pitch=top ? -1.5707f : .55f;
          StoreCurrentProjectionView(state.camera);
        }
      }
      end_tab();
    }
    if (!dataset_input && begin_tab("Align")) {
      SectionTitle("IMU leveling");
      reprocess|=ImGui::Checkbox("IMU auto level",&state.imu_level_enabled);
      TraceUi("align_auto");
      const auto available=lidar_viewer::ComputeImuLeveling(state.base_frame.imu,state.imu_mount);
      const auto resolved=lidar_viewer::ResolveDisplayAlignment(state.calibration,state.imu_level_enabled,available);
      const bool automatic=resolved.imu_active;
      ImGui::TextColored(automatic ? ImVec4(.42f,.85f,.65f,1) :
        (state.imu_level_enabled ? ImVec4(1,.75f,.42f,1) : ImVec4(.7f,.8f,.9f,1)),"%s",
        automatic ? "AUTO / IMU ACTIVE" : (state.imu_level_enabled ? "MANUAL FALLBACK" : "MANUAL"));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{8*s,5*s});
      ImGui::BeginChild("AlignmentStatus",{0,80*s},ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
      {
        ImGui::TextWrapped("%s",automatic ? (available.axes_corrected ? "Sensor axes normalized." : "Gravity alignment active.") :
          (!state.imu_level_enabled && available.applied) ? "IMU available. Auto level is off." : available.status.c_str());
        Tip(available.status.c_str());
        if (automatic && std::abs(available.roll_deg)>90)
          ImGui::TextWrapped("Large roll: verify mount axes.");
        else if (state.imu_level_enabled) ImGui::TextWrapped("%s",automatic ?
          "Pitch/roll below are from IMU." : "Using editable manual pitch / roll.");
      }
      ImGui::EndChild();ImGui::PopStyleVar();
      SectionTitle("Applied display transform");
      if (ImGui::BeginTable("AlignmentValues",2,ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Axis",ImGuiTableColumnFlags_WidthFixed,92*s);
        ImGui::TableSetupColumn("Value",ImGuiTableColumnFlags_WidthStretch);
        const auto input=[&](const char * label,const char * id,float & value,bool read_only,const char * format,float step,bool height=false) {
          ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::AlignTextToFramePadding();ImGui::TextUnformatted(label);
          ImGui::TableSetColumnIndex(1);ImGui::SetNextItemWidth(-1);
          const float previous=value;
          const bool changed=ImGui::InputFloat(id,&value,read_only ? 0 : step,read_only ? 0 : step*10,format,
            read_only ? ImGuiInputTextFlags_ReadOnly : 0);
          TraceUi(id);
          if (read_only) Tip("Controlled by IMU. Turn off auto level to edit the saved manual angle.");
          else Tip(height ? "Display height in metres (-5 to 5)." : "Type an angle in degrees. Full turns wrap into -180 to 180.");
          if (changed && !read_only) {
            if (!std::isfinite(value)) {value=previous;state.ui_error="Alignment values must be finite.";return false;}
            value=height ? std::clamp(value,-5.f,5.f) : std::remainder(value,360.f);
          }
          return changed && !read_only;
        };
        reprocess|=input("Yaw","##Yaw",state.calibration.yaw_deg,false,"%.2f deg",.1f);
        float pitch=resolved.effective.pitch_deg,roll=resolved.effective.roll_deg;
        if (input(automatic ? "Pitch (IMU)" : "Pitch","##Pitch",pitch,automatic,"%.2f deg",.1f)) {
          state.calibration.pitch_deg=pitch;reprocess=true;
        }
        if (input(automatic ? "Roll (IMU)" : "Roll","##Roll",roll,automatic,"%.2f deg",.1f)) {
          state.calibration.roll_deg=roll;reprocess=true;
        }
        reprocess|=input("Height Z","##Z",state.calibration.z_m,false,"%.3f m",.01f,true);
        ImGui::EndTable();
      }
      const float half=(ImGui::GetContentRegionAvail().x-10*s)*.5f;
      if (ImGui::Button("Reset rotation",{half,0})) {
        lidar_viewer::ResetDisplayRotation(state.calibration,state.imu_level_enabled);reprocess=true;
      }
      TraceUi("align_reset_rotation");Tip("Turn off IMU correction and reset yaw/pitch/roll to zero. Keep sensor height and mount settings.");
      ImGui::SameLine();
      if (ImGui::Button("Reset all",{half,0})) {
        state.calibration={};state.imu_level_enabled=false;reprocess=true;
      }
      TraceUi("align_reset_all");Tip("Turn off IMU correction and reset display angles and height. Sensor mounting settings are preserved.");
      if (state.base_frame.imu && !state.base_frame.imu->topic.empty()) {
        const auto & imu=*state.base_frame.imu;
        ImGui::TextDisabled("%s -> %s",imu.imu_frame.c_str(),imu.lidar_frame.c_str());
        Tip(imu.topic.c_str());
        if (imu.up) ImGui::TextDisabled("%s clock | offset %.1f ms",state.live_mode ? "Live" : "Bag",imu.skew_ms);
        const bool static_mount=imu.imu_to_lidar.has_value();
        const bool matches=state.imu_mount.confirmed && state.imu_mount.imu_frame==imu.imu_frame &&
          state.imu_mount.lidar_frame==imu.lidar_frame;
        ImGui::TextDisabled("Mount: %s",static_mount ? "recorded static TF" : (matches ? "profile" : "not configured for this frame"));
        const bool mounting_open=ImGui::CollapsingHeader("Sensor mounting (advanced)");TraceUi("align_mounting");
        if (mounting_open) {
          ImGui::TextWrapped("Sensor-axis conversion, not the floor tilt. Display corrections are below.");
          if (static_mount) ImGui::TextWrapped("Static TF is active; profile mounting values are not used.");
          ImGui::BeginDisabled(static_mount);
          if (ImGui::Checkbox("Use profile mount",&state.imu_mount.confirmed)) {
            if (state.imu_mount.confirmed) {
              state.imu_mount.imu_frame=imu.imu_frame;state.imu_mount.lidar_frame=imu.lidar_frame;
            }
            reprocess=true;
          }
          ImGui::TextWrapped("Profile pair: %s -> %s",state.imu_mount.imu_frame.c_str(),state.imu_mount.lidar_frame.c_str());
          if (!matches && state.imu_mount.confirmed && ImGui::Button("Use current frame pair",{-1,0})) {
            state.imu_mount.imu_frame=imu.imu_frame;state.imu_mount.lidar_frame=imu.lidar_frame;reprocess=true;
          }
          float mount[]{state.imu_mount.roll_deg,state.imu_mount.pitch_deg,state.imu_mount.yaw_deg};
          ImGui::TextUnformatted("Mount roll / pitch / yaw (deg)");ImGui::SetNextItemWidth(-1);
          if (ImGui::InputFloat3("##ImuMountRpy",mount,"%.2f")) {
            if (std::all_of(mount,mount+3,[](float v){return std::isfinite(v);})) {
              state.imu_mount.roll_deg=std::remainder(mount[0],360.f);
              state.imu_mount.pitch_deg=std::remainder(mount[1],360.f);
              state.imu_mount.yaw_deg=std::remainder(mount[2],360.f);reprocess=true;
            } else state.ui_error="Mount angles must be finite.";
          }
          reprocess|=ImGui::Checkbox("Automatic sensor axes",&state.imu_mount.auto_axes);
          TraceUi("align_axes");
          Tip("For upright VN100 NED/FRD and XT32 sensors, recognize raw versus prepared axes without duplicating a 180-degree conversion. Static TF takes priority. Disable for an intentionally inverted installation.");
          ImGui::EndDisabled();
        }
      }
      ImGui::TextWrapped("Display coordinates only. Raw input and recorded bags are unchanged.");
      end_tab();
    }
    alignment_changed=reprocess;
    if (begin_tab("Ground")) {
      auto & ground = state.algorithm_config.ground_removal;
      SectionTitle("Ground segmentation");
      if (ImGui::Checkbox("Classify ground", &state.ground_removal_enabled)) {
        reprocess=true;
        if (!state.ground_removal_enabled) state.ground_view=ViewerState::GroundView::All;
      }
      ImGui::TextDisabled("Show");ImGui::SetNextItemWidth(-1);
      int ground_view=static_cast<int>(state.ground_view);
      if (ImGui::Combo("##GroundView",&ground_view,"All points\0Ground only\0Obstacles only\0")) {
        state.ground_view=static_cast<ViewerState::GroundView>(ground_view);
        if (state.ground_view!=ViewerState::GroundView::All && !state.ground_removal_enabled) {
          state.ground_removal_enabled=true;reprocess=true;
        }
      }
      Tip("Display classified points only. ROI remains active. Switching views reuses the existing result and keeps playback running.");
      ImGui::TextDisabled("Method");
      const char * modes[] = {"fast_grid", "gseg3d", "patchworkpp", "fugseg", "grid", "ransac"};
      const char * labels[] = {"Fast grid", "GSeg3D", "Patchwork++", "FugSeg", "Grid", "RANSAC"};
      int mode = 0;
      for (int i = 0; i < 6; ++i) if (ground.mode == modes[i]) mode = i;
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##GroundMethod", &mode, labels, 6)) {
        ground.mode = modes[mode];
        state.ground_removal_enabled=true;
        reprocess = true;
      }
      ImGui::TextDisabled("%s",ground.mode=="gseg3d" ? "Connected planes / CPU" :
        ground.mode=="patchworkpp" ? "Adaptive local planes / CPU" : ground.mode=="fugseg" ? "Uncertainty-aware terrain / CPU" :
        ground.mode=="fast_grid" ? "Local terrain grid" : ground.mode=="ransac" ? "Global plane + local check" : "Connected height cells");
      Tip(ground.mode=="ransac" ? "Fits one dominant plane. Prefer local-plane methods when the terrain changes slope." :
        ground.mode=="patchworkpp" ? "Original Patchwork++: adaptive local planes, suitable for slopes and uneven roads. Sensor height follows Align." :
        ground.mode=="fugseg" ? "Original FugSeg: uncertainty-aware polar terrain extraction, including sparse and occluded areas." :
        ground.mode=="gseg3d" ? "Original GSeg3D: plane geometry and connected 3D cells. Ground reference controls the region-growing seeds." :
        "Ground reference and slope limits guide connected local heights. Compare Ground only and Obstacles only when tuning.");
      if (ground.mode == "gseg3d" || ground.mode == "fast_grid") {
        float cell = ground.gseg3d_cell_size_x_m;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderFloat("##GsegCells", &cell, .2f, 3.f, "XY cell: %.2f m", ImGuiSliderFlags_AlwaysClamp)) {
          ground.gseg3d_cell_size_x_m = ground.gseg3d_cell_size_y_m = cell;
          reprocess = true;
        }
        if (ground.mode == "fast_grid") {
          ImGui::SetNextItemWidth(-1);
          reprocess |= ImGui::SliderFloat("##SeedCells", &ground.gseg3d_cuda_cell_size_m, .25f, 2.f, "Fine cell: %.2f m", ImGuiSliderFlags_AlwaysClamp);
        }
        ImGui::SetNextItemWidth(-1);
        reprocess |= ImGui::SliderFloat("##Inliers", &ground.gseg3d_ground_inlier_threshold_m, .02f, .6f, "Inlier limit: %.2f m", ImGuiSliderFlags_AlwaysClamp);
      } else if (ground.mode=="fugseg") {
        ImGui::TextWrapped("Uncertainty-aware polar grid / TITS 2026");
        ImGui::SetNextItemWidth(-1);
        reprocess|=ImGui::SliderFloat("##FugInlier",&ground.fugseg_inlier_threshold_m,.02f,.5f,"Ground thickness: %.2f m",ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::CollapsingHeader("FugSeg settings")) {
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##FugBin",&ground.fugseg_bin_size_m,.25f,3.f,"Radial bin: %.2f m",ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderInt("##FugSegments",&ground.fugseg_num_segments,30,720,"Angular segments: %d",ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##FugSlope",&ground.fugseg_max_slope_change_deg,1,45,"Slope change: %.1f deg",ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##FugGap",&ground.fugseg_max_gap_m,1,30,"Max radial gap: %.1f m",ImGuiSliderFlags_AlwaysClamp);
          ImGui::TextDisabled("Angular noise H / V (deg)");ImGui::SetNextItemWidth(-1);
          float noise[]={ground.fugseg_horizontal_noise_deg,ground.fugseg_vertical_noise_deg};
          if (ImGui::InputFloat2("##FugAngleNoise",noise,"%.3f")) {
            if (std::isfinite(noise[0]) && std::isfinite(noise[1])) {
              ground.fugseg_horizontal_noise_deg=std::clamp(noise[0],.001f,1.f);
              ground.fugseg_vertical_noise_deg=std::clamp(noise[1],.001f,1.f);reprocess=true;
            }
          }
          Tip("Measurement standard deviations used by the original uncertainty model. Defaults are from the upstream KITTI configuration; tune them to measured sensor noise.");
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##FugRangeNoise",&ground.fugseg_range_noise_m,.001f,.1f,"Range noise: %.3f m",ImGuiSliderFlags_AlwaysClamp);
        }
      } else if (ground.mode=="patchworkpp") {
        ImGui::SetNextItemWidth(-1);
        reprocess|=ImGui::SliderFloat("##PatchDistance",&ground.patchwork_distance_threshold_m,.02f,.5f,"Ground thickness: %.2f m",ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::CollapsingHeader("Patchwork++ settings")) {
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##PatchSeeds",&ground.patchwork_seed_threshold_m,.02f,.5f,"Seed height: %.2f m",ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::SliderFloat("##PatchSlope",&ground.max_ground_angle_deg,1,45,"Max slope: %.0f deg",ImGuiSliderFlags_AlwaysClamp);
        }
      } else {
        ImGui::SetNextItemWidth(-1);
        reprocess |= ImGui::SliderFloat("##GroundCells", &ground.candidate_grid_cell_m, .1f, 3.f, "Cell size: %.2f m", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetNextItemWidth(-1);
        reprocess|=ImGui::SliderFloat("##PlaneDistance",&ground.final_distance_threshold_m,.02f,.5f,"Ground thickness: %.2f m",ImGuiSliderFlags_AlwaysClamp);
      }
      if (ground.mode=="gseg3d" || ground.mode=="fast_grid" || ground.mode=="patchworkpp" || ground.mode=="fugseg") {
        if (ImGui::CollapsingHeader("Ground reference")) {
          ImGui::TextDisabled("Expected ground Z after Align");ImGui::SetNextItemWidth(-1);
          reprocess|=ImGui::DragFloat("##GroundReferenceZ",&ground.gseg3d_ground_height_m,.01f,-5,5,"%.2f m",ImGuiSliderFlags_AlwaysClamp);
          Tip("Use 0 m when Align has placed the ground on the grid. This is a ground seed height, not a display translation.");
          if (ground.mode=="gseg3d") {
            ImGui::SetNextItemWidth(-1);
            reprocess|=ImGui::SliderFloat("##GroundContinuity",&ground.gseg3d_max_ground_height_deviation_m,.05f,.6f,"Height continuity: %.2f m",ImGuiSliderFlags_AlwaysClamp);
            Tip("Maximum centroid height change between connected cells. A limit below slope x cell width breaks ramps.");
          }
        }
      }
#ifdef LIDAR_VIEWER_HAS_CUDA
      if (ground.mode!="gseg3d" && ground.mode!="patchworkpp" && ground.mode!="fugseg")
        reprocess |= ImGui::Checkbox("Use CUDA", &ground.use_cuda);
#else
      if (ground.mode!="gseg3d" && ground.mode!="patchworkpp" && ground.mode!="fugseg") ImGui::TextDisabled("CPU backend");
#endif
      Metric("Ground points", CountText(state.status_ground));
      ImGui::TextDisabled("Processing: %.1f ms",state.status_processing_ms);
      end_tab();
    }
    if (begin_tab("ROI")) {
      BuildRoiEditor(state);
      end_tab();
    }
    if (begin_tab("Objects")) {
      auto & detection=state.algorithm_config.detection;
      auto & clustering=state.algorithm_config.clustering;
      SectionTitle("Object analysis");
      bool enabled=detection.enabled || clustering.enabled;
      if (enabled) state.object_method=detection.enabled ? (detection.model=="short_range" ? 3 : 0) : (clustering.mode=="euclidean_3d" ? 1 : 2);
      const bool toggled=ImGui::Checkbox("Enable objects",&enabled);
      if (state.object_method==0 || state.object_method==3) {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x-65*s);ImGui::BeginDisabled(!enabled);
        if (ImGui::Checkbox("IDs",&state.tracking_params.enabled)) {
          ResetTracking(state);TrackCurrentDetections(state);FilterDetections(state);
        }
        TraceUi("tracking_ids");
        ImGui::EndDisabled();Tip("Keep the same ID across frames using 3D Kalman prediction and global matching.");
      }
      ImGui::TextDisabled("Method");ImGui::SetNextItemWidth(-1);
      const bool method_changed=ImGui::Combo("##ObjectMethod",&state.object_method,
        "CenterPoint tiny (fast)\0Euclidean 3D (clusters)\0Projected XY (clusters)\0AWML ShortRange (quality)\0");
      const bool learned=state.object_method==0 || state.object_method==3;
      if (toggled || method_changed) {
        const bool was_detection=detection.enabled;
        const bool was_clustering=clustering.enabled;
        if (method_changed && learned) {
          const std::string model=state.object_method==3 ? "short_range" : "tiny";
          if (model!=detection.model) {
            detection.model=model;detection.min_confidence=model=="short_range" ? .5f : .35f;
            ++state.detector_revision;state.detector_completed=0;
            state.full_detections.clear();state.detector_error.clear();
            ResetTracking(state);
          }
        }
        detection.enabled=enabled && learned;
        clustering.enabled=enabled && !learned;
        if (!learned) clustering.mode=state.object_method==1 ? "euclidean_3d" : "euclidean";
        if (was_detection && !detection.enabled) {state.detector_cli_override=false;ResetTracking(state);}
        if (detection.enabled && !was_detection) {
          refresh_detector=true;
          TrackCurrentDetections(state);
          if (state.detector && (!state.detector->Status().configured || state.detector->Status().failed)) state.detector_retry=true;
        }
        refresh_objects=was_clustering!=clustering.enabled || clustering.enabled;
        if (clustering.enabled && !state.ground_removal_enabled) {state.ground_removal_enabled=true;reprocess=true;}
        FilterDetections(state);
      }
      if (learned) {
        ImGui::TextDisabled("%s",detection.model=="short_range" ? "TIER IV AWML / ConvNeXt 2.0" : "Autoware CenterPoint tiny v4.1");
        Tip(detection.model=="short_range" ?
          "Original AWML model; XY range +/-51.2 m. Uses the full gravity-aligned scan before ground/ROI filtering. Ground height and IMU alignment must be correct." :
          "Original Autoware tiny model; XY range +/-76.8 m. Uses the full gravity-aligned scan before ground/ROI filtering. Ground height and IMU alignment must be correct.");
        ImGui::Spacing();SectionTitle("Targets");
        const std::array<int,5> classes={3,4,0,2,1};
        bool filter_changed=false;
        if (ImGui::SmallButton("All")) {detection.classes=lidar_viewer::algorithms::DetectionParams::AllClasses;filter_changed=true;}
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear")) {detection.classes=0;filter_changed=true;}
        if (ImGui::BeginTable("##TargetClasses",2,ImGuiTableFlags_SizingStretchSame)) {
          for (const int id:classes) {
            ImGui::TableNextColumn();bool selected=(detection.classes & (1u<<id))!=0;
            if (ImGui::Checkbox(lidar_viewer::algorithms::DetectionClassName(id),&selected)) {
              if (selected) detection.classes|=1u<<id;else detection.classes&=~(1u<<id);
              filter_changed=true;
            }
          }
          ImGui::EndTable();
        }
        if (!detection.classes) ImGui::TextColored({1,.74f,.4f,1},"No targets selected");
        ImGui::SetNextItemWidth(-1);
        filter_changed|=ImGui::SliderFloat("##DetectionScore",&detection.min_confidence,.1f,.9f,"Min score: %.2f",ImGuiSliderFlags_AlwaysClamp);
        Tip("Class and score selections filter cached model results instantly. They do not reload the model or rerun inference. Scores are not calibrated probabilities.");
        if (filter_changed) FilterDetections(state);
        if (detection.min_confidence<(detection.model=="short_range" ? .5f : .35f)) {
          ImGui::TextColored({1,.74f,.4f,1},"Low-confidence candidates visible");
          Tip("Lower scores increase false positives. The quality preset starts at 0.50; this is an application default, not a calibrated probability.");
        }
        if (enabled) {
          ImGui::Spacing();SectionTitle("Runtime");
          const auto status=state.detector ? state.detector->Status() : lidar_viewer::DetectorStatus{};
          if (status.ready) {
            if (state.runtime_session!=state.tracking_session) {
              state.runtime_session=state.tracking_session;state.runtime_sample_valid=false;
            }
            if (state.detector_completed==state.detector_revision &&
                (!state.runtime_sample_valid || filter_changed || ImGui::GetTime()-state.runtime_sample_time>=1.)) {
              state.runtime_sample_valid=true;state.runtime_sample_time=ImGui::GetTime();
              state.runtime_shown=state.detections.size();state.runtime_candidates=state.full_detections.size();
              state.runtime_latency_ms=state.detector_latency_ms;state.runtime_detail=state.detector_detail;
            }
            Metric("Backend",status.provider=="CUDAExecutionProvider" ? "CUDA GPU" : "CPU");
            Metric("Last shown / found",state.runtime_sample_valid ?
              std::to_string(state.runtime_shown)+" / "+std::to_string(state.runtime_candidates) : "-- / --");
            if (state.runtime_sample_valid) {ImGui::TextDisabled("Pipeline: %.1f ms",state.runtime_latency_ms);Tip(state.runtime_detail.c_str());}
            else ImGui::TextDisabled("Pipeline: waiting for first result");
          } else ImGui::TextWrapped("%s",status.message.empty() ? "Waiting for LiDAR data..." : status.message.c_str());
          if (!state.detector_error.empty()) ImGui::TextWrapped("%s",state.detector_error.c_str());
          if (!status.configured || status.failed) {
            ImGui::TextWrapped("Setup: python3 tools/setup_detector.py\nNVIDIA GPU: add --cuda");
            if (ImGui::Button("Retry model",{-1,0})) state.detector_retry=true;
          }
        }
        const bool tracking_options=ImGui::CollapsingHeader("Tracking options");TraceUi("tracking_options");
        if (tracking_options) {
          ImGui::TextDisabled("Sensor-relative position and speed");
          Tip("No odometry is recorded in the current sensor bags. Motion is relative to the leveled LiDAR; these are not world-frame velocities.");
          bool reset=false;
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##TrackHold",&state.tracking_params.max_missed_s,.1f,2.f,"Keep missing: %.1f s");
          TraceUi("tracking_hold");
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderInt("##TrackConfirm",&state.tracking_params.confirmation_hits,1,10,"Confirm after: %d hits");
          TraceUi("tracking_confirm");
          ImGui::Checkbox("Motion trails",&state.tracking_params.show_trails);
          ImGui::SameLine();ImGui::Checkbox("Speed",&state.tracking_params.show_speed);
          if (ImGui::Button("Reset tracks")) reset=true;
          TraceUi("tracking_reset");
          if (reset) {ResetTracking(state);TrackCurrentDetections(state);FilterDetections(state);}
          if (state.tracker && state.tracking_params.enabled) {
            const auto & stats=state.tracker->Stats();
            ImGui::TextDisabled("%zu confirmed  |  %zu new  |  %zu predicted",stats.confirmed,stats.tentative,stats.coasting);
            ImGui::TextDisabled("Tracking: %.2f ms",stats.processing_ms);
            if (!state.base_frame.timestamp_ns) ImGui::TextDisabled("Scan rate: %.1f Hz (assumed)",state.input_fps_estimate.value_or(10.));
            if (stats.capacity_dropped) ImGui::TextWrapped("Tracking capacity reached: %zu candidates without IDs",stats.capacity_dropped);
          }
          if (!state.tracking_error.empty()) ImGui::TextWrapped("Tracking: %s",state.tracking_error.c_str());
        }
        if (ImGui::CollapsingHeader("Cluster overlay")) {
          if (ImGui::Checkbox("Show obstacle clusters",&clustering.enabled)) {
            refresh_objects=true;
            if (clustering.enabled && !state.ground_removal_enabled) {state.ground_removal_enabled=true;reprocess=true;}
          }
          Tip("Geometric groups have no semantic class. This optional overlay is independent of CenterPoint.");
        }
      } else {
        ImGui::TextWrapped("Groups nearby obstacle points. No person or vehicle classification.");
        Metric("Clusters",std::to_string(state.clusters.size()));
      }
      if (!learned || clustering.enabled) {
        ImGui::Spacing();SectionTitle("Clustering");
        ImGui::BeginDisabled(!enabled);
        ImGui::SetNextItemWidth(-1);
        refresh_objects|=ImGui::SliderFloat("##ClusterDistance",&clustering.tolerance_m,.1f,1.5f,"Distance: %.2f m",ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetNextItemWidth(-1);
        refresh_objects|=ImGui::SliderFloat("##ClusterVerticalGap",&clustering.max_vertical_gap_m,.1f,2.5f,"Vertical gap: %.2f m",ImGuiSliderFlags_AlwaysClamp);
        ImGui::EndDisabled();
      }
      end_tab();
    }
    ImGui::EndTabBar();
  }
  ImGui::EndDisabled();
  if (!state.roi_tab_active && !state.roi_tab_requested && !state.roi_gesture.active) state.roi_tool=ViewerState::RoiTool::Navigate;
  if (!state.ui_error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, .56f, .48f, 1));
    ImGui::TextWrapped("%s", state.ui_error.c_str());
    ImGui::PopStyleColor();
    if (ImGui::SmallButton("Dismiss error")) state.ui_error.clear();
  }
  ImGui::EndChild();
  ImGui::Separator();
  const bool pending_exclusions=state.applying_exclusions ||
    !lidar_viewer::SameRoiExclusions(state.roi,state.applied_exclusions);
  if (state.save_requested && pending_exclusions) {
    state.save_requested=false;SetTransientStatus(state,"Apply the edited ROI before saving");
  }
  ImGui::BeginDisabled(pending_exclusions);
  if ((ImGui::Button("Save profile", {-1, 0}) || state.save_requested) && !state.roi_gesture.active) {
    state.save_requested=false;
    try {
      SaveSettings(state);
      SetTransientStatus(state, "Profile saved");
      state.ui_error.clear();
    } catch (const std::exception & error) {
      state.ui_error = std::string("Profile was not saved: ") + error.what();
    }
  }
  ImGui::EndDisabled();
  Tip(pending_exclusions ? "Apply the edited ROI before saving this profile." : "Save the current settings for the next session. Ctrl+S.");
  const bool unsaved=!lidar_viewer::SameRoi(state.roi,state.saved_roi) ||
    !lidar_viewer::SameRoi(state.display_roi,state.saved_display_roi);
  ImGui::TextColored(unsaved ? ImVec4(1,.74f,.4f,1) : ImVec4(.48f,.65f,.65f,1),
    "%s",pending_exclusions ? "Apply ROI before saving" : unsaved ? "ROI changes not saved  /  Ctrl+S" : "Ctrl+S saves this profile");
  ImGui::End();

  // An unobtrusive divider belongs to application layout, not a data profile.
  const auto & side=layout.sidebar;
  ImGui::SetNextWindowPos({side.x+side.w+2*s,side.y});ImGui::SetNextWindowSize({12*s,side.h});
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{0,0});
  ImGui::Begin("##InspectorDivider",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoBackground|
    ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImGui::InvisibleButton("##Resize",{12*s,std::max(1.f,side.h)});TraceUi("inspector_resize");
  const bool resizing=ImGui::IsItemActive();
  if (ImGui::IsItemHovered() || resizing) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddLine({(a.x+b.x)*.5f,a.y+12*s},{(a.x+b.x)*.5f,b.y-12*s},ImGui::GetColorU32(ImGuiCol_SliderGrab),2*s);
    Tip("Drag to resize the inspector. Double-click to reset.");
  }
  if (resizing && ImGui::GetIO().MouseDelta.x!=0) {
    state.inspector_width=std::clamp(state.inspector_width+ImGui::GetIO().MouseDelta.x/s,280.f,520.f);state.app_settings_dirty=true;
  }
  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {state.inspector_width=320;state.app_settings_dirty=true;}
  if (!resizing && state.inspector_resizing && state.app_settings_dirty) SaveAppSettings(state);
  state.inspector_resizing=resizing;ImGui::End();ImGui::PopStyleVar();

  BeginPanel("Transport", layout.transport, false);
  const auto * group = !state.live_mode && has_input ? &state.file_groups[state.current_file_group] : nullptr;
  const std::size_t first_index = group ? group->first_index : 0;
  const std::size_t frame_count = group ? group->last_index - group->first_index + 1 : 0;
  const bool multiple = frame_count > 1;
  ImGui::BeginDisabled(!has_input || state.loading_bag_source.has_value() || (state.live_mode && !state.live_stats.connected));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.15f, .37f, .46f, 1));
  const bool showing_motion = state.live_mode ? !state.live_paused : state.is_playing;
  if (lidar_viewer::ActionButton("##PlayPause",showing_motion ? lidar_viewer::UiIcon::Pause : lidar_viewer::UiIcon::Play,
      showing_motion ? "Pause" : "Play",{88*s,0},s)) {
    TogglePlayback(state);
  }
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::BeginDisabled(!multiple);
  if (lidar_viewer::ActionButton("##Previous",lidar_viewer::UiIcon::Previous,nullptr,{32*s,0},s)) AdvanceFrame(state, -1);
  Tip("Previous frame  [P]");
  ImGui::SameLine();
  if (lidar_viewer::ActionButton("##Next",lidar_viewer::UiIcon::Next,nullptr,{32*s,0},s)) AdvanceFrame(state, 1);
  Tip("Next frame  [N]");
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(state.live_mode);
  ImGui::Checkbox("Loop", &state.loop_playback);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(92.f * s);
  ImGui::DragScalar("##Speed", ImGuiDataType_Double, &state.playback_fps, .1f, nullptr, nullptr, "%.1f fps");
  state.playback_fps = std::isfinite(state.playback_fps) ? std::clamp(state.playback_fps, .1, 120.) : 10.;
  if (layout.transport.w > 800.f * s) {
    ImGui::SameLine();
    if (state.live_mode) ImGui::TextDisabled("Live frame %llu", static_cast<unsigned long long>(state.live_display_sequence));
    else if (group) {
      const auto & first=state.frames[first_index];
      const bool recorded=first.format==InputFormat::Ros2Bag;
      const auto seconds=[&](std::size_t i) {
        return recorded ? static_cast<double>((static_cast<long double>(state.frames[i].bag_timestamp_ns)-first.bag_timestamp_ns)*1e-9L) :
          static_cast<double>(i-first_index)/state.input_fps_estimate.value_or(10.);
      };
      const auto clock=[](double value) {
        value=std::max(0.,value);char label[48];std::snprintf(label,sizeof(label),"%02lld:%04.1f",
          static_cast<long long>(value/60),std::fmod(value,60.));return std::string(label);
      };
      ImGui::TextDisabled("%s / %s",clock(seconds(state.current_index)).c_str(),clock(seconds(group->last_index)).c_str());
      Tip(recorded ? "Position in recorded time. Frame numbers are on the timeline below." :
        "Estimated time at the input scan rate. Frame numbers are on the timeline below.");
    }
  }
  ImGui::EndDisabled();
  ImGui::EndDisabled();
  ImGui::SameLine();
  BuildSourceSelector(state, std::max(80.f * s, ImGui::GetContentRegionAvail().x));
  if (!state.live_mode) {
    ImGui::BeginDisabled(state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active());
    if (ImGui::Button("Add folder",{115*s,0})) state.open_folder_picker=true;
    ImGui::EndDisabled();Tip("Remember a collection of bag recordings for the source list.");
    ImGui::SameLine();
  }
  if (state.live_mode) {
    ImGui::BeginDisabled(!state.live_stats.connected || state.live_stats.record_stopping);
    ImGui::PushStyleColor(ImGuiCol_Button, state.live_stats.record_busy ? ImVec4(.6f,.16f,.16f,1) : ImVec4(.32f,.18f,.2f,1));
    const char * label = state.live_stats.record_stopping ? "Saving..." : (state.live_stats.record_busy ? "Stop REC" : "Record");
    if (ImGui::Button(label, {110.f * s, 0})) {
      try {ToggleLiveRecording(state); state.ui_error.clear();}
      catch (const std::exception & error) {state.ui_error=error.what();}
    }
    ImGui::PopStyleColor(); ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("Rx %llu | IMU %llu | %.1f Hz", static_cast<unsigned long long>(state.live_stats.clouds),
      static_cast<unsigned long long>(state.live_stats.imu_samples), state.live_stats.cloud_hz);
  } else {
  if (!state.timeline_dragging) state.seek_draft = static_cast<int>(state.current_index - first_index);
  ImGui::SetNextItemWidth(-1);
  ImGui::BeginDisabled(!multiple || state.loading_bag_source.has_value());
  const std::string timeline_label = "Frame " + std::to_string(has_input ? state.seek_draft + 1 : 0) + " / " +
    std::to_string(frame_count);
  ImGui::SliderInt("##Timeline", &state.seek_draft, 0, frame_count ? static_cast<int>(frame_count - 1) : 0,
    timeline_label.c_str(), ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
  TraceUi("timeline");
  if (ImGui::IsItemActivated()) {
    state.timeline_dragging = true;
    state.requested_frame.reset();
  }
  if (ImGui::IsItemDeactivated()) {
    if (ImGui::IsItemDeactivatedAfterEdit())
      TryLoadFrame(state, first_index + static_cast<std::size_t>(state.seek_draft));
    state.timeline_dragging = false;
    state.last_frame_advance_time = glfwGetTime();
  }
  ImGui::EndDisabled();
  }
  BuildFolderPicker(state);
  if (state.open_live_dialog) {ImGui::OpenPopup("Live sensors");state.open_live_dialog=false;}
  BuildLiveDialog(state);
  BuildRoiExportDialog(state);
  ImGui::End();

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16.f * s, 4.f * s});
  const double status_now=ImGui::GetTime();
  if (status_now-state.status_sample_time>=1. || state.status_sample_time==0) {
    state.status_sample_time=status_now;state.status_processing_ms=state.processing_ms;
    state.status_roi_ms=state.roi_ms;state.status_fps=state.current_view_fps;
    state.status_points=VisiblePoints(state).count;state.status_live_display=state.live_display_sequence;
    state.status_ground=state.ground_removed_count;
    state.status_live_received=state.live_stats.clouds;
    state.status_source=group ? group->path.filename().u8string() :
      state.current_index<state.frames.size() ? state.frames[state.current_index].path.filename().u8string() : std::string{};
    if (std::getenv("LIDAR_VIEWER_UI_TRACE")) std::cout<<"Status sample "<<state.status_sample_time<<std::endl;
  }
  const bool status_busy=state.requested_frame.has_value() || state.geometry_future.valid();
  if (status_busy && !state.pending_read_started) state.pending_read_started=status_now;
  if (!status_busy) state.pending_read_started.reset();
  const bool long_wait=state.pending_read_started && status_now-*state.pending_read_started>.75;
  BeginPanel("Status", layout.status, false);
  ImGui::PushClipRect(
    {layout.status.x, layout.status.y},
    {layout.status.w > 700.f * s ? layout.status.w - 380.f * s : layout.status.w, layout.status.y + layout.status.h}, true);
  if (!state.ui_error.empty()) ImGui::TextColored({1.f, .56f, .48f, 1}, "%s", state.ui_error.c_str());
  else if (state.loading_bag_source && state.input_load) {
    const auto progress=state.input_load->Snapshot();
    ImGui::TextColored({.44f,.76f,1,1},"%s | %s frames | %s",
      state.input_load->Cancelled() ? "Canceling..." : lidar_viewer::InputLoadStageName(progress.stage),
      CountText(progress.frames).c_str(),LoadElapsed(state).c_str());
  }
  else if (!state.transient_status.empty()) ImGui::TextColored({.44f, .86f, .68f, 1}, "%s", state.transient_status.c_str());
  else if (long_wait) ImGui::TextDisabled("%s",state.requested_frame ? "Reading selected frame..." : "Processing selected frame...");
  else if (state.live_mode) ImGui::TextDisabled("%zu points | display %llu / received %llu | %s", state.status_points,
    static_cast<unsigned long long>(state.status_live_display), static_cast<unsigned long long>(state.status_live_received),
    !state.live_stats.connected ? "disconnected" : (state.live_paused ? "frozen" : (state.live_stats.cloud_age_ms > 1000 ? "stream stale" : "live")));
  else if (!has_input) ImGui::TextDisabled("Open a file, drop a bag, or connect live sensors.");
  else ImGui::TextDisabled("%zu points   |   %s",state.status_points,state.status_source.c_str());
  ImGui::PopClipRect();
  if (layout.status.w > 700.f * s) {
    ImGui::SameLine(std::max(450.f * s, layout.status.w - 360.f * s));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,{0,0});
    if (ImGui::BeginTable("##StatusMetrics",3,ImGuiTableFlags_SizingFixedFit)) {
      ImGui::TableSetupColumn("Process",ImGuiTableColumnFlags_WidthFixed,138*s);
      ImGui::TableSetupColumn("ROI",ImGuiTableColumnFlags_WidthFixed,108*s);
      ImGui::TableSetupColumn("FPS",ImGuiTableColumnFlags_WidthFixed,90*s);
      ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::TextDisabled("Process %.1f ms",state.status_processing_ms);
      ImGui::TableSetColumnIndex(1);ImGui::TextDisabled("ROI %.1f ms",state.status_roi_ms);
      ImGui::TableSetColumnIndex(2);ImGui::TextDisabled("UI %.0f fps",state.status_fps);ImGui::EndTable();
    }
    ImGui::PopStyleVar();
  }
  ImGui::End();
  ImGui::PopStyleVar();

  const auto & view=layout.viewport;
  if (view.w>300*s && view.h>100*s) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{6*s,6*s});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{5*s,0});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{6*s,5*s});
    BeginPanel("Canvas tools",{view.x+12*s,view.y+10*s,std::min(view.w-24*s,87*s),42*s},false);
    const bool top=state.camera.projection_mode==CameraState::ProjectionMode::View2D;
    const auto mode_button=[&](const char * label,bool selected) {
      if (selected) ImGui::PushStyleColor(ImGuiCol_Button,{.13f,.34f,.42f,1});
      const bool clicked=ImGui::Button(label,{32*s,30*s});
      TraceUi((std::string("camera_")+label).c_str());
      if (selected) ImGui::PopStyleColor();
      return clicked;
    };
    if (mode_button("3D",!top)) {SelectRoiTool(state,ViewerState::RoiTool::Navigate);SetProjectionMode(state,CameraState::ProjectionMode::View3D);}
    ImGui::SameLine();
    if (mode_button("2D",top)) SetProjectionMode(state,CameraState::ProjectionMode::View2D);
    ImGui::End();ImGui::PopStyleVar(3);
  }
  auto * draw=ImGui::GetBackgroundDrawList();
  const ImU32 muted = IM_COL32(131, 151, 174, 255);
  const auto handles=RoiHandlePoints(state);
  if (!handles.empty() && !state.loading_bag_source) {
    draw->PushClipRect({view.x,view.y},{view.x+view.w,view.y+view.h},true);
    const ImU32 accent=EditedRoi(state).boxes[SelectedRoi(state)].mode==lidar_viewer::RoiBoxMode::Exclude ?
      IM_COL32(255,163,115,255) : IM_COL32(110,211,224,255);
    for (int i=0;i<4;++i) draw->AddLine(handles[i*2].screen,handles[(i*2+2)%8].screen,accent,1.5f*s);
    draw->AddLine(handles[5].screen,handles.back().screen,accent,1.5f*s);
    for (const auto & h : handles) {
      const auto p=h.screen;
      const bool hover=std::hypot(p.x-state.camera.last_x,p.y-state.camera.last_y)<11*s;
      if (h.rotate) {draw->AddCircleFilled(p,7*s,IM_COL32(20,29,40,255));draw->AddCircle(p,7*s,accent,20,2*s);}
      else {draw->AddRectFilled({p.x-4*s,p.y-4*s},{p.x+4*s,p.y+4*s},hover ? IM_COL32(255,255,255,255) : accent,1*s);}
      if (hover && !ImGui::GetIO().WantCaptureMouse) ImGui::SetMouseCursor(h.rotate ? ImGuiMouseCursor_Hand : ImGuiMouseCursor_ResizeAll);
    }
    draw->PopClipRect();
  }
  if (has_input && VisiblePoints(state).count==0 && !state.base_frame.points.empty() &&
      !state.loading_bag_source && state.roi_tool==ViewerState::RoiTool::Navigate &&
      !ImGui::IsPopupOpen(nullptr,ImGuiPopupFlags_AnyPopup) && view.w>300*s && view.h>180*s) {
    const float width=std::min(440*s,view.w-32*s);
    BeginPanel("Empty region",{view.x+(view.w-width)*.5f,view.y+(view.h-156*s)*.5f,width,156*s},false);
    const bool cropped=state.display_roi.enabled && state.frame.points.empty();
    if (cropped) {
      ImGui::TextUnformatted("No points inside the display regions");
      ImGui::TextWrapped("Adjust the display boxes or show the full cloud.");
      if (ImGui::Button("Show full cloud")) {state.display_roi.enabled=false;state.display_roi_pending=true;}
      TraceUi("display_roi_show_full");
    } else {
      const bool all_skipped=state.exclusion_removed_count==state.base_frame.points.size();
      ImGui::TextUnformatted(state.ground_view==ViewerState::GroundView::Ground ?
        "No ground points to show" : "No obstacle points to show");
      ImGui::TextWrapped("%s",all_skipped ? "The applied ROI skips all points during ground detection." :
        "Check ground settings or show both classes.");
      if (ImGui::Button("Show all points")) state.ground_view=ViewerState::GroundView::All;
    }
    ImGui::End();
  }
  BuildInputLoading(state);
  BuildAppSettings(state);
  if(state.open_training && !state.training_workspace) state.training_workspace=std::make_unique<lidar_viewer::TrainingWorkspace>();
  if(state.training_workspace) state.training_workspace->Draw(state.open_training,CurrentDatasetRoot(state),s);
  if (has_input && state.imu_level_enabled) {
    const bool applied = state.imu_level_result.applied;
    const bool estimated = state.base_frame.imu &&
      state.base_frame.imu->source == lidar_viewer::ImuGravitySource::AccelGyro;
    draw->AddText({view.x + 14.f * s, view.y + 86.f * s},
      applied ? IM_COL32(105, 215, 167, 255) : IM_COL32(242, 179, 96, 255),
      applied ? (estimated ? "IMU level: estimated" : "IMU level: active") : "IMU level: unavailable");
  }
  if (state.live_mode && state.live_stats.record_busy) {
    draw->AddText({view.x + view.w - 160.f*s, view.y + 12.f*s}, IM_COL32(255,100,100,255),
      state.live_stats.record_stopping ? "Saving bag..." : "REC  raw sensors");
  }
  if (state.live_mode && (!state.live_stats.record_error.empty() || state.live_stats.record_dropped))
    draw->AddText({view.x + 14.f*s, view.y + view.h - 48.f*s}, IM_COL32(255,120,100,255), "Recording error - open Live for details");
  if (has_input && state.ground_removal_enabled && state.ground_view!=ViewerState::GroundView::Obstacles) {
    draw->AddCircleFilled({view.x + 18.f * s, view.y + view.h - 18.f * s}, 4.f * s, IM_COL32(255, 75, 65, 255));
    draw->AddText({view.x + 30.f * s, view.y + view.h - 26.f * s}, muted, "Ground");
  }
  if (state.roi_tool==ViewerState::RoiTool::Draw) {
    draw->AddText({view.x+14*s,view.y+view.h-28*s},IM_COL32(125,225,230,255),"Drag a new box  |  Esc cancels");
  }
  if (state.roi_gesture.active && !EditedRoi(state).boxes.empty()) {
    const auto & box=EditedRoi(state).boxes[SelectedRoi(state)];
    char label[160];std::snprintf(label,sizeof(label),"%.2f x %.2f x %.2f m  |  %s",box.size.x,box.size.y,box.size.z,
      box.mode==lidar_viewer::RoiBoxMode::Exclude ? "release to finish, then Apply" : "release to apply");
    draw->AddText({view.x+14*s,view.y+view.h-52*s},IM_COL32(255,220,115,255),label);
  }
  {
    auto bounded = [&](float & value, float low, float high, float fallback) {
      if (!std::isfinite(value)) {value = fallback; state.ui_error = "A non-finite value was rejected.";}
      value = std::clamp(value, low, high);
    };
    bounded(state.calibration.yaw_deg, -180, 180, 0);
    bounded(state.calibration.pitch_deg, -180, 180, 0);
    bounded(state.calibration.roll_deg, -180, 180, 0);
    bounded(state.calibration.z_m, -5, 5, 0);
    auto & ground = state.algorithm_config.ground_removal;
    bounded(ground.gseg3d_cell_size_x_m, .1f, 5, 1);
    bounded(ground.gseg3d_cell_size_y_m, .1f, 5, 1);
    bounded(ground.gseg3d_cuda_cell_size_m, .1f, 5, .5f);
    bounded(ground.gseg3d_ground_inlier_threshold_m, .02f, 1, .2f);
    bounded(ground.gseg3d_ground_height_m,-5,5,0);
    bounded(ground.gseg3d_max_ground_height_deviation_m,.01f,1,.3f);
    bounded(ground.patchwork_distance_threshold_m,.02f,.5f,.18f);
    bounded(ground.patchwork_seed_threshold_m,.02f,.5f,.25f);
    bounded(ground.max_ground_angle_deg,1,45,15);
    bounded(ground.final_distance_threshold_m,.02f,.5f,.2f);
    bounded(ground.candidate_grid_cell_m, .1f, 5, .8f);
    bounded(state.camera.point_size, 1, 8, 2);
    bounded(state.grid_settings.spacing, .1f, 50, 2);
    bounded(state.grid_settings.thickness, 1, 6, 1);
    bounded(state.ui_scale, .75f, 1.5f, 1);
    bounded(state.tracking_params.max_missed_s,.1f,2.f,.8f);
    state.tracking_params.confirmation_hits=std::clamp(state.tracking_params.confirmation_hits,1,10);
  }
  if (state.loading_bag_source) return;
  if (state.roi_pending) {RebuildRoiGeometry(state);state.roi_pending=false;}
  refresh_objects|=std::exchange(state.display_roi_pending,false);
  const bool exclusions_changed=std::exchange(state.apply_exclusions_requested,false);
  if (exclusions_changed) {
    state.analysis_roi=state.roi;state.applying_exclusions=true;reprocess=true;
    SetTransientStatus(state,"Applying ground ROI...");
  }
  const bool apply_to_next_frame=(state.is_playing && state.frames.size()>1) ||
    (state.live_mode && !state.live_paused && state.live_stats.connected && state.live_stats.cloud_age_ms<200);
  if (reprocess || refresh_objects || refresh_detector) ++state.geometry_config;
  if (reprocess) ++state.geometry_ground_config;
  if (alignment_changed) ++state.geometry_input_config;
  if (reprocess) ++state.ground_sequence;
  if (reprocess && (!apply_to_next_frame || exclusions_changed)) {
    try {
      if (state.async_geometry) QueueGeometry(state,state.base_frame,{},{},alignment_changed || refresh_detector);
      else {
        const auto previous_exclusions=state.applied_exclusions;
        state.applied_exclusions=state.analysis_roi;
        try {ApplyCalibration(state,nullptr,std::nullopt,alignment_changed || refresh_detector);}
        catch (...) {state.applied_exclusions=previous_exclusions;throw;}
        if (state.applying_exclusions) {
          state.applying_exclusions=false;SetTransientStatus(state,"ROI applied to ground segmentation");
        }
      }
      state.ui_error.clear();
    }
    catch (const std::exception & error) {
      if (state.applying_exclusions) {
        state.analysis_roi=state.applied_exclusions;state.applying_exclusions=false;
      }
      state.is_playing=false;state.ui_error=error.what();
    }
  } else if (!reprocess && !state.applying_exclusions && refresh_objects && !state.roi_gesture.active) {
    try {
      if (state.async_geometry) QueueGeometry(state,state.base_frame,{},{},false,true);
      else {
        ApplyRoiResult(state,state.processed_frame,state.processed_obstacles,state.roi_frame);
        std::cout<<"ROI applied: "<<state.roi.boxes.size()<<" boxes | "<<state.frame.points.size()<<" points | "
          <<state.roi_ms<<" ms | ground passes "<<state.ground_passes<<std::endl;
      }
      state.ui_error.clear();
    } catch (const std::exception & error) {state.roi_pending=false;state.ui_error=error.what();}
  }
  if (refresh_detector && !reprocess && !state.geometry_future.valid()) {
    try {
      if (!state.detector_input && state.async_geometry) QueueGeometry(state,state.base_frame,{},{},true,true);
      else RefreshDetectorInput(state);
    } catch (const std::exception & error) {state.ui_error=error.what();}
  }
}

void RenderRoiBoxes(ViewerState & state, const Mat4 & mvp)
{
  auto & renderer = state.roi_renderer;
  if (!EditedRoi(state).show_boxes || renderer.vertices.empty()) return;
  glUseProgram(state.grid.program);
  glUniformMatrix4fv(state.grid.mvp_loc, 1, GL_FALSE, mvp.m.data());
  glBindVertexArray(renderer.vao);
  if (renderer.dirty) {
    glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(renderer.vertices.size() * sizeof(Vec3)),
      renderer.vertices.data(), GL_DYNAMIC_DRAW);
    renderer.dirty = false;
  }
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glLineWidth(2.f);
  for (std::size_t i = 0; i < std::min(EditedRoi(state).boxes.size(), renderer.vertices.size() / 24); ++i) {
    if(!EditedRoi(state).boxes[i].enabled) continue;
    const bool display=state.roi_purpose==ViewerState::RoiPurpose::Display;
    if (i == SelectedRoi(state)) glUniform4f(state.grid.color_loc, display ? .43f : 1.f, display ? .83f : .52f, display ? .88f : .28f, 1.f);
    else glUniform4f(state.grid.color_loc, display ? .35f : 1.f, display ? .75f : .48f, display ? .85f : .3f, .9f);
    glDrawArrays(GL_LINES, static_cast<GLint>(i * 24), 24);
  }
  glLineWidth(1.f);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glBindVertexArray(0);
}

void RenderGrid(const ViewerState & state, const Mat4 & mvp)
{
  if (!state.grid_settings.enabled || state.grid.vertices.empty()) {
    return;
  }

  glUseProgram(state.grid.program);
  glUniformMatrix4fv(state.grid.mvp_loc, 1, GL_FALSE, mvp.m.data());
  glUniform4f(state.grid.color_loc, 0.42f, 0.48f, 0.56f, 0.38f);
  glBindVertexArray(state.grid.vao);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glLineWidth(state.grid_settings.thickness);
  glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(state.grid.vertices.size()));
  glLineWidth(1.0f);
  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);
}

void UploadClusterBoxes(ViewerState & state)
{
  std::vector<Vec3> fills,lines;
  for (auto & batch:state.cluster_boxes.batches) {
    batch.fill_first=fills.size();batch.line_first=lines.size();
    fills.insert(fills.end(),batch.fill_vertices.begin(),batch.fill_vertices.end());
    lines.insert(lines.end(),batch.line_vertices.begin(),batch.line_vertices.end());
  }
  glBindBuffer(GL_ARRAY_BUFFER,state.cluster_boxes.fill_vbo);
  glBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(fills.size()*sizeof(Vec3)),fills.data(),GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER,state.cluster_boxes.line_vbo);
  glBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(lines.size()*sizeof(Vec3)),lines.data(),GL_DYNAMIC_DRAW);
  state.cluster_boxes.dirty=false;
}

void RenderClusterBoxes(ViewerState & state,const Mat4 & mvp)
{
  if (state.ground_view==ViewerState::GroundView::Ground ||
      (!state.algorithm_config.clustering.enabled && !state.algorithm_config.detection.enabled) || state.cluster_boxes.batches.empty()) return;
  if (state.cluster_boxes.dirty) UploadClusterBoxes(state);
  glUseProgram(state.grid.program);glUniformMatrix4fv(state.grid.mvp_loc,1,GL_FALSE,mvp.m.data());
  glBindVertexArray(state.cluster_boxes.vao);glEnable(GL_DEPTH_TEST);glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  for (const auto & batch:state.cluster_boxes.batches) {
    if (!batch.fill_vertices.empty()) {
      glDepthMask(GL_FALSE);glBindBuffer(GL_ARRAY_BUFFER,state.cluster_boxes.fill_vbo);
      glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vec3),reinterpret_cast<void *>(0));
      glUniform4f(state.grid.color_loc,batch.fill_color.r,batch.fill_color.g,batch.fill_color.b,batch.fill_color.a);
      glDrawArrays(GL_TRIANGLES,static_cast<GLint>(batch.fill_first),static_cast<GLsizei>(batch.fill_vertices.size()));
    }
    if (!batch.line_vertices.empty()) {
      glBindBuffer(GL_ARRAY_BUFFER,state.cluster_boxes.line_vbo);
      glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vec3),reinterpret_cast<void *>(0));
      glUniform4f(state.grid.color_loc,batch.line_color.r,batch.line_color.g,batch.line_color.b,batch.line_color.a);
      glLineWidth(2.f);glDrawArrays(GL_LINES,static_cast<GLint>(batch.line_first),static_cast<GLsizei>(batch.line_vertices.size()));
    }
  }
  glDepthMask(GL_TRUE);glLineWidth(1.f);glDisable(GL_BLEND);
}

void RenderDetectionLabels(const ViewerState & state,const Mat4 & mvp)
{
  if (!state.algorithm_config.detection.enabled || state.ground_view==ViewerState::GroundView::Ground) return;
  const auto & canvas=state.ui_layout.viewport;
  auto * draw=ImGui::GetForegroundDrawList();
  draw->PushClipRect({canvas.x,canvas.y},{canvas.x+canvas.w,canvas.y+canvas.h},true);
  std::vector<ImVec4> labels;
  for (std::size_t index=0;index<state.detections.size();++index) {
    const auto & detection=state.detections[index];
    const auto * track=index<state.visible_tracks.size() && state.visible_tracks[index].id ? &state.visible_tracks[index] : nullptr;
    if (track && state.tracking_params.show_trails && track->trail.size()>1) {
      std::optional<ImVec2> previous;
      for (const auto & point:track->trail) {
        const auto & matrix=mvp.m;
        const float w=matrix[3]*point.x+matrix[7]*point.y+matrix[11]*point.z+matrix[15];
        const float depth=(matrix[2]*point.x+matrix[6]*point.y+matrix[10]*point.z+matrix[14])/std::max(w,1e-8f);
        if (w<=1e-5f || depth < -1 || depth>1) {previous.reset();continue;}
        const ImVec2 p{canvas.x+((matrix[0]*point.x+matrix[4]*point.y+matrix[8]*point.z+matrix[12])/w+1)*.5f*canvas.w,
          canvas.y+(1-(matrix[1]*point.x+matrix[5]*point.y+matrix[9]*point.z+matrix[13])/w)*.5f*canvas.h};
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || std::abs(p.x)>1e6f || std::abs(p.y)>1e6f) {previous.reset();continue;}
        if (previous) {const auto color=DetectionClassColors(detection.class_id).second;
          draw->AddLine(*previous,p,ImGui::ColorConvertFloat4ToU32({color.r,color.g,color.b,.65f}),1.5f);}
        previous=p;
      }
    }
    const auto & b=detection.oriented_bounds;const auto & m=mvp.m;
    float left=std::numeric_limits<float>::max(),right=-left,top=left,bottom=-left;
    for(const float dx:{-1.f,1.f}) for(const float dy:{-1.f,1.f}) for(const float dz:{-1.f,1.f}) {
      const Vec3 p=b.valid ? b.center+b.axis_x*(dx*b.half_length)+b.axis_y*(dy*b.half_width)+
        Vec3{0,0,(dz<0 ? b.min_z : b.max_z)-b.center.z} :
        Vec3{dx<0 ? detection.bounds.min.x : detection.bounds.max.x,
          dy<0 ? detection.bounds.min.y : detection.bounds.max.y,dz<0 ? detection.bounds.min.z : detection.bounds.max.z};
      const float w=m[3]*p.x+m[7]*p.y+m[11]*p.z+m[15];if(w<=1e-5f) continue;
      const float z=(m[2]*p.x+m[6]*p.y+m[10]*p.z+m[14])/w;if(z < -1 || z>1) continue;
      const float x=canvas.x+((m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12])/w+1)*.5f*canvas.w;
      const float y=canvas.y+(1-(m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13])/w)*.5f*canvas.h;
      left=std::min(left,x);right=std::max(right,x);top=std::min(top,y);bottom=std::max(bottom,y);
    }
    if(left>right) continue;
    char text[160];
    if (track) {
      if (track->predicted) std::snprintf(text,sizeof(text),"%s #%llu predicted",lidar_viewer::algorithms::DetectionClassName(detection.class_id),static_cast<unsigned long long>(track->id));
      else std::snprintf(text,sizeof(text),"%s #%llu %.2f%s",lidar_viewer::algorithms::DetectionClassName(detection.class_id),
        static_cast<unsigned long long>(track->id),detection.confidence,track->confirmed ? "" : " new");
      const auto length=std::strlen(text);
      if (state.tracking_params.show_speed && track->confirmed)
        std::snprintf(text+length,sizeof(text)-length," | %.1f m/s",std::hypot(track->velocity.x,track->velocity.y));
    } else std::snprintf(text,sizeof(text),"%s %.2f",lidar_viewer::algorithms::DetectionClassName(detection.class_id),detection.confidence);
    const auto size=ImGui::CalcTextSize(text);
    // In a top view, projecting the box's 3D top-center still lands in its
    // middle. Anchor outside the complete projected silhouette instead.
    if(size.x+8>canvas.w) continue;
    const ImVec2 a{std::clamp((left+right-size.x)*.5f,canvas.x+4,canvas.x+canvas.w-size.x-4),
      top-size.y-8>=canvas.y+2 ? top-size.y-8 : bottom+8};
    const ImVec4 rectangle{a.x-4,a.y-2,a.x+size.x+4,a.y+size.y+2};
    if (rectangle.x<canvas.x || rectangle.z>canvas.x+canvas.w || rectangle.y<canvas.y || rectangle.w>canvas.y+canvas.h) continue;
    if (std::any_of(labels.begin(),labels.end(),[&](const ImVec4 & other) {
        return rectangle.x<other.z+3 && rectangle.z>other.x-3 && rectangle.y<other.w+3 && rectangle.w>other.y-3;
      })) continue;
    labels.push_back(rectangle);
    const auto color=DetectionClassColors(detection.class_id).second;
    draw->AddRectFilled({a.x-4,a.y-2},{a.x+size.x+4,a.y+size.y+2},IM_COL32(12,17,24,210),3);
    draw->AddText(a,ImGui::ColorConvertFloat4ToU32({color.r,color.g,color.b,1}),text);
  }
  draw->PopClipRect();
}

void RenderAxes(ViewerState & state, const Mat4 & mvp)
{
  const float length=std::max(8.f,std::min(state.frame.radius*.45f,30.f));
  glUseProgram(state.grid.program);
  glUniformMatrix4fv(state.grid.mvp_loc,1,GL_FALSE,mvp.m.data());
  glBindVertexArray(state.axes_vao);
  if (length!=state.axes_length) {
    const std::array<Vec3,6> vertices{{{0,0,0},{length,0,0},{0,0,0},{0,length,0},{0,0,0},{0,0,length}}};
    glBindBuffer(GL_ARRAY_BUFFER,state.axes_vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices.data(),GL_STATIC_DRAW);
    state.axes_length=length;
  }
  glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);glDisable(GL_BLEND);glLineWidth(3.f);
  const std::array<Color,3> colors{{{.94f,.29f,.26f,1},{.21f,.82f,.46f,1},{.3f,.62f,.98f,1}}};
  for (std::size_t i=0;i<colors.size();++i) {
    const auto c=colors[i];glUniform4f(state.grid.color_loc,c.r,c.g,c.b,c.a);
    glDrawArrays(GL_LINES,static_cast<GLint>(i*2),2);
  }
  glLineWidth(1.f);glBindVertexArray(0);
}

}  // namespace

int RunViewer(int argc, char ** argv)
{
  try {
    fs::path input_path = lidar_viewer::ProjectResourcePath(kDefaultInputPath);
    std::string bag_topic = "/b2/points";
    std::string imu_topic;
    bool input_selected = false;
    bool inspect = false;
    bool detect_objects = false;
    bool autoplay = false;
    bool imu_level = false;
    bool start_2d = false;
    bool start_live = false;
    fs::path record_folder = lidar_viewer::ProjectResourcePath("recordings");
#ifdef LIDAR_VIEWER_USER_SETTINGS
    record_folder=lidar_viewer::UserHome()/"Documents/LiDARViewer/recordings";
#endif
    std::optional<float> sensor_height;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: lidar_studio [INPUT] [--live] [--topic /b2/points] [--sensor-height METRES] [--imu-level] [--imu-topic TOPIC] [--2d] [--play] [--detect] [--inspect] [--record-folder DIR]\n"
                  << "INPUT: .bin/.npy/.pcd file, dataset folder/metadata.json, bag collection, ROS 2 bag, .mcap, .db3 or metadata.yaml.\n"
                  << "Without INPUT, show an empty viewer. Click Open file to select a recording.\n"
                  << "Open file (Ctrl+O) selects another recording; Add folder remembers bag collections.\n"
                  << "--live: subscribe to live LiDAR/IMU. Pause freezes the display; Play returns to the latest frame.\n"
                  << "--record-folder: parent folder for recordings started explicitly with Record.\n"
                  << "--sensor-height: explicit display Z translation (0..5 m); no automatic bag leveling.\n"
                  << "--imu-level: gravity-align roll/pitch using recorded IMU and a known mount rotation.\n"
                  << "--imu-topic: select an IMU topic; otherwise choose one from recorded topics.\n"
                  << "--2d: open top view; right-drag or Q/E rotates, left-drag pans.\n"
                  << "--detect: enable the installed original CenterPoint model.\n"
                  << "--inspect: index and load the first frame without opening a window.\n"
                  << "ROS 2 bag backend: " << (RosbagInputAvailable() ? "enabled" : "not built") << '\n';
        return 0;
      } else if (arg == "--topic" || arg == "--sensor-height" || arg == "--imu-topic" || arg == "--record-folder") {
        if (++i >= argc) {
          throw std::runtime_error("Missing value for " + arg);
        }
        if (arg == "--record-folder") {
          record_folder=argv[i];
          if (record_folder.empty()) throw std::runtime_error("Recording folder must not be empty");
        } else if (arg == "--imu-topic") {
          imu_topic = argv[i];
          if (imu_topic.empty()) throw std::runtime_error("IMU topic must not be empty");
        } else if (arg == "--topic") {
          bag_topic = argv[i];
          if (bag_topic.empty()) {
            throw std::runtime_error("Bag topic must not be empty");
          }
        } else {
          std::size_t consumed = 0;
          const std::string value = argv[i];
          const float height = std::stof(value, &consumed);
          if (consumed != value.size() || !std::isfinite(height) || height < 0 || height > 5) {
            throw std::runtime_error("Sensor height must be a finite value in 0..5 metres");
          }
          sensor_height = height;
        }
      } else if (arg == "--inspect") {
        inspect = true;
      } else if (arg == "--detect") {
        detect_objects=true;
      } else if (arg == "--play") {
        autoplay = true;
      } else if (arg == "--imu-level") {
        imu_level = true;
      } else if (arg == "--2d") {
        start_2d = true;
      } else if (arg == "--live") {
        start_live = true;
      } else if (!arg.empty() && arg.front() == '-') {
        throw std::runtime_error("Unknown option: " + arg);
      } else if (input_selected) {
        throw std::runtime_error("Only one input path is supported");
      } else {
        input_path = arg;
        input_selected = true;
      }
    }

    if (start_live && (input_selected || inspect)) throw std::runtime_error("Use --live without INPUT or --inspect");
    ViewerState state;
    const auto copy_text = [](const std::string & value, auto & buffer) {
      if (value.size() >= buffer.size()) throw std::runtime_error("Live setting is too long");
      std::copy(value.begin(), value.end(), buffer.begin());
    };
    copy_text(bag_topic, state.live_points_topic);
    copy_text(imu_topic.empty() ? "/vn100/imu/data_ned" : imu_topic, state.live_imu_topic);
    copy_text(record_folder.string(), state.live_record_folder);
    state.bag_topic = bag_topic;
    state.bag_imu_topic = imu_topic;
    state.imu_level_cli_override = imu_level;
    state.detector_cli_override=detect_objects;
    state.bag_sensor_height_override = sensor_height;
    try {
      state.data_folders_config = DataFoldersConfigPath();
      state.hidden_sources_config=lidar_viewer::UserConfigurationDirectory()/"hidden_recordings.cfg";
      state.hidden_sources=LoadHiddenBagSources(state.hidden_sources_config);
      state.data_folders = LoadBagFolders(state.data_folders_config);
      for (const auto & folder : state.data_folders)
        state.bag_sources = FilterBagSources(MergeBagSources(std::move(state.bag_sources), DiscoverBagSources(folder)),state.hidden_sources);
    } catch (const std::exception & error) {
      state.ui_error = std::string("Saved data folders could not be loaded: ") + error.what();
    }
    const bool load_initial_input = input_selected || inspect;
    if (input_selected) {
      const auto supplied = DiscoverBagSources(input_path);
      if (!supplied.empty()) {
        input_path = supplied.front().path;
        UnhideBagSource(state,input_path);
        state.bag_sources = MergeBagSources(std::move(state.bag_sources), supplied);
      }
    } else if (inspect && !state.bag_sources.empty()) {
      input_path = state.bag_sources.front().path;
    }
    if (load_initial_input && IsRos2BagInput(input_path)) {
      const auto canonical = fs::canonical(input_path);
      for (std::size_t i = 0; i < state.bag_sources.size(); ++i)
        if (state.bag_sources[i].path == canonical) state.current_bag_source = i;
    }
    state.dataset_profile = !state.frames.empty() && state.frames.front().dataset_sample ? "dataset" :
      load_initial_input ? DetectDatasetProfile(input_path) : "rosbag";
    state.algorithm_config = lidar_viewer::LoadPipelineConfig(AlgorithmsConfigPath());
    LoadSettings(state);
    LoadAppSettings(state);
    if (detect_objects) state.algorithm_config.detection.enabled=true;
    if (imu_level) state.imu_level_enabled = true;
    if (start_2d) SetProjectionMode(state, CameraState::ProjectionMode::View2D);
    if (sensor_height) {
      state.calibration.z_m = *sensor_height;
    }
    const bool async_initial_bag=load_initial_input && !inspect && IsRos2BagInput(input_path);
    if (inspect && !input_selected && !state.bag_sources.empty()) {
      // A disconnected or damaged recording must not hide other registered bags.
      for (std::size_t i = 0; i < state.bag_sources.size(); ++i) {
        try {
          auto frames = ResolveInputFrames(state.bag_sources[i].path, bag_topic, imu_topic);
          if (frames.empty()) continue;
          auto first = LoadFrame(frames.front());
          state.frames = std::move(frames);
          StoreFrameInCache(state, 0, std::move(first));
          input_path = state.bag_sources[i].path;
          state.current_bag_source = i;
          break;
        } catch (const std::exception & error) {
          std::cerr << "Skipping unavailable bag " << state.bag_sources[i].path << ": " << error.what() << '\n';
        }
      }
      if (state.frames.empty()) throw std::runtime_error("No registered bag can be opened. Run with a different bag or collection folder.");
    } else if (load_initial_input && !async_initial_bag) {
      state.frames = ResolveInputFrames(input_path, bag_topic, imu_topic);
      state.dataset_tab_requested=!state.frames.empty() && state.frames.front().dataset_sample!=nullptr;
    }
    state.input_fps_estimate = EstimateInputFps(state);
    SyncPlaybackFpsToInput(state);
    BuildFileGroups(state);
    if (!state.frames.empty()) LoadFrameByIndex(state, 0);
    state.last_frame_advance_time = 0.0;
    state.is_playing = autoplay && !state.frames.empty();
    if (state.frames.empty() && !async_initial_bag) {
      std::cout << "No bag loaded. Click Open file to select a recording." << std::endl;
    } else if (!state.frames.empty()) {
      std::cout << "Input: " << input_path << " | " << state.frames.size()
                << " frames | first frame " << state.base_frame.points.size()
                << " points | profile " << state.dataset_profile;
      if (state.input_fps_estimate) {
        std::cout << " | recorded median " << *state.input_fps_estimate << " Hz";
      }
      std::cout << std::endl;
    }
    if (!state.bag_sources.empty()) {
      std::cout << "Bag list: " << state.bag_sources.size() << " recordings" << std::endl;
      for (const auto & source : state.bag_sources) std::cout << "  " << BagLabel(state,source) << '\n';
    }
    if (inspect) {
      std::cout << "ROI: " << (state.roi.enabled ? "on" : "off") << " | " << state.roi.boxes.size()
                << " boxes | retained " << state.frame.points.size() << " / " << state.base_frame.points.size() << " points\n";
      std::cout << "Display ROI: " << (state.display_roi.enabled ? "on" : "off") << " | "
                << state.display_roi.boxes.size() << " boxes | retained " << state.frame.points.size()
                << " / " << state.processed_frame.points.size() << " points\n";
      std::cout << "Display: " << GroundViewKey(state.ground_view) << " | visible " << VisiblePoints(state).count
                << " | ground " << state.ground_removed_count << " | obstacles "
                << state.frame.points.size()-std::min(state.frame.points.size(),state.ground_removed_count)
                << " | ground passes " << state.ground_passes << '\n';
      std::cout << "Ground ROI: skipped " << state.exclusion_removed_count << " | input "
                << state.base_frame.points.size()-state.exclusion_removed_count << " | detector input "
                << (state.detector_input ? state.detector_input->size() : 0) << " points\n";
      if (state.imu_level_enabled) {
        std::cout << "IMU leveling: " << state.imu_level_result.status << '\n';
        std::cout << "IMU correction: roll "<<state.imu_level_result.roll_deg<<" pitch "<<state.imu_level_result.pitch_deg
          <<" | automatic axes "<<state.imu_level_result.axes_corrected<<'\n';
      }
      if (detect_objects && state.detector_input && !state.detector_input->empty()) {
        const double deadline=DetectorClock()+90;
        while (state.detector_completed!=state.detector_revision) {
          UpdateDetector(state);
          const auto status=state.detector ? state.detector->Status() : lidar_viewer::DetectorStatus{};
          if (!status.configured || status.failed) throw std::runtime_error(status.message);
          if (DetectorClock()>deadline) throw std::runtime_error("Detector timed out");
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!state.detector_error.empty()) throw std::runtime_error(state.detector_error);
        std::cout<<"Detector: "<<state.algorithm_config.detection.model<<" | "<<state.detector->Status().provider<<" | "<<state.detector_latency_ms<<" ms | "<<state.detections.size()<<" boxes\n";
        for (const auto & detection:state.detections) {
          const auto & b=detection.oriented_bounds;
          std::cout<<"  "<<lidar_viewer::algorithms::DetectionClassName(detection.class_id)<<" score "<<detection.confidence<<
            " center "<<b.center.x<<' '<<b.center.y<<' '<<b.center.z<<" yaw "<<std::atan2(b.axis_x.y,b.axis_x.x)<<'\n';
        }
      }
      if (detect_objects && (!state.detector_input || state.detector_input->empty())) std::cout<<"Detector: no valid LiDAR returns | 0 boxes\n";
      return 0;
    }

    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

#if defined(GLFW_X11_CLASS_NAME) && !defined(_WIN32) && !defined(__APPLE__)
    glfwWindowHintString(GLFW_X11_CLASS_NAME,"lidar-viewer");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME,"lidar-viewer");
#endif
#if defined(GLFW_WAYLAND_APP_ID) && !defined(_WIN32) && !defined(__APPLE__)
    glfwWindowHintString(GLFW_WAYLAND_APP_ID,"lidar-viewer");
#endif
    GLFWwindow * window = glfwCreateWindow(kWindowWidth, kWindowHeight, "LiDAR Studio", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }

#ifndef __APPLE__
    // Native macOS icons come from the application bundle's ICNS resource.
    // GLFW copies these pixels, so local arrays can be released immediately.
    auto icon32=lidar_viewer::app_icon::Decode<32*32>(lidar_viewer::app_icon::rgba32);
    auto icon128=lidar_viewer::app_icon::Decode<128*128>(lidar_viewer::app_icon::rgba128);
    GLFWimage icons[]={{32,32,icon32.data()},{128,128,icon128.data()}};
    glfwSetWindowIcon(window,2,icons);
#endif
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowSizeLimits(window, 800, 600, GLFW_DONT_CARE, GLFW_DONT_CARE);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
      glfwDestroyWindow(window);
      glfwTerminate();
      throw std::runtime_error("Failed to initialize GLEW");
    }

    glfwSetWindowUserPointer(window, &state);
    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    glfwSetCursorPosCallback(window, CursorCallback);
    glfwSetScrollCallback(window, ScrollCallback);
    glfwSetKeyCallback(window, KeyCallback);
    glfwSetDropCallback(window,DropCallback);
    ConfigureInterface(window, state);
    glfwGetCursorPos(window,&state.camera.last_x,&state.camera.last_y);
    const ImGuiStyle base_ui_style = ImGui::GetStyle();

    glGenVertexArrays(1, &state.vao);
    glGenBuffers(1, &state.vbo);

    glBindVertexArray(state.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(KittiPoint), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(KittiPoint), reinterpret_cast<void *>(offsetof(KittiPoint, intensity)));


    state.grid.program = CreateGridProgram();
    state.grid.mvp_loc = glGetUniformLocation(state.grid.program, "u_mvp");
    state.grid.color_loc = glGetUniformLocation(state.grid.program, "u_color");
    glGenVertexArrays(1, &state.grid.vao);
    glGenBuffers(1, &state.grid.vbo);
    glBindVertexArray(state.grid.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.grid.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), reinterpret_cast<void *>(0));

    glGenVertexArrays(1, &state.cluster_boxes.vao);
    glGenBuffers(1, &state.cluster_boxes.fill_vbo);
    glGenBuffers(1, &state.cluster_boxes.line_vbo);
    glBindVertexArray(state.cluster_boxes.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.cluster_boxes.fill_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), reinterpret_cast<void *>(0));

    glGenVertexArrays(1, &state.roi_renderer.vao);
    glGenBuffers(1, &state.roi_renderer.vbo);
    glBindVertexArray(state.roi_renderer.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.roi_renderer.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), reinterpret_cast<void *>(0));

    glGenVertexArrays(1,&state.axes_vao);glGenBuffers(1,&state.axes_vbo);
    glBindVertexArray(state.axes_vao);glBindBuffer(GL_ARRAY_BUFFER,state.axes_vbo);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vec3),reinterpret_cast<void *>(0));

    UploadFrame(state);
    if (!state.camera.has_saved_view) {
      ResetCameraToFrame(state);
    }
    RebuildGrid(state);

    const GLuint program = CreateProgram();
    const GLint mvp_loc = glGetUniformLocation(program, "u_mvp");
    const GLint point_size_loc = glGetUniformLocation(program, "u_point_size");
    const GLint color_mode_loc = glGetUniformLocation(program, "u_color_mode");
    const GLint min_height_loc = glGetUniformLocation(program, "u_min_height");
    const GLint max_height_loc = glGetUniformLocation(program, "u_max_height");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.04f, 0.05f, 0.07f, 1.0f);

    state.fps_measurement_started_at = glfwGetTime();
    state.playback_measurement_started_at = state.fps_measurement_started_at;
    state.async_geometry=true;

    if (start_live) ConnectLive(state);
    if (async_initial_bag && state.current_bag_source) {
      const auto selected=*state.current_bag_source;
      state.current_bag_source.reset();
      OpenBagSource(state,selected);
      state.input_load_autoplay=autoplay;
    }

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      CompleteBagPicker(state, window);
      CompleteDatasetPicker(state,window);
      UpdateRoiExport(state);
      CompleteBagLoad(state);
      UpdateSourceDurations(state);
      const bool geometry_ready=CompleteGeometry(state);
      const bool requested_frame_ready=CompleteFrameRequest(state) || geometry_ready;
      const bool live_detector_ready=UpdateDetector(state);
      if (!live_detector_ready && !geometry_ready) UpdateLiveFrame(state);
      int fb_width = 0, fb_height = 0, window_width = 0, window_height = 0;
      glfwGetFramebufferSize(window, &fb_width, &fb_height);
      glfwGetWindowSize(window, &window_width, &window_height);
      if (fb_width <= 0 || fb_height <= 0 || window_width <= 0 || window_height <= 0) {
        glfwWaitEventsTimeout(.05);
        continue;
      }
      state.ui_layout = lidar_viewer::ComputeUiLayout(
        static_cast<float>(window_width), static_cast<float>(window_height),
        state.ui_scale * state.font_raster_scale / std::max(
          static_cast<float>(fb_width) / window_width, static_cast<float>(fb_height) / window_height),state.inspector_width);
      ImGui::GetStyle() = base_ui_style;
      ImGui::GetStyle().ScaleAllSizes(state.ui_layout.scale);
      ImGui::GetIO().FontGlobalScale = state.ui_layout.scale / state.font_raster_scale;
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      BuildViewerInterface(state, window);
      HandlePlaybackShortcuts(state);
      const bool detector_result_ready=UpdateDetector(state) || live_detector_ready;
      const double frame_begin_time = glfwGetTime();
      bool advanced_frame = requested_frame_ready;

      if (!state.transient_status.empty() && glfwGetTime() >= state.transient_status_until) {
        state.transient_status.clear();
      }

      if (state.is_playing && !state.timeline_dragging && !state.requested_frame && state.frames.size() > 1 && !state.geometry_future.valid() && !DetectorFramePending(state) && !detector_result_ready) {
        const double now_time = glfwGetTime();
        const double frame_interval = 1.0 / std::max(0.1, state.playback_fps);
        if ((now_time - state.last_frame_advance_time) >= frame_interval) {
          const auto previous=state.current_index;
          AdvanceFrame(state, 1);
          if (state.current_index!=previous) {
            state.last_frame_advance_time = now_time;
            advanced_frame = true;
          }
        }
      }

      if (state.frame_dirty) {
        UploadFrame(state);
        state.frame_dirty = false;
      }
      if (state.fit_view_requested) {
        ResetCameraToFrame(state);
        state.fit_view_requested = false;
      }
      if(!state.frames.empty() && state.frames.front().dataset_sample && state.dataset_fitted_index!=state.current_index) {
        state.dataset_fitted_index=state.current_index;
        if(state.dataset_fit_each) {
          state.camera.target=state.frame.center;
          state.camera.distance=state.camera.projection_mode==CameraState::ProjectionMode::View2D ?
            lidar_viewer::TopViewHalfHeight(std::max(.1f,state.base_frame.radius*1.15f)) : std::max(1.f,state.base_frame.radius*2.2f);
          StoreCurrentProjectionView(state.camera);
        }
      }

      glViewport(0, 0, fb_width, fb_height);
      glDisable(GL_SCISSOR_TEST);
      glClearColor(.045f, .058f, .079f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      const auto & canvas = state.ui_layout.viewport;
      const float sx = static_cast<float>(fb_width) / static_cast<float>(window_width);
      const float sy = static_cast<float>(fb_height) / static_cast<float>(window_height);
      const int canvas_x = static_cast<int>(std::lround(canvas.x * sx));
      const int canvas_y = fb_height - static_cast<int>(std::lround((canvas.y + canvas.h) * sy));
      const int canvas_w = std::max(1, static_cast<int>(std::lround(canvas.w * sx)));
      const int canvas_h = std::max(1, static_cast<int>(std::lround(canvas.h * sy)));
      glViewport(canvas_x, canvas_y, canvas_w, canvas_h);
      glEnable(GL_SCISSOR_TEST);
      glScissor(canvas_x, canvas_y, canvas_w, canvas_h);
      glClearColor(.035f, .045f, .063f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      const float aspect = static_cast<float>(canvas_w) / static_cast<float>(canvas_h);
      const Vec3 eye =
        (state.camera.projection_mode == CameraState::ProjectionMode::View2D)
        ? Vec3{state.camera.target.x, state.camera.target.y, state.camera.target.z + std::max(state.camera.distance, 10.0f)}
        : CameraPosition(state.camera);
      const Mat4 projection =
        (state.camera.projection_mode == CameraState::ProjectionMode::View2D)
        ? Orthographic(
            -lidar_viewer::TopViewHalfHeight(state.camera.distance) * aspect,
            lidar_viewer::TopViewHalfHeight(state.camera.distance) * aspect,
            -lidar_viewer::TopViewHalfHeight(state.camera.distance),
            lidar_viewer::TopViewHalfHeight(state.camera.distance),
            0.1f,
            10000.0f)
        : Perspective(45.0f * kPi / 180.0f, aspect, 0.1f, 10000.0f);
      const Vec3 up =
        (state.camera.projection_mode == CameraState::ProjectionMode::View2D)
        ? lidar_viewer::TopViewUp(state.camera.yaw)
        : Vec3{0.0f, 0.0f, 1.0f};
      const Mat4 view = LookAt(eye, state.camera.target, up);
      const Mat4 mvp = Multiply(projection, view);

      glUseProgram(program);
      glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp.m.data());
      glUniform1f(point_size_loc, state.camera.point_size);
      glUniform1i(color_mode_loc, state.color_mode);
      glUniform1f(min_height_loc, state.height_color_min);
      glUniform1f(max_height_loc, state.height_color_max);
      if (state.grid.dirty) {
        RebuildGrid(state);
      }
      glBindVertexArray(state.vao);
      const auto visible=VisiblePoints(state);
      glDrawArrays(GL_POINTS,static_cast<GLint>(visible.first),static_cast<GLsizei>(visible.count));
      RenderGrid(state, mvp);
      RenderClusterBoxes(state, mvp);
      RenderDetectionLabels(state,mvp);
      RenderRoiBoxes(state, mvp);
      RenderAxes(state, mvp);
      glDisable(GL_SCISSOR_TEST);
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);

      ++state.fps_measurement_frames;
      const double fps_elapsed = frame_begin_time - state.fps_measurement_started_at;
      if (fps_elapsed >= 0.25) {
        state.current_view_fps =
          static_cast<double>(state.fps_measurement_frames) / std::max(fps_elapsed, 1e-6);
        state.fps_measurement_started_at = frame_begin_time;
        state.fps_measurement_frames = 0;
      }

      if (advanced_frame) {
        ++state.playback_measurement_advances;
      }
      const double playback_elapsed = frame_begin_time - state.playback_measurement_started_at;
      if (playback_elapsed >= 0.25) {
        state.current_playback_fps =
          static_cast<double>(state.playback_measurement_advances) / std::max(playback_elapsed, 1e-6);
        state.playback_measurement_started_at = frame_begin_time;
        state.playback_measurement_advances = 0;
      }
      // VSync is not enforced by every driver/headless display. Sleep between
      // frames instead of continuously repainting an unchanged cloud. Input
      // events wake GLFW immediately; ROS reception/recording stays independent.
      const bool active=(state.live_mode && !state.live_paused) || state.is_playing ||
        state.geometry_future.valid() || state.requested_frame.has_value() || state.loading_bag_source.has_value() ||
        ImGui::IsAnyItemActive();
      const double active_hz=state.is_playing && !state.live_mode ? std::max(60.,state.playback_fps) : 60.;
      const double remaining=1./(active ? active_hz : 20.)-(glfwGetTime()-frame_begin_time);
      if (remaining>0) glfwWaitEventsTimeout(remaining);
    }

    if (state.input_load) state.input_load->Cancel();
    if (state.export_progress) state.export_progress->cancel.store(true);
    state.live.Disconnect();
    glDeleteProgram(program);
    glDeleteProgram(state.grid.program);
    glDeleteBuffers(1, &state.vbo);
    glDeleteBuffers(1, &state.grid.vbo);
    glDeleteBuffers(1, &state.cluster_boxes.fill_vbo);
    glDeleteBuffers(1, &state.cluster_boxes.line_vbo);
    glDeleteBuffers(1, &state.roi_renderer.vbo);
    glDeleteBuffers(1,&state.axes_vbo);glDeleteVertexArrays(1,&state.axes_vao);
    glDeleteVertexArrays(1, &state.vao);
    glDeleteVertexArrays(1, &state.grid.vao);
    glDeleteVertexArrays(1, &state.cluster_boxes.vao);
    glDeleteVertexArrays(1, &state.roi_renderer.vao);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
