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
#include "lidar_viewer/file_dialog.hpp"
#include "lidar_viewer/live_io.hpp"
#include "lidar_viewer/platform.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
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

struct CalibrationSettings
{
  float yaw_deg{0.0f};
  float pitch_deg{0.0f};
  float roll_deg{0.0f};
  float z_m{0.0f};
};

std::string FileDisplayName(const fs::path & path)
{
  return path.has_filename() ? path.filename().string() : path.string();
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
  struct BagLoadResult
  {
    std::vector<FrameHandle> frames;
    FrameData first_frame;
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
  std::optional<std::size_t> current_bag_source;
  std::optional<std::size_t> loading_bag_source;
  std::future<BagLoadResult> bag_load_future;
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
  float font_raster_scale{1.0f};
  float height_color_min{-1.0f};
  float height_color_max{4.0f};
  lidar_viewer::UiLayout ui_layout;
  std::string ui_error;
  bool ground_removal_enabled{false};
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
  GridRenderer grid;
  ClusterBoxRenderer cluster_boxes;
  lidar_viewer::RoiSettings roi;
  lidar_viewer::RoiSettings saved_roi;
  lidar_viewer::RoiHistory roi_history;
  enum class RoiTool {Navigate, Draw, Move};
  RoiTool roi_tool{RoiTool::Navigate};
  bool roi_tab_requested{false}, roi_tab_active{false}, roi_pending{false}, roi_numeric_edit{false};
  bool roi_reveal_selection{true}, save_requested{false};
  float roi_new_z{0}, roi_new_height{6};
  int roi_preset{0};
  struct RoiGesture {
    bool active{false};
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
  std::string dataset_profile{"default"};
  std::unordered_map<std::size_t, FrameData> frame_cache;
  std::deque<std::size_t> frame_cache_order;
  std::future<FrameData> prefetch_future;
  std::optional<std::size_t> prefetched_index;
};

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
    state.camera.distance = std::max(bookmark.distance, 2.0f);
  } else if (mode == CameraState::ProjectionMode::View2D) {
    state.camera.target.z = 0.0f;
    state.camera.distance = std::max(state.camera.distance, 2.0f);
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
  std::vector<Vec3> & line_vertices)
{
  const Vec3 p000{bounds.min.x, bounds.min.y, bounds.min.z};
  const Vec3 p100{bounds.max.x, bounds.min.y, bounds.min.z};
  const Vec3 p110{bounds.max.x, bounds.max.y, bounds.min.z};
  const Vec3 p010{bounds.min.x, bounds.max.y, bounds.min.z};
  const Vec3 p001{bounds.min.x, bounds.min.y, bounds.max.z};
  const Vec3 p101{bounds.max.x, bounds.min.y, bounds.max.z};
  const Vec3 p111{bounds.max.x, bounds.max.y, bounds.max.z};
  const Vec3 p011{bounds.min.x, bounds.max.y, bounds.max.z};

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

  if (state.algorithm_config.detection.enabled && !state.detections.empty()) {
    state.cluster_boxes.batches.reserve(5);
    for (const auto & detection : state.detections) {
      const auto [fill_color, line_color] = DetectionClassColors(detection.class_id);
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
      AppendClusterBox(detection.bounds, batch_it->fill_vertices, batch_it->line_vertices);
    }
  } else {
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
      AppendClusterBox(cluster.bounds, batch.fill_vertices, batch.line_vertices);
    }
  }

  state.cluster_boxes.dirty = true;
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

    state.file_groups.push_back({
      current_path,
      FileDisplayName(current_path),
      first_index,
      last_index
    });
    first_index = last_index + 1;
  }

  state.current_file_group = 0;
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
  vertices.clear(); vertices.reserve(state.roi.boxes.size()*24);
  for (const auto & box : state.roi.boxes) {
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
  const lidar_viewer::PreparedRoi roi(state.roi, coordinates);
  FrameData frame;
  frame.frame_id=processed.frame_id; frame.imu=processed.imu; frame.ground_z=processed.ground_z;
  if (!roi.RejectsAll()) {
    frame.points.assign(processed.points.begin(),processed.points.begin()+static_cast<std::ptrdiff_t>(obstacles));
    roi.Filter(frame.points);
  }
  auto clusters=lidar_viewer::algorithms::RunClusteringPlaceholder(frame.points,state.algorithm_config.clustering);
  auto detections=lidar_viewer::algorithms::RunDetectionPlaceholder(frame.points,clusters,state.algorithm_config.detection);
  const auto retained_obstacles=frame.points.size();
  if (!roi.RejectsAll()) {
    frame.points.reserve(frame.points.size()+processed.points.size()-obstacles);
    for (std::size_t i=obstacles;i<processed.points.size();++i) {
      const auto & point=processed.points[i];
      if (roi.Contains({point.x,point.y,point.z})) frame.points.push_back(point);
    }
  }
  RecomputeFrameBounds(frame);
  state.ground_removed_count=frame.points.size()-retained_obstacles;
  state.frame=std::move(frame); state.clusters=std::move(clusters); state.detections=std::move(detections);
  state.roi_frame=coordinates;
  RebuildRoiGeometry(state); RebuildClusterBoxes(state);
  state.frame_dirty=true; state.roi_pending=false;
  state.roi_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

void ApplyCalibration(ViewerState & state, const FrameData * source = nullptr,
  std::optional<float> source_ground_offset = std::nullopt)
{
  const auto started=std::chrono::steady_clock::now();
  FrameData frame=source ? *source : state.base_frame;
  auto calibration=state.calibration;
  lidar_viewer::ImuLevelingResult leveling;
  leveling.status="Off";
  if (state.imu_level_enabled) {
    leveling=lidar_viewer::ComputeImuLeveling(frame.imu,state.imu_mount);
    if (leveling.applied) {calibration.roll_deg=leveling.roll_deg; calibration.pitch_deg=leveling.pitch_deg;}
  }
  const auto transform=BuildCalibrationTransform(calibration);
  lidar_viewer::RoiCoordinateFrame coordinates;
  coordinates.rotation=transform.rotation;
  coordinates.translation=coordinates.Rotate({0,0,-source_ground_offset.value_or(state.global_ground_offset)})+
    Vec3{0,0,transform.z_offset};
  for (auto & point : frame.points) {
    const auto rotated=ApplyCalibrationTransform({point.x,point.y,point.z},transform);
    point.x=rotated.x; point.y=rotated.y; point.z=rotated.z;
  }
  std::size_t obstacles=frame.points.size();
  if (state.ground_removal_enabled && !frame.points.empty()) {
    auto params=state.algorithm_config.ground_removal; params.enabled=true;
    auto result=lidar_viewer::algorithms::RunGroundRemovalPlaceholder(frame.points,params);
    obstacles=result.non_ground_points.size();
    frame.points=std::move(result.non_ground_points);
    frame.points.reserve(frame.points.size()+result.ground_points.size());
    for (auto point : result.ground_points) {point.intensity=-1; frame.points.push_back(point);}
    ++state.ground_passes;
  }
  // Cache segmentation before the ROI. Editing a box only filters this result;
  // surrounding terrain and ground classification remain identical.
  ApplyRoiResult(state,frame,obstacles,coordinates);
  state.processed_frame=std::move(frame); state.processed_obstacles=obstacles;
  if (state.imu_level_enabled && leveling.applied!=state.imu_level_result.applied)
    std::cout << "IMU leveling " << (leveling.applied ? "active: " : "unavailable: ") << leveling.status << std::endl;
  state.imu_level_result=std::move(leveling);
  state.processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
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
  while (state.frame_cache_order.size() > kMaxCachedFrames) {
    const std::size_t evict_index = state.frame_cache_order.front();
    state.frame_cache_order.pop_front();
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
  } catch (...) {
  }
  state.prefetched_index.reset();
}

void QueueNextFramePrefetch(ViewerState & state)
{
  if (state.frames.size() <= 1) {
    return;
  }

  ConsumeReadyPrefetch(state);
  const std::size_t next_index = (state.current_index + 1) % state.frames.size();
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
  state.prefetch_future = std::async(std::launch::async, [handle]() {
    return LoadFrame(handle);
  });
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
    LoadFrameByIndex(state, index);
    state.ui_error.clear();
    state.last_frame_advance_time = glfwGetTime();
    return true;
  } catch (const std::exception & error) {
    state.is_playing = false;
    state.ui_error = std::string("Frame could not be loaded: ") + error.what();
    std::cerr << state.ui_error << '\n';
    return false;
  }
}


void MarkGridDirty(ViewerState & state)
{
  state.grid_settings.spacing = std::clamp(state.grid_settings.spacing, 0.5f, 50.0f);
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

void OpenBagSource(ViewerState & state, std::size_t index)
{
  if (state.live.Active() || state.loading_bag_source || index >= state.bag_sources.size() ||
      (state.current_bag_source && *state.current_bag_source == index)) return;
  state.is_playing = false;
  state.timeline_dragging = false;
  try {
    const auto path = state.bag_sources[index].path;
    const auto topic = state.bag_topic;
    const auto imu_topic = state.bag_imu_topic;
    state.bag_load_future = std::async(std::launch::async, [path, topic, imu_topic]() {
      ViewerState::BagLoadResult result;
      result.frames = ResolveInputFrames(path, topic, imu_topic);
      if (result.frames.empty()) throw std::runtime_error("No point cloud frames in this bag");
      result.first_frame = LoadFrame(result.frames.front());
      return result;
    });
    state.loading_bag_source = index;
    state.ui_error.clear();
  } catch (const std::exception & error) {
    state.ui_error = std::string("Input could not be opened: ") + error.what();
  }
}

void LoadSettings(ViewerState & state);
fs::path AlgorithmsConfigPath();
std::string DetectDatasetProfile(const fs::path & input_path);

void StartBagPicker(ViewerState & state)
{
  if (state.live.Active() || state.loading_bag_source || state.folder_scan_future.valid() || state.bag_picker.Active()) return;
  state.is_playing = false;
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

void OpenInputPath(ViewerState & state, const fs::path & input)
{
  if (state.live.Active() || state.loading_bag_source || state.folder_scan_future.valid())
    throw std::runtime_error("Disconnect live input or wait for loading before opening a file");
  auto sources = DiscoverBagSources(input);
  if (sources.empty() && input.extension()==".bin" && fs::is_regular_file(input)) {
    const auto path=fs::canonical(input);
    sources.push_back({path,path.filename().u8string(),{}, {}});
  }
  if (sources.size() != 1)
    throw std::runtime_error("Select metadata.yaml, .db3, .mcap or a .bin point cloud");
  const auto selected = sources.front().path;
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
    if (count!=1) throw std::runtime_error("Drop one BIN file or one bag folder at a time");
    if (state->bag_picker.Active()) return;
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

void CompleteBagLoad(ViewerState & state)
{
  if (!state.loading_bag_source || !state.bag_load_future.valid() ||
      state.bag_load_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
  const auto selected = *state.loading_bag_source;
  try {
    auto result = state.bag_load_future.get();
    // Fit/process the replacement before discarding any currently displayed data.
    ViewerState staged;
    const auto profile=DetectDatasetProfile(state.bag_sources[selected].path);
    const bool change_profile = state.dataset_profile != profile;
    staged.dataset_profile = profile;
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
      staged.imu_level_enabled = state.imu_level_enabled;
      staged.imu_mount = state.imu_mount;
      staged.roi = state.roi;
      staged.selected_roi = state.selected_roi;
    }
    const float ground_offset=result.frames.front().format==InputFormat::KittiXyzi ? result.first_frame.ground_z : 0.f;
    ApplyGlobalAlignment(result.first_frame,ground_offset);
    ApplyCalibration(staged, &result.first_frame,ground_offset);
    // Cache keys are frame indices. Drain the previous reader before reusing them.
    if (state.prefetch_future.valid()) {
      try { state.prefetch_future.get(); } catch (...) {}
    }
    state.prefetched_index.reset();
    state.frame_cache.clear();
    state.frame_cache_order.clear();
    state.frames = std::move(result.frames);
    state.base_frame = std::move(result.first_frame);
    state.live_mode = false;
    state.live_has_frame = false;
    state.frame = std::move(staged.frame);
    state.processed_frame=std::move(staged.processed_frame);
    state.processed_obstacles=staged.processed_obstacles;
    state.processing_ms=staged.processing_ms; state.roi_ms=staged.roi_ms;
    state.ground_passes+=staged.ground_passes;
    state.roi_pending=false; state.roi_gesture.active=false; state.roi_tool=ViewerState::RoiTool::Navigate;
    if (change_profile) {state.saved_roi=staged.saved_roi; state.roi_history.Clear();}
    state.ground_removed_count = staged.ground_removed_count;
    state.clusters = std::move(staged.clusters);
    state.detections = std::move(staged.detections);
    state.cluster_boxes.batches = std::move(staged.cluster_boxes.batches);
    state.cluster_boxes.dirty = true;
    state.frame_dirty = true;
    state.imu_level_result = std::move(staged.imu_level_result);
    state.roi = std::move(staged.roi);
    state.selected_roi = staged.selected_roi;
    state.roi_frame = staged.roi_frame;
    state.roi_renderer.vertices = std::move(staged.roi_renderer.vertices);
    state.roi_renderer.dirty = true;
    if (change_profile) {
      state.calibration = staged.calibration;
      state.imu_level_enabled = staged.imu_level_enabled;
      state.imu_mount = staged.imu_mount;
      state.algorithm_config = staged.algorithm_config;
      state.ground_removal_enabled = staged.ground_removal_enabled;
      state.camera = staged.camera;
      state.grid_settings = staged.grid_settings;
      state.grid.dirty = true;
      state.color_mode = staged.color_mode;
      state.height_color_min = staged.height_color_min;
      state.height_color_max = staged.height_color_max;
      state.ui_scale = staged.ui_scale;
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
  } catch (const std::exception & error) {
    state.ui_error = std::string("Input could not be opened: ") + error.what();
    std::cerr << state.ui_error << std::endl;
  }
  state.loading_bag_source.reset();
  state.is_playing = false;
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
  if (state.prefetch_future.valid()) {try {state.prefetch_future.get();} catch (...) {}}
  state.prefetched_index.reset(); state.frame_cache.clear(); state.frame_cache_order.clear();
  state.frames.clear(); state.file_groups.clear(); state.current_bag_source.reset();
  state.base_frame = {}; state.frame = {}; state.processed_frame={}; state.processed_obstacles=0;
  state.roi_gesture.active=false; state.roi_pending=false; state.roi_tool=ViewerState::RoiTool::Navigate;
  state.clusters.clear(); state.detections.clear();
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
  if (!state.live_stats.connected || state.live_paused) return;
  auto next = state.live.TakeLatest();
  if (!next) return;
  try {
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
  for (const float value : {state.camera.point_size, state.camera.distance,
      state.calibration.yaw_deg, state.calibration.pitch_deg, state.calibration.roll_deg,
      state.calibration.z_m, state.grid_settings.spacing, state.grid_settings.thickness,
      state.height_color_min, state.height_color_max, state.ui_scale,
      state.imu_mount.roll_deg, state.imu_mount.pitch_deg, state.imu_mount.yaw_deg}) {
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
  out << "ui_scale=" << state.ui_scale << '\n';
  out << "loop_playback=" << (state.loop_playback ? 1 : 0) << '\n';
  CameraState camera_snapshot = state.camera;
  StoreCurrentProjectionView(camera_snapshot);

  out << "point_size=" << camera_snapshot.point_size << '\n';
  out << "ground_cell_size_m=" << state.algorithm_config.ground_removal.candidate_grid_cell_m << '\n';
  out << "ground_mode=" << state.algorithm_config.ground_removal.mode << '\n';
  out << "ground_use_cuda=" << state.algorithm_config.ground_removal.use_cuda << '\n';
  out << "gseg3d_ground_inlier_threshold_m=" << state.algorithm_config.ground_removal.gseg3d_ground_inlier_threshold_m << '\n';
  out << "gseg3d_cell_size_x_m=" << state.algorithm_config.ground_removal.gseg3d_cell_size_x_m << '\n';
  out << "gseg3d_cell_size_y_m=" << state.algorithm_config.ground_removal.gseg3d_cell_size_y_m << '\n';
  out << "gseg3d_cuda_cell_size_m=" << state.algorithm_config.ground_removal.gseg3d_cuda_cell_size_m << '\n';
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
  out << "clustering_enabled=" << (state.algorithm_config.clustering.enabled ? 1 : 0) << '\n';
  out << "detection_enabled=" << (state.algorithm_config.detection.enabled ? 1 : 0) << '\n';
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
  out << lidar_viewer::SerializeRoiSettings(state.roi);
  if (!out) {
    throw std::runtime_error("Failed to serialize settings");
  }
  lidar_viewer::settings::AtomicWriteText(destination, out.str());
  state.saved_roi=state.roi;
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
        if (value == "fast_grid" || value == "gseg3d" || value == "patchworkpp" || value == "grid" || value == "ransac")
          state.algorithm_config.ground_removal.mode = value;
        else throw std::invalid_argument("Unknown ground segmentation method");
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
        state.camera.distance = std::clamp(ParseFiniteFloat(value), 1.0f, 5000.0f);
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
        state.camera.saved_2d_view.distance = std::clamp(ParseFiniteFloat(value), 1.0f, 5000.0f);
      } else if (key == "camera_2d_valid") {
        state.camera.saved_2d_view.valid = (ParseStrictInt(value) != 0);
      } else if (key == "ground_removal_enabled") {
        state.ground_removal_enabled = (ParseStrictInt(value) != 0);
      } else if (key == "clustering_enabled") {
        state.algorithm_config.clustering.enabled = (ParseStrictInt(value) != 0);
      } else if (key == "detection_enabled") {
        state.algorithm_config.detection.enabled = (ParseStrictInt(value) != 0);
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
  }
  MarkGridDirty(state);
}

void LoadSettings(ViewerState & state)
{
#ifdef LIDAR_VIEWER_USER_SETTINGS
  LoadSettingsFile(state,lidar_viewer::ProjectResourcePath(kProfilesDir)/(state.dataset_profile+".cfg"));
#endif
  LoadSettingsFile(state, ProfileSettingsPath(state.dataset_profile));
  if (state.dataset_profile == "default") {
    LoadSettingsFile(state, SettingsPath());
  }
  state.saved_roi=state.roi; state.roi_history.Clear();
}


void AdvanceFrame(ViewerState & state, int direction)
{
  if (state.live_mode || state.frames.empty() || state.loading_bag_source) {
    return;
  }
  if (direction > 0) {
    if (state.current_index + 1 >= state.frames.size() && !state.loop_playback) {
      state.is_playing = false;
      return;
    }
    TryLoadFrame(state, (state.current_index + 1) % state.frames.size());
  } else {
    if (state.current_index == 0 && !state.loop_playback) {
      return;
    }
    TryLoadFrame(state, state.current_index == 0 ? state.frames.size() - 1 : state.current_index - 1);
  }
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
  state.is_playing=false; SetLivePaused(state,true);
}

void FinishRoiGesture(ViewerState & state, bool cancel);
void SelectRoiTool(ViewerState & state, ViewerState::RoiTool tool)
{
  if (state.roi_gesture.active) FinishRoiGesture(state,true);
  state.roi_tool=tool;
  if (tool!=ViewerState::RoiTool::Navigate) {
    SetProjectionMode(state,CameraState::ProjectionMode::View2D);
    state.roi_tab_requested=true; FreezeForRoi(state);
  }
}

void FinishRoiGesture(ViewerState & state, bool cancel)
{
  if (!state.roi_gesture.active) return;
  if (state.roi_tool==ViewerState::RoiTool::Draw && !state.roi.boxes.empty()) {
    const auto & box=state.roi.boxes[state.selected_roi];
    if (box.size.x<.1f || box.size.y<.1f) cancel=true;
  }
  if (cancel) state.roi=state.roi_gesture.before;
  else if (!lidar_viewer::SameRoi(state.roi_gesture.before,state.roi)) state.roi_history.Record(state.roi_gesture.before);
  state.roi_gesture.active=false; state.roi_pending=true; state.roi_reveal_selection=true;
  if (state.selected_roi>=state.roi.boxes.size()) state.selected_roi=state.roi.boxes.empty() ? 0 : state.roi.boxes.size()-1;
  RebuildRoiGeometry(state);
  if (state.roi_tool==ViewerState::RoiTool::Draw) state.roi_tool=ViewerState::RoiTool::Navigate;
}

std::optional<Vec3> RoiPointerPoint(const ViewerState & state, double x, double y, float z)
{
  const auto & view=state.ui_layout.viewport;
  if (view.w<1 || view.h<1) return {};
  return lidar_viewer::RoiPointFromTopView(static_cast<float>(2*(x-view.x)/view.w-1),
    static_cast<float>(1-2*(y-view.y)/view.h),view.w/view.h,std::max(state.camera.distance,2.f),
    state.camera.target,state.camera.yaw,state.roi_frame,z);
}

void DuplicateSelectedRoi(ViewerState & state)
{
  if (state.roi.boxes.empty() || state.roi.boxes.size()>=lidar_viewer::kMaxRoiBoxes) return;
  auto copy=lidar_viewer::DuplicateRoi(state.roi.boxes[state.selected_roi],state.roi);
  state.roi_history.Record(state.roi); state.roi.boxes.push_back(std::move(copy));
  state.selected_roi=state.roi.boxes.size()-1; state.roi_pending=true; state.roi_reveal_selection=true; FreezeForRoi(state);
}

void DeleteSelectedRoi(ViewerState & state)
{
  if (state.roi.boxes.empty()) return;
  state.roi_history.Record(state.roi);
  state.roi.boxes.erase(state.roi.boxes.begin()+static_cast<std::ptrdiff_t>(state.selected_roi));
  if (state.selected_roi) --state.selected_roi;
  state.roi_pending=true; FreezeForRoi(state);
}

void RestoreRoiHistory(ViewerState & state, bool redo)
{
  if (state.roi_gesture.active) {FinishRoiGesture(state,true); return;}
  if (redo ? state.roi_history.Redo(state.roi) : state.roi_history.Undo(state.roi)) {
    state.selected_roi=state.roi.boxes.empty() ? 0 : std::min(state.selected_roi,state.roi.boxes.size()-1);
    state.roi_pending=true; state.roi_numeric_edit=false; state.roi_reveal_selection=true; FreezeForRoi(state);
    RebuildRoiGeometry(state);
  }
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
    gesture.before=state->roi;
    if (state->roi_tool==ViewerState::RoiTool::Draw) {
      if (state->roi.boxes.size()>=lidar_viewer::kMaxRoiBoxes) return;
      const auto point=RoiPointerPoint(*state,x,y,state->roi_new_z);
      if (!point) {state->ui_error="Cannot draw on the LiDAR XY plane at this tilt."; return;}
      gesture.start=*point;
      const auto right_point=RoiPointerPoint(*state,x+10,y,state->roi_new_z);
      if (!right_point) return;
      const auto right=*right_point-*point;
      const float yaw=std::atan2(right.y,right.x)*180.f/kPi;
      auto box=lidar_viewer::RoiFromDrag(*point,*point,state->roi_new_z,state->roi_new_height,
        lidar_viewer::NextRoiName(state->roi),yaw);
      state->roi.boxes.push_back(box); state->selected_roi=state->roi.boxes.size()-1;
      state->roi.enabled=true; state->roi.show_boxes=true; gesture.original=box;
    } else {
      bool found=false;
      for (std::size_t n=0;n<state->roi.boxes.size();++n) {
        const auto i=n==0 ? state->selected_roi : (n<=state->selected_roi ? n-1 : n);
        const auto & box=state->roi.boxes[i];
        const auto point=RoiPointerPoint(*state,x,y,box.center.z);
        auto hit_box=box; hit_box.enabled=true;
        if (point && lidar_viewer::PreparedRoi({true,true,{hit_box}}).Contains(*point)) {
          state->selected_roi=i; gesture.start=*point; gesture.original=box; found=true; break;
        }
      }
      if (!found) return;
    }
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
      auto box=state->roi.boxes[state->selected_roi];
      try {
        if (state->roi_tool==ViewerState::RoiTool::Draw)
          box=lidar_viewer::RoiFromDrag(state->roi_gesture.start,*point,box.center.z,box.size.z,box.name,
            state->roi_gesture.original.yaw_deg);
        else box.center=state->roi_gesture.original.center+(*point-state->roi_gesture.start);
        lidar_viewer::ValidateRoiBox(box);
        state->roi.boxes[state->selected_roi]=box; RebuildRoiGeometry(*state);
      } catch (const std::exception & error) {state->ui_error=error.what();}
    }
    return;
  }
  if (!PointerInScene(window, *state) ||
      (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)) return;
  const float scale = std::max(kPanBaseScale * state->camera.distance, 0.01f);
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
  state->camera.distance = std::clamp(
    state->camera.distance * static_cast<float>(std::pow(0.9, yoffset)), 1.0f, 5000.0f);
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
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_N: AdvanceFrame(*state, 1); break;
    case GLFW_KEY_LEFT:
    case GLFW_KEY_P: AdvanceFrame(*state, -1); break;
    case GLFW_KEY_F: state->fit_view_requested = true; break;
    case GLFW_KEY_C: state->color_mode = (state->color_mode + 1) % 3; break;
    case GLFW_KEY_Q:
    case GLFW_KEY_E:
      if (state->camera.projection_mode == CameraState::ProjectionMode::View2D)
        state->camera.yaw = lidar_viewer::RotateTopView(state->camera.yaw, (key == GLFW_KEY_Q ? -1.f : 1.f) * kPi / 36);
      break;
    case GLFW_KEY_LEFT_BRACKET: state->camera.point_size = std::max(1.f, state->camera.point_size - 1.f); break;
    case GLFW_KEY_RIGHT_BRACKET: state->camera.point_size = std::min(8.f, state->camera.point_size + 1.f); break;
    case GLFW_KEY_SPACE:
      if (state->live_mode) {SetLivePaused(*state, !state->live_paused); break;}
      if (state->frames.empty() || state->loading_bag_source) break;
      state->is_playing = !state->is_playing;
      state->last_frame_advance_time = glfwGetTime();
      break;
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
  style.Colors[ImGuiCol_TextDisabled] = {0.48f, 0.56f, 0.66f, 1};
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
  style.Colors[ImGuiCol_Header] = {0.12f, 0.28f, 0.37f, 1};
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
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(std::min(22*ImGui::GetFontSize(),ImGui::GetIO().DisplaySize.x-32));
    ImGui::TextUnformatted(value);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

std::string BagLabel(const BagSource & source)
{
  return (source.recording_date.empty() ? "Unknown date" :
    source.recording_date + "  " + source.recording_time) + "  |  " + source.display_name;
}

void BuildSourceSelector(ViewerState & state, float width)
{
  const auto * group = state.file_groups.empty() ? nullptr : &state.file_groups[state.current_file_group];
  const auto preview = state.live_mode ? std::string("LIVE | ") + state.live_points_topic.data() :
    state.current_bag_source ? BagLabel(state.bag_sources[*state.current_bag_source]) :
    (group ? group->display_name : "No input loaded");
  ImGui::SetNextItemWidth(width);
  const float popup_width = std::min(720.f * state.ui_layout.scale, ImGui::GetIO().DisplaySize.x - 32.f);
  ImGui::SetNextWindowSizeConstraints({popup_width, 0}, {popup_width, 600.f * state.ui_layout.scale});
  ImGui::BeginDisabled(state.loading_bag_source.has_value() || state.folder_scan_future.valid() ||
    state.bag_picker.Active() || state.live.Active() || (!group && state.bag_sources.empty()));
  if (ImGui::BeginCombo("##Source", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
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
      for (std::size_t i = 0; i < state.bag_sources.size(); ++i) {
        const auto & source = state.bag_sources[i];
        const auto date = source.recording_date.empty() ? "Unknown date" : source.recording_date;
        if (!state.bag_date_filter.empty() && date != state.bag_date_filter) continue;
        std::string searchable = source.display_name;
        std::transform(searchable.begin(), searchable.end(), searchable.begin(), lower);
        if (!needle.empty() && searchable.find(needle) == std::string::npos) continue;
        ++shown;
        const bool selected = state.current_bag_source && *state.current_bag_source == i;
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(BagLabel(source).c_str(), selected)) OpenBagSource(state, i);
        Tip(source.path.string().c_str());
        ImGui::PopID();
      }
      if (!shown) ImGui::TextDisabled("No bags match this date or name.");
    }
    if (!state.current_bag_source && !state.frames.empty() && state.frames.front().format != InputFormat::Ros2Bag) {
      if (!state.bag_sources.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Current BIN sequence");
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
    ImGui::TextDisabled("Open file: select a recording. Add folder: register a collection.");
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
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
      auto merged = MergeBagSources(state.bag_sources, found.sources);
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
  if (state.roi.boxes.empty()) return;
  const auto & box=state.roi.boxes[state.selected_roi];
  state.camera.target=state.roi_frame.Transform(box.center);
  state.camera.distance=std::clamp(1.25f*std::max({box.size.x,box.size.y,box.size.z}),2.f,5000.f);
}

void BuildRoiEditor(ViewerState & state)
{
  // End the previous typing transaction before another field or button takes
  // focus. A button's mouse-down is active before its click changes the ROI.
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Tab))
    state.roi_numeric_edit=false;
  const auto before=state.roi;
  bool history_handled=false;
  const float s=state.ui_layout.scale;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{8*s,6*s});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*s,5*s});
  SectionTitle("Regions of interest");
  ImGui::Checkbox("Crop points",&state.roi.enabled);
  ImGui::SameLine(); ImGui::Checkbox("Outlines",&state.roi.show_boxes);
  ImGui::TextDisabled("%zu / %zu points  |  %zu boxes",state.frame.points.size(),state.base_frame.points.size(),state.roi.boxes.size());
  ImGui::BeginDisabled(state.roi.boxes.size()>=lidar_viewer::kMaxRoiBoxes || state.roi_gesture.active);
  ImGui::PushStyleColor(ImGuiCol_Button,{.12f,.36f,.44f,1});
  if (ImGui::Button(state.roi_tool==ViewerState::RoiTool::Draw ? "Drawing... Esc to cancel" : "Draw box in 2D  [R]",{-1,0}))
    SelectRoiTool(state,ViewerState::RoiTool::Draw);
  ImGui::PopStyleColor();
  Tip("Drag a footprint on the canvas. Height is set in Creation settings. The view and live display pause for editing.");
  const float third=(ImGui::GetContentRegionAvail().x-16*s)/3;
  if (ImGui::Button("+ Box",{third,0})) {
    lidar_viewer::RoiBox box; box.name=lidar_viewer::NextRoiName(state.roi);
    box.center=state.roi_frame.InverseTransform(state.camera.target); box.center.z=state.roi_new_z; box.size.z=state.roi_new_height;
    if (state.roi_preset==1) box.size={4,4,state.roi_new_height};
    if (state.roi_preset==2) {box.center={8,0,state.roi_new_z};box.size={12,8,state.roi_new_height};}
    try {
      lidar_viewer::ValidateRoiBox(box);
      state.roi.boxes.push_back(box); state.selected_roi=state.roi.boxes.size()-1;
      state.roi.enabled=true; state.roi.show_boxes=true;
    } catch (const std::exception & error) {state.ui_error=error.what();}
  }
  Tip("Create the chosen preset at the current view center. Front preset is relative to the sensor.");
  ImGui::SameLine();
  ImGui::BeginDisabled(state.roi.boxes.empty());
  if (ImGui::Button("Duplicate",{third,0})) {
    try {
      auto copy=lidar_viewer::DuplicateRoi(state.roi.boxes[state.selected_roi],state.roi);
      state.roi.boxes.push_back(std::move(copy));state.selected_roi=state.roi.boxes.size()-1;
    } catch (const std::exception & error) {state.ui_error=error.what();}
  }
  Tip("Copy beside the selected box, with a 0.5 m gap. Ctrl+D.");
  ImGui::EndDisabled(); ImGui::EndDisabled();
  ImGui::SameLine(); ImGui::BeginDisabled(state.roi.boxes.empty() || state.roi_gesture.active);
  ImGui::PushStyleColor(ImGuiCol_Text,{1,.55f,.5f,1});
  if (ImGui::Button("Delete",{third,0})) {
    state.roi.boxes.erase(state.roi.boxes.begin()+static_cast<std::ptrdiff_t>(state.selected_roi));
    if (state.selected_roi) --state.selected_roi;
  }
  ImGui::PopStyleColor(); ImGui::EndDisabled(); Tip("Delete selected box. Undo with Ctrl+Z.");
  if (before.boxes.size()!=state.roi.boxes.size()) state.roi_reveal_selection=true;
  if (ImGui::BeginListBox("##RoiList",{-1,112*s})) {
    for (std::size_t i=0;i<state.roi.boxes.size();++i) {
      ImGui::PushID(static_cast<int>(i));
      auto & box=state.roi.boxes[i];
      ImGui::BeginDisabled(state.roi_gesture.active);
      ImGui::Checkbox("##Enabled",&box.enabled); ImGui::SameLine();
      if (ImGui::Selectable(box.name.empty() ? "Unnamed box" : box.name.c_str(),i==state.selected_roi)) state.selected_roi=i;
      if (i==state.selected_roi && state.roi_reveal_selection) ImGui::SetScrollHereY(.5f);
      ImGui::EndDisabled(); ImGui::PopID();
    }
    state.roi_reveal_selection=false;
    ImGui::EndListBox();
  }
  if (state.roi.boxes.empty()) {
    ImGui::TextWrapped("Draw an area in the top view, or add a box and enter its dimensions.");
  } else {
    state.selected_roi=std::min(state.selected_roi,state.roi.boxes.size()-1);
    ImGui::BeginDisabled(state.roi_gesture.active);
    ImGui::PushID(static_cast<int>(state.selected_roi));
    auto box=state.roi.boxes[state.selected_roi];
    std::array<char,128> name{}; std::copy(box.name.begin(),box.name.end(),name.begin());
    ImGui::SetNextItemWidth(-1);
    bool edited=ImGui::InputText("##RoiName",name.data(),name.size());
    if (edited) box.name=name.data();
    float center[]{box.center.x,box.center.y,box.center.z}, size[]{box.size.x,box.size.y,box.size.z};
    ImGui::TextUnformatted("Center X / Y / Z (m)"); ImGui::SetNextItemWidth(-1);
    edited|=ImGui::InputFloat3("##RoiCenter",center,"%.3f");
    ImGui::TextUnformatted("Size X / Y / Z (m)"); ImGui::SetNextItemWidth(-1);
    edited|=ImGui::InputFloat3("##RoiSize",size,"%.3f");
    ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Yaw"); ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(60*s,ImGui::GetContentRegionAvail().x-74*s));
    edited|=ImGui::InputFloat("##RoiYawInput",&box.yaw_deg,0,0,"%.2f deg");
    ImGui::SameLine(); if (ImGui::Button("Reset",{66*s,0})) {box.yaw_deg=0;edited=true;}
    box.center={center[0],center[1],center[2]};box.size={size[0],size[1],size[2]};
    if (edited) {
      try {
        lidar_viewer::ValidateRoiBox(box); box.yaw_deg=std::remainder(box.yaw_deg,360.f);
        state.roi.boxes[state.selected_roi]=box; state.ui_error.clear();
      } catch (const std::exception & error) {state.ui_error=error.what();}
    }
    const float half=(ImGui::GetContentRegionAvail().x-8*s)*.5f;
    if (ImGui::Button("Focus box",{half,0})) FocusSelectedRoi(state);
    ImGui::SameLine(); if (ImGui::Button("Move in 2D",{half,0})) SelectRoiTool(state,ViewerState::RoiTool::Move);
    Tip("Click a box in the canvas and drag to move it. Height and yaw are preserved.");
    ImGui::PopID();
    ImGui::EndDisabled();
  }
  ImGui::BeginDisabled(state.roi_gesture.active);
  const float half=(ImGui::GetContentRegionAvail().x-8*s)*.5f;
  ImGui::BeginDisabled(!state.roi_history.CanUndo());
  if (ImGui::Button("Undo",{half,0})) {RestoreRoiHistory(state,false);history_handled=true;}
  ImGui::EndDisabled();ImGui::SameLine();ImGui::BeginDisabled(!state.roi_history.CanRedo());
  if (ImGui::Button("Redo",{half,0})) {RestoreRoiHistory(state,true);history_handled=true;}
  ImGui::EndDisabled();ImGui::EndDisabled();
  if (state.roi.enabled && std::none_of(state.roi.boxes.begin(),state.roi.boxes.end(),[](const auto & b){return b.enabled;}))
    ImGui::TextColored({1,.72f,.36f,1},"No enabled boxes: crop is empty.");
  if (ImGui::CollapsingHeader("Creation settings")) {
    ImGui::SetNextItemWidth(-1);ImGui::Combo("##RoiPreset",&state.roi_preset,"View center / 20 x 20 m\0View center / 4 x 4 m\0Sensor front / 12 x 8 m\0");
    ImGui::TextUnformatted("New box: center Z / height (m)");
    float z[]{state.roi_new_z,state.roi_new_height};ImGui::SetNextItemWidth(-1);
    if (ImGui::InputFloat2("##RoiNewHeight",z,"%.2f")) {
      if (std::isfinite(z[0])) state.roi_new_z=std::clamp(z[0],-1e6f,1e6f);
      if (std::isfinite(z[1])) state.roi_new_height=std::clamp(z[1],.001f,1e6f);
    }
  }
  if (!state.roi.boxes.empty() && ImGui::CollapsingHeader("Coordinates & corners")) {
    ImGui::TextWrapped("Coordinates use the raw LiDAR frame. Enabled boxes form a union. IMU leveling transforms points and boxes together.");
    const auto corners=lidar_viewer::RoiCorners(state.roi.boxes[state.selected_roi]);
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i=0;i<corners.size();++i) {
      const auto p=corners[i];ImGui::Text("%zu: %.2f, %.2f, %.2f",i+1,p.x,p.y,p.z);
      out<<p.x<<", "<<p.y<<", "<<p.z<<'\n';
    }
    if (ImGui::Button("Copy 8 corners")) ImGui::SetClipboardText(out.str().c_str());
  }
  if (!history_handled && !lidar_viewer::SameRoi(before,state.roi)) {
    if (!state.roi_numeric_edit) state.roi_history.Record(before);
    state.roi_numeric_edit=ImGui::IsAnyItemActive();
    state.roi_pending=true; FreezeForRoi(state); RebuildRoiGeometry(state);
  }
  if (!ImGui::IsAnyItemActive()) state.roi_numeric_edit=false;
  ImGui::PopStyleVar(2);
}

void BuildViewerInterface(ViewerState & state, GLFWwindow *)
{
  const auto layout = state.ui_layout;
  const float s = layout.scale;
  bool reprocess = false;
  const bool has_input = state.live_mode ? state.live_has_frame : !state.frames.empty();
  BeginPanel("Header", layout.header, false);
  ImGui::SetCursorPosY(15.f * s);
  ImGui::TextColored({.35f, .82f, .96f, 1.f}, "LiDAR");
  ImGui::SameLine();
  ImGui::TextUnformatted("Studio");
  ImGui::SameLine(190.f*s);
  ImGui::BeginDisabled(state.live.Active() || state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active());
  if (ImGui::Button("Open file",{110*s,0})) StartBagPicker(state);
  ImGui::EndDisabled(); Tip("Open BIN or ROS bag. Ctrl+O.");
  ImGui::SameLine();
  ImGui::BeginDisabled(state.loading_bag_source.has_value() || state.folder_scan_future.valid() || state.bag_picker.Active());
  if (ImGui::Button("Live...",{85*s,0})) state.open_live_dialog=true;
  ImGui::EndDisabled();
  if (layout.header.w>1150*s) {
    ImGui::SameLine();
    if (ImGui::Button("ROI editor",{110*s,0})) state.roi_tab_requested=true;
  }
  const float header_right = std::max(320.f * s, layout.header.w - 360.f * s);
  ImGui::SameLine(header_right);
  ImGui::TextColored(!state.ui_error.empty() ? ImVec4(1.f, .56f, .48f, 1) :
    (state.loading_bag_source ? ImVec4(.44f, .76f, 1, 1) :
      (state.is_playing ? ImVec4(.42f, .85f, .65f, 1) : ImVec4(.76f, .81f, .88f, 1))),
    "%s", !state.ui_error.empty() ? "ERROR" :
      (state.live_mode ? (state.live_stats.connected ? (state.live_paused ? "FROZEN" :
        (state.live_stats.clouds==0 ? "WAITING" : (state.live_stats.cloud_age_ms>1000 ? "STALE" : "LIVE"))) : "DISCONNECTED") :
      (state.loading_bag_source ? "OPENING INPUT" :
        (has_input ? (state.is_playing ? "PLAYING" : "PAUSED") : "READY"))));
  ImGui::SameLine();
  ImGui::TextDisabled("  %s", state.dataset_profile.c_str());
  ImGui::SameLine();
  if (ImGui::SmallButton("Help")) ImGui::OpenPopup("Shortcuts");
  if (ImGui::BeginPopup("Shortcuts")) {
    ImGui::TextUnformatted("Keyboard & mouse");
    ImGui::Separator();
    ImGui::TextUnformatted("Ctrl+O       Open file chooser\nCtrl+S       Save profile\nR            Draw ROI in top view\nCtrl+Z / Y   Undo / redo ROI\nCtrl+D       Duplicate ROI (ROI tab)\nDelete       Delete ROI (ROI tab)\nEsc          Cancel ROI tool / quit\nSpace        Play / pause\nN / P        Next / previous frame\nF            Fit all points\nC            Cycle point colors\n[ / ]        Point size\nQ / E        Rotate 2D view\nLeft drag    Orbit (3D) / pan (2D)\nRight drag   Pan (3D) / rotate (2D)\nWheel        Zoom");
    ImGui::EndPopup();
  }
  ImGui::End();

  BeginPanel("Inspector", layout.sidebar, false);
  ImGui::TextUnformatted("Inspector");
  ImGui::SameLine();
  ImGui::TextDisabled("  %s", state.live_mode ? "LIVE ROS 2" : has_input ?
    (state.frames.front().format == InputFormat::Ros2Bag ? "ROS 2 BAG" : "BIN") : "NO DATA");
  ImGui::Spacing();
  ImGui::BeginChild("InspectorBody", {0, std::max(1.f, ImGui::GetContentRegionAvail().y - 84.f * s)});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {1.f * s, 6.f * s});
  const bool tabs_open = ImGui::BeginTabBar("InspectorTabs");
  ImGui::PopStyleVar();
  state.roi_tab_active=false;
  const auto begin_tab = [s,&state](const char * label) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {1.f * s, 6.f * s});
    const bool is_roi=std::string(label)=="ROI";
    const bool open=ImGui::BeginTabItem(label,nullptr,is_roi && state.roi_tab_requested ? ImGuiTabItemFlags_SetSelected : 0);
    if (is_roi && open) {if (state.roi_tab_requested) ImGui::SetScrollY(0);state.roi_tab_requested=false;state.roi_tab_active=true;}
    ImGui::PopStyleVar();
    return open;
  };
  if (tabs_open) {
    if (begin_tab("View")) {
      SectionTitle("Camera");
      const bool is2d = state.camera.projection_mode == CameraState::ProjectionMode::View2D;
      if (ImGui::Selectable("3D perspective", !is2d)) SetProjectionMode(state, CameraState::ProjectionMode::View3D);
      if (ImGui::Selectable("2D top view", is2d)) SetProjectionMode(state, CameraState::ProjectionMode::View2D);
      ImGui::BeginDisabled(!has_input);
      if (ImGui::Button("Fit all points", {-1, 0})) ResetCameraToFrame(state);
      Tip("Fit the currently visible points. This changes the camera, never point coordinates.");
      ImGui::EndDisabled();
      if (ImGui::Button("Sensor view", {-1, 0})) {
        state.camera.target = {0, 0, 0};
        state.camera.distance = 35;
        state.camera.pitch = .55f;
        state.camera.yaw = -.8f;
        if (state.camera.projection_mode == CameraState::ProjectionMode::View2D) {
          state.camera.pitch = -1.5707f;
          state.camera.yaw = 0;
        }
        StoreCurrentProjectionView(state.camera);
      }
      if (state.camera.projection_mode == CameraState::ProjectionMode::View2D) {
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderAngle("##TopRotation", &state.camera.yaw, -180, 180, "2D rotation: %.1f deg");
        if (ImGui::Button("Reset 2D rotation", {-1, 0})) state.camera.yaw = 0;
        Tip("Left drag: pan. Right drag or Q/E: rotate. Wheel: zoom.");
      }
      SectionTitle("Point appearance");
      ImGui::SetNextItemWidth(-1);
      ImGui::Combo("##Color", &state.color_mode, "Intensity\0Height\0Uniform\0");
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##PointSize", &state.camera.point_size, 1, 8, "Point size: %.1f px", ImGuiSliderFlags_AlwaysClamp);
      if (state.color_mode == 1) {
        ImGui::TextDisabled("Height color range (m)");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloatRange2("##HeightRange", &state.height_color_min, &state.height_color_max, .05f, -100, 100, "%.2f", "%.2f");
        if (!std::isfinite(state.height_color_min)) state.height_color_min = -1;
        if (!std::isfinite(state.height_color_max)) state.height_color_max = 4;
        state.height_color_max = std::max(state.height_color_min + .01f, state.height_color_max);
      }
      SectionTitle("Reference grid");
      ImGui::Checkbox("Show grid", &state.grid_settings.enabled);
      bool grid_dirty = false;
      ImGui::SetNextItemWidth(-1);
      grid_dirty |= ImGui::SliderFloat("##Spacing", &state.grid_settings.spacing, .5f, 20.f, "Spacing: %.1f m", ImGuiSliderFlags_AlwaysClamp);
      ImGui::SetNextItemWidth(-1);
      grid_dirty |= ImGui::SliderInt("##Extent", &state.grid_settings.count, 2, 100, "Grid count: %d", ImGuiSliderFlags_AlwaysClamp);
      ImGui::SetNextItemWidth(-1);
      grid_dirty |= ImGui::SliderFloat("##GridLines", &state.grid_settings.thickness, 1, 4, "Line width: %.1f", ImGuiSliderFlags_AlwaysClamp);
      if (grid_dirty) MarkGridDirty(state);
      SectionTitle("Interface");
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##UiScale", &state.ui_scale, .75f, 1.5f, "UI size: %.2fx", ImGuiSliderFlags_AlwaysClamp);
      Tip("Relative to system DPI; automatically limited to keep the viewport usable.");
      ImGui::EndTabItem();
    }
    if (begin_tab("Ground")) {
      auto & ground = state.algorithm_config.ground_removal;
      SectionTitle("Ground segmentation");
      reprocess |= ImGui::Checkbox("Highlight ground", &state.ground_removal_enabled);
      ImGui::TextDisabled("Ground points are shown in red.");
      const char * modes[] = {"fast_grid", "gseg3d", "patchworkpp", "grid", "ransac"};
      const char * labels[] = {"Fast grid", "GSeg3D (reference CPU)", "Patchwork++", "Grid", "RANSAC"};
      int mode = 0;
      for (int i = 0; i < 5; ++i) if (ground.mode == modes[i]) mode = i;
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##GroundMethod", &mode, labels, 5)) {
        ground.mode = modes[mode];
        reprocess = true;
      }
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
      } else {
        ImGui::SetNextItemWidth(-1);
        reprocess |= ImGui::SliderFloat("##GroundCells", &ground.candidate_grid_cell_m, .1f, 3.f, "Cell size: %.2f m", ImGuiSliderFlags_AlwaysClamp);
      }
#ifdef LIDAR_VIEWER_HAS_CUDA
      ImGui::BeginDisabled(ground.mode == "gseg3d");
      reprocess |= ImGui::Checkbox("Use CUDA when supported", &ground.use_cuda);
      ImGui::EndDisabled();
#else
      ImGui::TextDisabled("CPU backend");
#endif
      Metric("Ground points", std::to_string(state.ground_removed_count));
      SectionTitle("Object analysis");
      if (ImGui::Checkbox("Cluster obstacles", &state.algorithm_config.clustering.enabled)) {
        reprocess = true;
        if (!state.algorithm_config.clustering.enabled) state.algorithm_config.detection.enabled = false;
      }
      ImGui::BeginDisabled(!state.algorithm_config.clustering.enabled);
      reprocess |= ImGui::Checkbox("Geometric labels (experimental)", &state.algorithm_config.detection.enabled);
      ImGui::EndDisabled();
      if (!state.algorithm_config.clustering.enabled) ImGui::TextDisabled("Requires obstacle clustering.");
      ImGui::TextWrapped("Labels are shape-based heuristics, not trained object recognition.");
      Metric("Clusters", std::to_string(state.clusters.size()));
      Metric("Labels", std::to_string(state.detections.size()));
      ImGui::EndTabItem();
    }
    if (begin_tab("Align")) {
      SectionTitle("IMU leveling");
      reprocess |= ImGui::Checkbox("IMU auto level", &state.imu_level_enabled);
      const auto available = lidar_viewer::ComputeImuLeveling(state.base_frame.imu, state.imu_mount);
      ImGui::TextWrapped("%s", state.imu_level_enabled ? state.imu_level_result.status.c_str() : available.status.c_str());
      Tip("Align gravity with +Z before ground segmentation. Yaw and sensor height remain adjustable.");
      if (state.base_frame.imu && !state.base_frame.imu->topic.empty()) {
        const auto & imu = *state.base_frame.imu;
        ImGui::TextWrapped("Topic: %s", imu.topic.c_str());
        if (imu.up) ImGui::TextDisabled("Sample offset: %.1f ms", imu.skew_ms);
        ImGui::TextDisabled("Clock: %s received", state.live_mode ? "live" : "bag");
        if (ImGui::CollapsingHeader("IMU mounting rotation")) {
          ImGui::TextWrapped("%s -> %s", imu.imu_frame.c_str(), imu.lidar_frame.c_str());
          if (ImGui::Checkbox("Use profile mount rotation", &state.imu_mount.confirmed)) {
            if (state.imu_mount.confirmed) {
              state.imu_mount.imu_frame = imu.imu_frame;
              state.imu_mount.lidar_frame = imu.lidar_frame;
            }
            reprocess = true;
          }
          ImGui::SetNextItemWidth(-1);
          reprocess |= ImGui::SliderFloat("##ImuMountRoll", &state.imu_mount.roll_deg, -180, 180, "Mount roll: %.1f deg", ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess |= ImGui::SliderFloat("##ImuMountPitch", &state.imu_mount.pitch_deg, -180, 180, "Mount pitch: %.1f deg", ImGuiSliderFlags_AlwaysClamp);
          ImGui::SetNextItemWidth(-1);
          reprocess |= ImGui::SliderFloat("##ImuMountYaw", &state.imu_mount.yaw_deg, -180, 180, "Mount yaw: %.1f deg", ImGuiSliderFlags_AlwaysClamp);
          Tip("Static TF takes priority. Profile rotation is used only for the configured frame pair.");
        }
      }
      if (state.imu_level_result.applied)
        ImGui::TextDisabled("Roll %.2f deg / Pitch %.2f deg", state.imu_level_result.roll_deg, state.imu_level_result.pitch_deg);
      SectionTitle("Display transform");
      ImGui::TextWrapped("Adjust the view frame only. The original dataset is never changed.");
      ImGui::SetNextItemWidth(-1);
      reprocess |= ImGui::SliderFloat("##Yaw", &state.calibration.yaw_deg, -180, 180, "Yaw: %.2f deg", ImGuiSliderFlags_AlwaysClamp);
      ImGui::BeginDisabled(state.imu_level_enabled);
      ImGui::SetNextItemWidth(-1);
      reprocess |= ImGui::SliderFloat("##Pitch", &state.calibration.pitch_deg, -180, 180, "Pitch: %.2f deg", ImGuiSliderFlags_AlwaysClamp);
      ImGui::SetNextItemWidth(-1);
      reprocess |= ImGui::SliderFloat("##Roll", &state.calibration.roll_deg, -180, 180, "Roll: %.2f deg", ImGuiSliderFlags_AlwaysClamp);
      ImGui::EndDisabled();
      ImGui::SetNextItemWidth(-1);
      reprocess |= ImGui::SliderFloat("##Z", &state.calibration.z_m, -5, 5, "Z offset: %.3f m", ImGuiSliderFlags_AlwaysClamp);
      ImGui::TextDisabled("Ctrl+click a slider to enter a value.");
      if (ImGui::Button("Reset alignment", {-1, 0})) {
        state.calibration = {0, 0, 0, 0};
        reprocess = true;
      }
      ImGui::EndTabItem();
    }
    if (begin_tab("ROI")) {
      BuildRoiEditor(state);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  if (!state.ui_error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, .56f, .48f, 1));
    ImGui::TextWrapped("%s", state.ui_error.c_str());
    ImGui::PopStyleColor();
    if (ImGui::SmallButton("Dismiss error")) state.ui_error.clear();
  }
  ImGui::EndChild();
  ImGui::Separator();
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
  const bool unsaved=!lidar_viewer::SameRoi(state.roi,state.saved_roi);
  ImGui::TextColored(unsaved ? ImVec4(1,.74f,.4f,1) : ImVec4(.48f,.65f,.65f,1),
    "%s",unsaved ? "ROI changes not saved  /  Ctrl+S" : "Ctrl+S saves this profile");
  ImGui::End();

  BeginPanel("Transport", layout.transport, false);
  const auto * group = !state.live_mode && has_input ? &state.file_groups[state.current_file_group] : nullptr;
  const std::size_t first_index = group ? group->first_index : 0;
  const std::size_t frame_count = group ? group->last_index - group->first_index + 1 : 0;
  const bool multiple = frame_count > 1;
  ImGui::BeginDisabled(!has_input || state.loading_bag_source.has_value() || (state.live_mode && !state.live_stats.connected));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.15f, .37f, .46f, 1));
  const bool showing_motion = state.live_mode ? !state.live_paused : state.is_playing;
  if (ImGui::Button(showing_motion ? "Pause" : "Play", {88.f * s, 0})) {
    if (state.live_mode) SetLivePaused(state, !state.live_paused);
    else {state.is_playing = !state.is_playing; state.last_frame_advance_time = glfwGetTime();}
  }
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::BeginDisabled(!multiple);
  if (ImGui::Button("<")) {state.is_playing = false; AdvanceFrame(state, -1);}
  Tip("Previous frame  [P]");
  ImGui::SameLine();
  if (ImGui::Button(">")) {state.is_playing = false; AdvanceFrame(state, 1);}
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
    else ImGui::TextDisabled("Frame %zu / %zu", has_input ? state.current_index - first_index + 1 : 0, frame_count);
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
  if (ImGui::IsItemActivated()) {
    state.timeline_dragging = true;
    state.is_playing = false;
  }
  if (ImGui::IsItemDeactivated()) {
    if (ImGui::IsItemDeactivatedAfterEdit())
      TryLoadFrame(state, first_index + static_cast<std::size_t>(state.seek_draft));
    state.timeline_dragging = false;
  }
  ImGui::EndDisabled();
  }
  BuildFolderPicker(state);
  if (state.open_live_dialog) {ImGui::OpenPopup("Live sensors");state.open_live_dialog=false;}
  BuildLiveDialog(state);
  ImGui::End();

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16.f * s, 4.f * s});
  BeginPanel("Status", layout.status, false);
  ImGui::PushClipRect(
    {layout.status.x, layout.status.y},
    {layout.status.w > 700.f * s ? layout.status.w - 380.f * s : layout.status.w, layout.status.y + layout.status.h}, true);
  if (!state.ui_error.empty()) ImGui::TextColored({1.f, .56f, .48f, 1}, "%s", state.ui_error.c_str());
  else if (state.loading_bag_source) ImGui::TextColored({.44f, .76f, 1.f, 1}, "Opening %s...",
    state.bag_sources[*state.loading_bag_source].display_name.c_str());
  else if (!state.transient_status.empty()) ImGui::TextColored({.44f, .86f, .68f, 1}, "%s", state.transient_status.c_str());
  else if (state.live_mode) ImGui::TextDisabled("%zu points | display %llu / received %llu | %s", state.frame.points.size(),
    static_cast<unsigned long long>(state.live_display_sequence), static_cast<unsigned long long>(state.live_stats.clouds),
    !state.live_stats.connected ? "disconnected" : (state.live_paused ? "frozen" : (state.live_stats.cloud_age_ms > 1000 ? "stream stale" : "live")));
  else if (!has_input) ImGui::TextDisabled("Open a file, drop a bag, or connect live sensors.");
  else ImGui::TextDisabled("%zu points   |   %s", state.frame.points.size(), state.frames[state.current_index].label.c_str());
  ImGui::PopClipRect();
  if (layout.status.w > 700.f * s) {
    ImGui::SameLine(std::max(450.f * s, layout.status.w - 360.f * s));
    ImGui::TextDisabled("Process %.1f ms  |  ROI %.1f ms  |  %.0f fps",state.processing_ms,state.roi_ms,state.current_view_fps);
  }
  ImGui::End();
  ImGui::PopStyleVar();

  const auto & view=layout.viewport;
  if (view.w>260*s && view.h>100*s) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{8*s,6*s});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6*s,0});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8*s,5*s});
    BeginPanel("Canvas tools",{view.x+12*s,view.y+10*s,std::min(view.w-24*s,414*s),44*s},false);
    const bool top=state.camera.projection_mode==CameraState::ProjectionMode::View2D;
    const auto tool_button=[&](const char * label,bool selected) {
      if (selected) ImGui::PushStyleColor(ImGuiCol_Button,{.14f,.37f,.45f,1});
      const bool clicked=ImGui::Button(label);
      if (selected) ImGui::PopStyleColor();
      return clicked;
    };
    if (tool_button("3D",!top)) {SelectRoiTool(state,ViewerState::RoiTool::Navigate);SetProjectionMode(state,CameraState::ProjectionMode::View3D);}
    ImGui::SameLine();
    if (tool_button("2D",top)) SetProjectionMode(state,CameraState::ProjectionMode::View2D);
    ImGui::SameLine();
    if (ImGui::Button("Fit")) state.fit_view_requested=true;
    ImGui::SameLine();
    if (tool_button("Draw box",state.roi_tool==ViewerState::RoiTool::Draw)) SelectRoiTool(state,ViewerState::RoiTool::Draw);
    Tip("Drag a box footprint in 2D. R. Esc cancels.");
    ImGui::SameLine(); ImGui::BeginDisabled(state.roi.boxes.empty());
    if (tool_button("Move",state.roi_tool==ViewerState::RoiTool::Move)) SelectRoiTool(state,ViewerState::RoiTool::Move);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (tool_button("Navigate",state.roi_tool==ViewerState::RoiTool::Navigate)) SelectRoiTool(state,ViewerState::RoiTool::Navigate);
    ImGui::End();ImGui::PopStyleVar(3);
  }
  auto * draw=ImGui::GetBackgroundDrawList();
  const ImU32 muted = IM_COL32(131, 151, 174, 255);
  draw->AddText({view.x + 14.f * s, view.y + 64.f * s}, muted,
    state.camera.projection_mode == CameraState::ProjectionMode::View2D ? "TOP VIEW  /  XY" : "PERSPECTIVE  /  XYZ");
  if (!has_input && !ImGui::IsPopupOpen(nullptr,ImGuiPopupFlags_AnyPopup) && view.w>300*s && view.h>230*s) {
    const float width=std::min(460*s,view.w-32*s);
    BeginPanel("Welcome",{view.x+(view.w-width)*.5f,view.y+(view.h-190*s)*.5f,width,190*s},false);
    ImGui::TextColored({.78f,.9f,.96f,1},"%s",state.live_mode ? "Waiting for LiDAR" :
      (state.loading_bag_source ? "Opening point cloud..." : "Start inspecting your LiDAR data"));
    ImGui::Spacing();
    ImGui::TextWrapped("%s",state.live_mode ? state.live_points_topic.data() :
      "Open a BIN file or ROS bag. You can also drop a file or a single bag folder here.");
    ImGui::Spacing();
    ImGui::BeginDisabled(state.live.Active() || state.loading_bag_source.has_value() || state.bag_picker.Active());
    if (ImGui::Button("Open file",{135*s,0})) StartBagPicker(state);
    ImGui::EndDisabled();ImGui::SameLine();
    if (ImGui::Button(state.live_mode ? "Live settings" : "Connect live",{135*s,0})) state.open_live_dialog=true;
    ImGui::End();
  }
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
  if (has_input && state.ground_removal_enabled) {
    draw->AddCircleFilled({view.x + 18.f * s, view.y + view.h - 18.f * s}, 4.f * s, IM_COL32(255, 75, 65, 255));
    draw->AddText({view.x + 30.f * s, view.y + view.h - 26.f * s}, muted, "Ground");
  }
  if (has_input && state.roi.enabled) {
    const auto text = "ROI: " + std::to_string(state.frame.points.size()) + " / " +
      std::to_string(state.base_frame.points.size()) + " points";
    draw->AddText({view.x + 14.f * s, view.y + (state.imu_level_enabled ? 108.f : 86.f) * s},
      IM_COL32(105, 215, 225, 255), text.c_str());
  }
  if (state.roi_tool!=ViewerState::RoiTool::Navigate) {
    draw->AddText({view.x+14*s,view.y+view.h-28*s},IM_COL32(125,225,230,255),
      state.roi_tool==ViewerState::RoiTool::Draw ? "Drag to create a box  |  Esc cancels" : "Drag a box to move it  |  Esc returns to navigation");
  }
  if (state.roi_gesture.active && !state.roi.boxes.empty()) {
    const auto & box=state.roi.boxes[state.selected_roi];
    char label[160];std::snprintf(label,sizeof(label),"%.2f x %.2f x %.2f m  |  release to apply",box.size.x,box.size.y,box.size.z);
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
    bounded(ground.candidate_grid_cell_m, .1f, 5, .8f);
    bounded(state.camera.point_size, 1, 8, 2);
    bounded(state.grid_settings.spacing, .5f, 50, 2);
    bounded(state.grid_settings.thickness, 1, 6, 1);
    bounded(state.ui_scale, .75f, 1.5f, 1);
  }
  if (reprocess) {
    try { ApplyCalibration(state); state.ui_error.clear(); }
    catch (const std::exception & error) { state.is_playing = false; state.ui_error = error.what(); }
  } else if (state.roi_pending && !state.roi_gesture.active && !ImGui::IsAnyItemActive()) {
    try {
      ApplyRoiResult(state,state.processed_frame,state.processed_obstacles,state.roi_frame);
      state.ui_error.clear();
      std::cout<<"ROI applied: "<<state.roi.boxes.size()<<" boxes | "<<state.frame.points.size()<<" points | "
        <<state.roi_ms<<" ms | ground passes "<<state.ground_passes<<std::endl;
    } catch (const std::exception & error) {state.roi_pending=false;state.ui_error=error.what();}
  }
}

void RenderRoiBoxes(ViewerState & state, const Mat4 & mvp)
{
  auto & renderer = state.roi_renderer;
  if (!state.roi.show_boxes || renderer.vertices.empty()) return;
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
  for (std::size_t i = 0; i < std::min(state.roi.boxes.size(), renderer.vertices.size() / 24); ++i) {
    if (i == state.selected_roi) glUniform4f(state.grid.color_loc, 1.f, .8f, .25f, .95f);
    else if (state.roi.enabled && state.roi.boxes[i].enabled) glUniform4f(state.grid.color_loc, .25f, .85f, .95f, .85f);
    else glUniform4f(state.grid.color_loc, .55f, .6f, .65f, .5f);
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
  state.cluster_boxes.dirty = false;
}

void RenderClusterBoxes(ViewerState & state, const Mat4 & mvp)
{
  if (
    (!state.algorithm_config.clustering.enabled && !state.algorithm_config.detection.enabled) ||
    state.cluster_boxes.batches.empty())
  {
    return;
  }

  if (state.cluster_boxes.dirty) {
    UploadClusterBoxes(state);
  }

  glUseProgram(state.grid.program);
  glUniformMatrix4fv(state.grid.mvp_loc, 1, GL_FALSE, mvp.m.data());
  glBindVertexArray(state.cluster_boxes.vao);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  for (const auto & batch : state.cluster_boxes.batches) {
    if (!batch.fill_vertices.empty()) {
      glDepthMask(GL_FALSE);
      glBindBuffer(GL_ARRAY_BUFFER, state.cluster_boxes.fill_vbo);
      glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(batch.fill_vertices.size() * sizeof(Vec3)),
        batch.fill_vertices.data(),
        GL_DYNAMIC_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), reinterpret_cast<void *>(0));
      glUniform4f(
        state.grid.color_loc,
        batch.fill_color.r,
        batch.fill_color.g,
        batch.fill_color.b,
        batch.fill_color.a);
      glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(batch.fill_vertices.size()));
      glDepthMask(GL_TRUE);
    }

    if (!batch.line_vertices.empty()) {
      glBindBuffer(GL_ARRAY_BUFFER, state.cluster_boxes.line_vbo);
      glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(batch.line_vertices.size() * sizeof(Vec3)),
        batch.line_vertices.data(),
        GL_DYNAMIC_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), reinterpret_cast<void *>(0));
      glUniform4f(
        state.grid.color_loc,
        batch.line_color.r,
        batch.line_color.g,
        batch.line_color.b,
        batch.line_color.a);
      glLineWidth(2.0f);
      glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(batch.line_vertices.size()));
    }
  }

  glLineWidth(1.0f);
  glDisable(GL_BLEND);
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
        std::cout << "Usage: lidar_viewer [INPUT] [--live] [--topic /b2/points] [--sensor-height METRES] [--imu-level] [--imu-topic TOPIC] [--2d] [--play] [--inspect] [--record-folder DIR]\n"
                  << "INPUT: .bin file/directory, bag collection folder, ROS 2 bag, .mcap, .db3 or metadata.yaml.\n"
                  << "Without INPUT, show an empty viewer. Click Open file to select a recording.\n"
                  << "Open file (Ctrl+O) selects another recording; Add folder remembers bag collections.\n"
                  << "--live: subscribe to live LiDAR/IMU. Pause freezes the display; Play returns to the latest frame.\n"
                  << "--record-folder: parent folder for recordings started explicitly with Record.\n"
                  << "--sensor-height: explicit display Z translation (0..5 m); no automatic bag leveling.\n"
                  << "--imu-level: gravity-align roll/pitch using recorded IMU and a known mount rotation.\n"
                  << "--imu-topic: select an IMU topic; otherwise choose one from recorded topics.\n"
                  << "--2d: open top view; right-drag or Q/E rotates, left-drag pans.\n"
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
    state.bag_sensor_height_override = sensor_height;
    try {
      state.data_folders_config = DataFoldersConfigPath();
      state.data_folders = LoadBagFolders(state.data_folders_config);
      for (const auto & folder : state.data_folders)
        state.bag_sources = MergeBagSources(std::move(state.bag_sources), DiscoverBagSources(folder));
    } catch (const std::exception & error) {
      state.ui_error = std::string("Saved data folders could not be loaded: ") + error.what();
    }
    const bool load_initial_input = input_selected || inspect;
    if (input_selected) {
      const auto supplied = DiscoverBagSources(input_path);
      if (!supplied.empty()) {
        input_path = supplied.front().path;
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
    state.dataset_profile = load_initial_input ? DetectDatasetProfile(input_path) : "rosbag";
    state.algorithm_config = lidar_viewer::LoadPipelineConfig(AlgorithmsConfigPath());
    LoadSettings(state);
    if (imu_level) state.imu_level_enabled = true;
    if (start_2d) SetProjectionMode(state, CameraState::ProjectionMode::View2D);
    if (sensor_height) {
      state.calibration.z_m = *sensor_height;
    }
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
    } else if (load_initial_input) {
      state.frames = ResolveInputFrames(input_path, bag_topic, imu_topic);
    }
    state.input_fps_estimate = EstimateInputFps(state);
    SyncPlaybackFpsToInput(state);
    BuildFileGroups(state);
    if (!state.frames.empty()) LoadFrameByIndex(state, 0);
    state.last_frame_advance_time = 0.0;
    state.is_playing = autoplay && !state.frames.empty();
    if (state.frames.empty()) {
      std::cout << "No bag loaded. Click Open file to select a recording." << std::endl;
    } else {
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
      for (const auto & source : state.bag_sources) std::cout << "  " << BagLabel(source) << '\n';
    }
    if (inspect) {
      std::cout << "ROI: " << (state.roi.enabled ? "on" : "off") << " | " << state.roi.boxes.size()
                << " boxes | retained " << state.frame.points.size() << " / " << state.base_frame.points.size() << " points\n";
      if (state.imu_level_enabled) std::cout << "IMU leveling: " << state.imu_level_result.status << '\n';
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

    GLFWwindow * window = glfwCreateWindow(kWindowWidth, kWindowHeight, "LiDAR Studio", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }

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

    if (start_live) ConnectLive(state);

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      CompleteBagPicker(state, window);
      CompleteBagLoad(state);
      UpdateLiveFrame(state);
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
          static_cast<float>(fb_width) / window_width, static_cast<float>(fb_height) / window_height));
      ImGui::GetStyle() = base_ui_style;
      ImGui::GetStyle().ScaleAllSizes(state.ui_layout.scale);
      ImGui::GetIO().FontGlobalScale = state.ui_layout.scale / state.font_raster_scale;
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      BuildViewerInterface(state, window);
      const double frame_begin_time = glfwGetTime();
      bool advanced_frame = false;

      if (!state.transient_status.empty() && glfwGetTime() >= state.transient_status_until) {
        state.transient_status.clear();
      }

      if (state.is_playing && state.frames.size() > 1) {
        const double now_time = glfwGetTime();
        const double frame_interval = 1.0 / std::max(0.1, state.playback_fps);
        if ((now_time - state.last_frame_advance_time) >= frame_interval) {
          AdvanceFrame(state, 1);
          state.last_frame_advance_time = now_time;
          advanced_frame = true;
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
            -std::max(state.camera.distance, 2.0f) * aspect,
            std::max(state.camera.distance, 2.0f) * aspect,
            -std::max(state.camera.distance, 2.0f),
            std::max(state.camera.distance, 2.0f),
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
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(state.frame.points.size()));
      RenderGrid(state, mvp);
      RenderClusterBoxes(state, mvp);
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
    }

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
