#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "lidar_viewer/data_io.hpp"
#include "lidar_viewer/config.hpp"
#include "lidar_viewer/algorithms/pipeline.hpp"
#include "lidar_viewer/algorithms/ground_removal.hpp"
#include "lidar_viewer/app.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
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
constexpr float kHudPanelX = 18.0f;
constexpr float kHudPanelY = 18.0f;
constexpr float kHudPanelW = 396.0f;
constexpr float kHudPanelH = 430.0f;
constexpr float kPlayButtonX = kHudPanelX + 16.0f;
constexpr float kPlayButtonY = kHudPanelY + 62.0f;
constexpr float kPlayButtonW = 70.0f;
constexpr float kPlayButtonH = 18.0f;
constexpr float kPauseButtonX = kHudPanelX + 92.0f;
constexpr float kPauseButtonY = kHudPanelY + 62.0f;
constexpr float kPauseButtonW = 74.0f;
constexpr float kPauseButtonH = 18.0f;
constexpr float kGroundButtonX = kHudPanelX + 176.0f;
constexpr float kGroundButtonY = kHudPanelY + 62.0f;
constexpr float kGroundButtonW = 72.0f;
constexpr float kGroundButtonH = 18.0f;
constexpr float kClusterButtonX = kHudPanelX + 254.0f;
constexpr float kClusterButtonY = kHudPanelY + 62.0f;
constexpr float kClusterButtonW = 46.0f;
constexpr float kClusterButtonH = 18.0f;
constexpr float kDetectButtonX = kHudPanelX + 306.0f;
constexpr float kDetectButtonY = kHudPanelY + 62.0f;
constexpr float kDetectButtonW = 56.0f;
constexpr float kDetectButtonH = 18.0f;
constexpr float kView2DButtonX = kHudPanelX + 16.0f;
constexpr float kView2DButtonY = kHudPanelY + 84.0f;
constexpr float kView2DButtonW = 54.0f;
constexpr float kView2DButtonH = 18.0f;
constexpr float kView3DButtonX = kHudPanelX + 76.0f;
constexpr float kView3DButtonY = kHudPanelY + 84.0f;
constexpr float kView3DButtonW = 54.0f;
constexpr float kView3DButtonH = 18.0f;
constexpr float kGridPanelX = 18.0f;
constexpr float kGridPanelY = 464.0f;
constexpr float kGridPanelW = 396.0f;
constexpr float kGridPanelH = 356.0f;
constexpr const char * kSettingsFileName = "lidar_viewer_settings.cfg";
constexpr const char * kProfilesDir = "config/profiles";
constexpr float kTimelineX = 380.0f;
constexpr float kTimelineY = 18.0f;
constexpr float kTimelineW = 980.0f;
constexpr float kTimelineH = 66.0f;
constexpr float kTimelineTrackX = kTimelineX + 18.0f;
constexpr float kTimelineTrackY = kTimelineY + 42.0f;
constexpr float kTimelineTrackW = 700.0f;
constexpr float kTimelineTrackH = 6.0f;
constexpr float kFileSelectorX = kTimelineX + 744.0f;
constexpr float kFileSelectorY = kTimelineY + 24.0f;
constexpr float kFileSelectorW = 218.0f;
constexpr float kFileSelectorH = 22.0f;
constexpr float kFileSelectorListRowH = 18.0f;
constexpr int kFileSelectorVisibleRows = 6;
constexpr const char * kAlgorithmsConfigPath = "config/algorithms.yaml";
constexpr const char * kDefaultInputPath = "/home/juno/lidar/dataset/gangnam/2019_10_04_12_28_25.bin";

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
      frag_color = vec4(0.92, 0.95, 0.98, 1.0);
      return;

      float value = v_intensity;
      if (u_color_mode == 1) {
        float denom = max(1e-6, u_max_height - u_min_height);
        value = (v_height - u_min_height) / denom;
      }
      frag_color = vec4(turbo(value), 1.0);
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

GLuint CreateUiProgram()
{
  constexpr const char * kVertexShader = R"(
    #version 330 core
    layout(location = 0) in vec2 a_position;
    layout(location = 1) in vec4 a_color;

    uniform vec2 u_viewport;
    out vec4 v_color;

    void main() {
      float x = (a_position.x / u_viewport.x) * 2.0 - 1.0;
      float y = 1.0 - (a_position.y / u_viewport.y) * 2.0;
      gl_Position = vec4(x, y, 0.0, 1.0);
      v_color = a_color;
    }
  )";

  constexpr const char * kFragmentShader = R"(
    #version 330 core
    in vec4 v_color;
    out vec4 frag_color;

    void main() {
      frag_color = v_color;
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
    throw std::runtime_error("UI program link failed:\n" + log);
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

struct UiVertex
{
  float x;
  float y;
  float r;
  float g;
  float b;
  float a;
};

struct Color
{
  float r;
  float g;
  float b;
  float a;
};

struct UiRenderer
{
  GLuint program{0};
  GLuint vao{0};
  GLuint vbo{0};
  GLint viewport_loc{-1};
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
  float yaw_deg{1.2f};
  float pitch_deg{-1.8f};
  float roll_deg{-0.7f};
  float z_m{2.0f};
};

struct GridButton
{
  float x;
  float y;
  float w;
  float h;
};

using GlyphRows = std::array<const char *, 7>;

GlyphRows GlyphFor(char ch)
{
  switch (ch) {
    case 'A': return {{{"01110"}, {"10001"}, {"10001"}, {"11111"}, {"10001"}, {"10001"}, {"10001"}}};
    case 'B': return {{{"11110"}, {"10001"}, {"10001"}, {"11110"}, {"10001"}, {"10001"}, {"11110"}}};
    case 'C': return {{{"01110"}, {"10001"}, {"10000"}, {"10000"}, {"10000"}, {"10001"}, {"01110"}}};
    case 'D': return {{{"11110"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"11110"}}};
    case 'E': return {{{"11111"}, {"10000"}, {"10000"}, {"11110"}, {"10000"}, {"10000"}, {"11111"}}};
    case 'F': return {{{"11111"}, {"10000"}, {"10000"}, {"11110"}, {"10000"}, {"10000"}, {"10000"}}};
    case 'G': return {{{"01110"}, {"10001"}, {"10000"}, {"10111"}, {"10001"}, {"10001"}, {"01111"}}};
    case 'H': return {{{"10001"}, {"10001"}, {"10001"}, {"11111"}, {"10001"}, {"10001"}, {"10001"}}};
    case 'I': return {{{"11111"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}, {"11111"}}};
    case 'J': return {{{"00111"}, {"00010"}, {"00010"}, {"00010"}, {"10010"}, {"10010"}, {"01100"}}};
    case 'K': return {{{"10001"}, {"10010"}, {"10100"}, {"11000"}, {"10100"}, {"10010"}, {"10001"}}};
    case 'L': return {{{"10000"}, {"10000"}, {"10000"}, {"10000"}, {"10000"}, {"10000"}, {"11111"}}};
    case 'M': return {{{"10001"}, {"11011"}, {"10101"}, {"10101"}, {"10001"}, {"10001"}, {"10001"}}};
    case 'N': return {{{"10001"}, {"11001"}, {"10101"}, {"10011"}, {"10001"}, {"10001"}, {"10001"}}};
    case 'O': return {{{"01110"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"01110"}}};
    case 'P': return {{{"11110"}, {"10001"}, {"10001"}, {"11110"}, {"10000"}, {"10000"}, {"10000"}}};
    case 'Q': return {{{"01110"}, {"10001"}, {"10001"}, {"10001"}, {"10101"}, {"10010"}, {"01101"}}};
    case 'R': return {{{"11110"}, {"10001"}, {"10001"}, {"11110"}, {"10100"}, {"10010"}, {"10001"}}};
    case 'S': return {{{"01111"}, {"10000"}, {"10000"}, {"01110"}, {"00001"}, {"00001"}, {"11110"}}};
    case 'T': return {{{"11111"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}}};
    case 'U': return {{{"10001"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"01110"}}};
    case 'V': return {{{"10001"}, {"10001"}, {"10001"}, {"10001"}, {"10001"}, {"01010"}, {"00100"}}};
    case 'W': return {{{"10001"}, {"10001"}, {"10001"}, {"10101"}, {"10101"}, {"10101"}, {"01010"}}};
    case 'X': return {{{"10001"}, {"10001"}, {"01010"}, {"00100"}, {"01010"}, {"10001"}, {"10001"}}};
    case 'Y': return {{{"10001"}, {"10001"}, {"01010"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}}};
    case 'Z': return {{{"11111"}, {"00001"}, {"00010"}, {"00100"}, {"01000"}, {"10000"}, {"11111"}}};
    case '0': return {{{"01110"}, {"10001"}, {"10011"}, {"10101"}, {"11001"}, {"10001"}, {"01110"}}};
    case '1': return {{{"00100"}, {"01100"}, {"00100"}, {"00100"}, {"00100"}, {"00100"}, {"01110"}}};
    case '2': return {{{"01110"}, {"10001"}, {"00001"}, {"00010"}, {"00100"}, {"01000"}, {"11111"}}};
    case '3': return {{{"11110"}, {"00001"}, {"00001"}, {"01110"}, {"00001"}, {"00001"}, {"11110"}}};
    case '4': return {{{"00010"}, {"00110"}, {"01010"}, {"10010"}, {"11111"}, {"00010"}, {"00010"}}};
    case '5': return {{{"11111"}, {"10000"}, {"10000"}, {"11110"}, {"00001"}, {"00001"}, {"11110"}}};
    case '6': return {{{"01110"}, {"10000"}, {"10000"}, {"11110"}, {"10001"}, {"10001"}, {"01110"}}};
    case '7': return {{{"11111"}, {"00001"}, {"00010"}, {"00100"}, {"01000"}, {"01000"}, {"01000"}}};
    case '8': return {{{"01110"}, {"10001"}, {"10001"}, {"01110"}, {"10001"}, {"10001"}, {"01110"}}};
    case '9': return {{{"01110"}, {"10001"}, {"10001"}, {"01111"}, {"00001"}, {"00001"}, {"01110"}}};
    case '[': return {{{"01110"}, {"01000"}, {"01000"}, {"01000"}, {"01000"}, {"01000"}, {"01110"}}};
    case ']': return {{{"01110"}, {"00010"}, {"00010"}, {"00010"}, {"00010"}, {"00010"}, {"01110"}}};
    case '/': return {{{"00001"}, {"00010"}, {"00100"}, {"00100"}, {"01000"}, {"10000"}, {"00000"}}};
    case '.': return {{{"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00110"}, {"00110"}}};
    case ':': return {{{"00000"}, {"00110"}, {"00110"}, {"00000"}, {"00110"}, {"00110"}, {"00000"}}};
    case '-': return {{{"00000"}, {"00000"}, {"00000"}, {"01110"}, {"00000"}, {"00000"}, {"00000"}}};
    case '_': return {{{"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}, {"11111"}}};
    case ' ': return {{{"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}, {"00000"}}};
    default: return {{{"11111"}, {"00001"}, {"00010"}, {"00100"}, {"00100"}, {"00000"}, {"00100"}}};
  }
}

void AppendRect(
  std::vector<UiVertex> & vertices,
  float x0,
  float y0,
  float x1,
  float y1,
  Color color)
{
  vertices.push_back({x0, y0, color.r, color.g, color.b, color.a});
  vertices.push_back({x1, y0, color.r, color.g, color.b, color.a});
  vertices.push_back({x1, y1, color.r, color.g, color.b, color.a});
  vertices.push_back({x0, y0, color.r, color.g, color.b, color.a});
  vertices.push_back({x1, y1, color.r, color.g, color.b, color.a});
  vertices.push_back({x0, y1, color.r, color.g, color.b, color.a});
}

std::string ToUpperCopy(std::string text)
{
  for (char & ch : text) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  return text;
}

std::string EllipsizeMiddle(const std::string & text, std::size_t max_chars)
{
  if (text.size() <= max_chars) {
    return text;
  }
  if (max_chars <= 3) {
    return text.substr(0, max_chars);
  }
  const std::size_t left = (max_chars - 3) / 2;
  const std::size_t right = max_chars - 3 - left;
  return text.substr(0, left) + "..." + text.substr(text.size() - right);
}

std::string FileDisplayName(const fs::path & path)
{
  if (path.has_filename()) {
    return path.filename().string();
  }
  return path.string();
}

void AppendGlyph(
  std::vector<UiVertex> & vertices,
  float x,
  float y,
  float scale,
  char ch,
  Color color)
{
  const GlyphRows glyph = GlyphFor(ch);
  for (std::size_t row = 0; row < glyph.size(); ++row) {
    for (int col = 0; col < 5; ++col) {
      if (glyph[row][col] != '1') {
        continue;
      }
      const float x0 = x + static_cast<float>(col) * scale;
      const float y0 = y + static_cast<float>(row) * scale;
      AppendRect(vertices, x0, y0, x0 + scale, y0 + scale, color);
    }
  }
}

void AppendText(
  std::vector<UiVertex> & vertices,
  float x,
  float y,
  float scale,
  const std::string & text,
  Color color)
{
  float cursor = x;
  for (char ch : ToUpperCopy(text)) {
    AppendGlyph(vertices, cursor, y, scale, ch, color);
    cursor += scale * 6.0f;
  }
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
  struct FileGroup
  {
    fs::path path;
    std::string display_name;
    std::size_t first_index{0};
    std::size_t last_index{0};
  };

  std::vector<FrameHandle> frames;
  std::vector<FileGroup> file_groups;
  std::size_t current_index{0};
  std::size_t current_file_group{0};
  FrameData base_frame;
  FrameData frame;
  bool alignment_initialized{false};
  float global_ground_offset{0.0f};
  CameraState camera;
  GLuint vao{0};
  GLuint vbo{0};
  bool frame_dirty{false};
  bool fit_view_requested{false};
  int color_mode{0};
  bool is_playing{false};
  bool timeline_dragging{false};
  bool file_selector_open{false};
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
  UiRenderer ui;
  GridRenderer grid;
  ClusterBoxRenderer cluster_boxes;
  GridSettings grid_settings;
  CalibrationSettings calibration;
  lidar_viewer::algorithms::PipelineConfig algorithm_config;
  std::vector<lidar_viewer::algorithms::Cluster> clusters;
  std::vector<lidar_viewer::algorithms::Detection> detections;
  std::string dataset_profile{"default"};
  int active_slider{-1};
  int active_value_box{-1};
  std::string value_input_buffer;
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
    state.camera.pitch = -0.35f;
    state.camera.yaw = 0.2f;
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
    state.camera.pitch = -0.35f;
    state.camera.yaw = 0.2f;
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

void AppendOrientedBox(
  const lidar_viewer::algorithms::OrientedBox3D & bounds,
  std::vector<Vec3> & fill_vertices,
  std::vector<Vec3> & line_vertices)
{
  if (!bounds.valid) {
    return;
  }

  const Vec3 dx = bounds.axis_x * bounds.half_length;
  const Vec3 dy = bounds.axis_y * bounds.half_width;
  const Vec3 center_low{bounds.center.x, bounds.center.y, bounds.min_z};
  const Vec3 center_high{bounds.center.x, bounds.center.y, bounds.max_z};

  const Vec3 p000 = center_low - dx - dy;
  const Vec3 p100 = center_low + dx - dy;
  const Vec3 p110 = center_low + dx + dy;
  const Vec3 p010 = center_low - dx + dy;
  const Vec3 p001 = center_high - dx - dy;
  const Vec3 p101 = center_high + dx - dy;
  const Vec3 p111 = center_high + dx + dy;
  const Vec3 p011 = center_high - dx + dy;

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

std::string DetectionClassLabel(int class_id)
{
  switch (class_id) {
    case lidar_viewer::algorithms::DetectionClassCar:
      return "CAR";
    case lidar_viewer::algorithms::DetectionClassBus:
      return "BUS";
    case lidar_viewer::algorithms::DetectionClassTruck:
      return "TRUCK";
    case lidar_viewer::algorithms::DetectionClassPerson:
      return "PERSON";
    case lidar_viewer::algorithms::DetectionClassCyclist:
      return "CYCLIST";
    case lidar_viewer::algorithms::DetectionClassTree:
      return "TREE";
    default:
      return "UNKNOWN";
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
  if (frame.points.empty()) {
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

  for (auto & p : frame.points) {
    p.z -= ground_offset;
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
  frame.ground_z -= ground_offset;
}

void ApplyCalibration(ViewerState & state)
{
  state.frame = state.base_frame;
  if (state.frame.points.empty()) {
    return;
  }

  const CalibrationTransform transform = BuildCalibrationTransform(state.calibration);
  for (auto & p : state.frame.points) {
    const Vec3 rotated = ApplyCalibrationTransform({p.x, p.y, p.z}, transform);
    p.x = rotated.x;
    p.y = rotated.y;
    p.z = rotated.z;
    p.intensity = 0.0f;
  }

  state.ground_removed_count = 0;
  if (state.ground_removal_enabled) {
    auto params = state.algorithm_config.ground_removal;
    params.enabled = true;
    auto result = lidar_viewer::algorithms::RunGroundRemovalPlaceholder(state.frame.points, params);
    state.ground_removed_count = result.ground_points.size();
    state.clusters = lidar_viewer::algorithms::RunClusteringPlaceholder(
      result.non_ground_points,
      state.algorithm_config.clustering);
    state.detections = lidar_viewer::algorithms::RunDetectionPlaceholder(
      result.non_ground_points,
      state.clusters,
      state.algorithm_config.detection);
    state.frame.points.clear();
    state.frame.points.reserve(result.non_ground_points.size() + result.ground_points.size());
    state.frame.points.insert(
      state.frame.points.end(),
      result.non_ground_points.begin(),
      result.non_ground_points.end());
    for (auto point : result.ground_points) {
      point.intensity = -1.0f;
      state.frame.points.push_back(point);
    }
  } else {
    state.clusters = lidar_viewer::algorithms::RunClusteringPlaceholder(
      state.frame.points,
      state.algorithm_config.clustering);
    state.detections = lidar_viewer::algorithms::RunDetectionPlaceholder(
      state.frame.points,
      state.clusters,
      state.algorithm_config.detection);
  }

  RebuildClusterBoxes(state);
  RecomputeFrameBounds(state.frame);
  state.frame_dirty = true;
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
  state.current_index = index % state.frames.size();
  UpdateCurrentFileGroup(state);
  ConsumeReadyPrefetch(state);
  const auto cache_it = state.frame_cache.find(state.current_index);
  if (cache_it != state.frame_cache.end()) {
    state.base_frame = cache_it->second;
  } else if (state.prefetched_index.has_value() && *state.prefetched_index == state.current_index && state.prefetch_future.valid()) {
    state.base_frame = state.prefetch_future.get();
    StoreFrameInCache(state, state.current_index, state.base_frame);
    state.prefetched_index.reset();
  } else {
    state.base_frame = LoadFrame(state.frames[state.current_index]);
    StoreFrameInCache(state, state.current_index, state.base_frame);
  }
  if (!state.alignment_initialized) {
    state.global_ground_offset = (state.frames[state.current_index].format == InputFormat::OusterLegacyContainer)
      ? 0.0f
      : state.base_frame.ground_z;
    state.alignment_initialized = true;
  }
  ApplyGlobalAlignment(state.base_frame, state.global_ground_offset);
  ApplyCalibration(state);
  QueueNextFramePrefetch(state);
  // std::cout << "Loaded " << state.frames[state.current_index].label << " (" << state.base_frame.points.size() << " points)\n";
}

bool IsInsideRect(double x, double y, float rx, float ry, float rw, float rh)
{
  return x >= rx && x <= (rx + rw) && y >= ry && y <= (ry + rh);
}

GridButton GridToggleButton()
{
  return {kGridPanelX + 16.0f, kGridPanelY + 34.0f, 78.0f, 18.0f};
}

GridButton GridMinusButton(int row)
{
  return {kGridPanelX + 148.0f, kGridPanelY + 34.0f + 26.0f * static_cast<float>(row), 18.0f, 16.0f};
}

GridButton GridPlusButton(int row)
{
  return {kGridPanelX + 246.0f, kGridPanelY + 34.0f + 26.0f * static_cast<float>(row), 18.0f, 16.0f};
}

GridButton GridSaveButton()
{
  return {kGridPanelX + kGridPanelW - 96.0f, kGridPanelY + kGridPanelH - 28.0f, 78.0f, 18.0f};
}

GridButton GroundRemovalButton()
{
  return {kGroundButtonX, kGroundButtonY, kGroundButtonW, kGroundButtonH};
}

GridButton ClusteringButton()
{
  return {kClusterButtonX, kClusterButtonY, kClusterButtonW, kClusterButtonH};
}

GridButton DetectionButton()
{
  return {kDetectButtonX, kDetectButtonY, kDetectButtonW, kDetectButtonH};
}

GridButton View2DButton()
{
  return {kView2DButtonX, kView2DButtonY, kView2DButtonW, kView2DButtonH};
}

GridButton View3DButton()
{
  return {kView3DButtonX, kView3DButtonY, kView3DButtonW, kView3DButtonH};
}

GridButton CalibrationSliderRect(int row)
{
  return {kGridPanelX + 86.0f, kGridPanelY + 152.0f + 28.0f * static_cast<float>(row), 126.0f, 14.0f};
}

GridButton CalibrationValueBoxRect(int row)
{
  return {kGridPanelX + 224.0f, kGridPanelY + 150.0f + 28.0f * static_cast<float>(row), 70.0f, 18.0f};
}

GridButton PointSizeSliderRect()
{
  return {kGridPanelX + 86.0f, kGridPanelY + 270.0f, 126.0f, 14.0f};
}

GridButton PointSizeValueBoxRect()
{
  return {kGridPanelX + 224.0f, kGridPanelY + 268.0f, 70.0f, 18.0f};
}

GridButton GroundCellSizeSliderRect()
{
  return {kGridPanelX + 86.0f, kGridPanelY + 294.0f, 126.0f, 14.0f};
}

GridButton GroundCellSizeValueBoxRect()
{
  return {kGridPanelX + 224.0f, kGridPanelY + 292.0f, 70.0f, 18.0f};
}

GridButton TimelineTrackRect()
{
  return {kTimelineTrackX, kTimelineTrackY - 6.0f, kTimelineTrackW, 18.0f};
}

GridButton FileSelectorRect()
{
  return {kFileSelectorX, kFileSelectorY, kFileSelectorW, kFileSelectorH};
}

GridButton FileSelectorListRect()
{
  return {
    kFileSelectorX,
    kFileSelectorY + kFileSelectorH + 4.0f,
    kFileSelectorW,
    static_cast<float>(kFileSelectorVisibleRows) * kFileSelectorListRowH + 8.0f
  };
}

void ScrubToNormalized(ViewerState & state, double normalized)
{
  if (state.frames.empty()) {
    return;
  }

  normalized = std::clamp(normalized, 0.0, 1.0);
  const auto & group = state.file_groups[state.current_file_group];
  const std::size_t file_frame_count = group.last_index - group.first_index + 1;
  const std::size_t max_file_offset = file_frame_count - 1;
  const std::size_t target_offset =
    static_cast<std::size_t>(std::llround(normalized * static_cast<double>(max_file_offset)));
  const std::size_t target_index = group.first_index + target_offset;
  if (target_index != state.current_index) {
    LoadFrameByIndex(state, target_index);
  }
  state.last_frame_advance_time = glfwGetTime();
}

void ScrubToMouse(ViewerState & state, double mouse_x)
{
  const double normalized = (mouse_x - kTimelineTrackX) / kTimelineTrackW;
  ScrubToNormalized(state, normalized);
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

fs::path SettingsPath()
{
  return fs::current_path() / kSettingsFileName;
}

std::string DetectDatasetProfile(const fs::path & input_path)
{
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
  return fs::current_path() / kProfilesDir / (profile + ".cfg");
}

fs::path AlgorithmsConfigPath()
{
  return fs::current_path() / kAlgorithmsConfigPath;
}

void SaveSettings(const ViewerState & state)
{
  std::error_code ec;
  fs::create_directories(fs::current_path() / kProfilesDir, ec);

  std::ofstream out(ProfileSettingsPath(state.dataset_profile), std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to open settings file for write");
  }

  out << "profile=" << state.dataset_profile << '\n';
  out << "grid_enabled=" << (state.grid_settings.enabled ? 1 : 0) << '\n';
  out << "grid_spacing=" << state.grid_settings.spacing << '\n';
  out << "grid_thickness=" << state.grid_settings.thickness << '\n';
  out << "grid_count=" << state.grid_settings.count << '\n';
  out << "color_mode=" << state.color_mode << '\n';
  CameraState camera_snapshot = state.camera;
  StoreCurrentProjectionView(camera_snapshot);

  out << "point_size=" << camera_snapshot.point_size << '\n';
  out << "ground_cell_size_m=" << state.algorithm_config.ground_removal.candidate_grid_cell_m << '\n';
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
}

void LoadSettingsFile(ViewerState & state, const fs::path & path)
{
  std::ifstream in(path);
  if (!in) {
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    try {
      if (key == "grid_enabled") {
        state.grid_settings.enabled = (std::stoi(value) != 0);
      } else if (key == "grid_spacing") {
        state.grid_settings.spacing = std::stof(value);
      } else if (key == "grid_thickness") {
        state.grid_settings.thickness = std::stof(value);
      } else if (key == "grid_count") {
        state.grid_settings.count = std::stoi(value);
      } else if (key == "color_mode") {
        state.color_mode = std::clamp(std::stoi(value), 0, 1);
      } else if (key == "point_size") {
        state.camera.point_size = std::clamp(std::stof(value), 1.0f, 8.0f);
      } else if (key == "ground_cell_size_m") {
        state.algorithm_config.ground_removal.candidate_grid_cell_m =
          std::clamp(std::stof(value), 0.1f, 0.5f);
      } else if (key == "projection_mode") {
        state.camera.projection_mode =
          (value == "2d") ? CameraState::ProjectionMode::View2D : CameraState::ProjectionMode::View3D;
      } else if (key == "camera_target_x") {
        state.camera.target.x = std::stof(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_target_y") {
        state.camera.target.y = std::stof(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_target_z") {
        state.camera.target.z = std::stof(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_yaw_rad") {
        state.camera.yaw = std::stof(value);
        state.camera.has_saved_view = true;
      } else if (key == "camera_pitch_rad") {
        state.camera.pitch = std::clamp(std::stof(value), -1.5707f, 1.5707f);
        state.camera.has_saved_view = true;
      } else if (key == "camera_distance") {
        state.camera.distance = std::clamp(std::stof(value), 1.0f, 5000.0f);
        state.camera.has_saved_view = true;
      } else if (key == "camera_3d_target_x") {
        state.camera.saved_3d_view.target.x = std::stof(value);
      } else if (key == "camera_3d_target_y") {
        state.camera.saved_3d_view.target.y = std::stof(value);
      } else if (key == "camera_3d_target_z") {
        state.camera.saved_3d_view.target.z = std::stof(value);
      } else if (key == "camera_3d_yaw_rad") {
        state.camera.saved_3d_view.yaw = std::stof(value);
      } else if (key == "camera_3d_pitch_rad") {
        state.camera.saved_3d_view.pitch = std::clamp(std::stof(value), -1.5707f, 1.5707f);
      } else if (key == "camera_3d_distance") {
        state.camera.saved_3d_view.distance = std::clamp(std::stof(value), 1.0f, 5000.0f);
      } else if (key == "camera_3d_valid") {
        state.camera.saved_3d_view.valid = (std::stoi(value) != 0);
      } else if (key == "camera_2d_target_x") {
        state.camera.saved_2d_view.target.x = std::stof(value);
      } else if (key == "camera_2d_target_y") {
        state.camera.saved_2d_view.target.y = std::stof(value);
      } else if (key == "camera_2d_target_z") {
        state.camera.saved_2d_view.target.z = std::stof(value);
      } else if (key == "camera_2d_yaw_rad") {
        state.camera.saved_2d_view.yaw = std::stof(value);
      } else if (key == "camera_2d_pitch_rad") {
        state.camera.saved_2d_view.pitch = std::clamp(std::stof(value), -1.5707f, 1.5707f);
      } else if (key == "camera_2d_distance") {
        state.camera.saved_2d_view.distance = std::clamp(std::stof(value), 1.0f, 5000.0f);
      } else if (key == "camera_2d_valid") {
        state.camera.saved_2d_view.valid = (std::stoi(value) != 0);
      } else if (key == "ground_removal_enabled") {
        state.ground_removal_enabled = (std::stoi(value) != 0);
      } else if (key == "clustering_enabled") {
        state.algorithm_config.clustering.enabled = (std::stoi(value) != 0);
      } else if (key == "detection_enabled") {
        state.algorithm_config.detection.enabled = (std::stoi(value) != 0);
      } else if (key == "calibration_yaw_deg") {
        state.calibration.yaw_deg = std::clamp(std::stof(value), -180.0f, 180.0f);
      } else if (key == "calibration_pitch_deg") {
        state.calibration.pitch_deg = std::clamp(std::stof(value), -180.0f, 180.0f);
      } else if (key == "calibration_roll_deg") {
        state.calibration.roll_deg = std::clamp(std::stof(value), -180.0f, 180.0f);
      } else if (key == "calibration_z_m") {
        state.calibration.z_m = std::clamp(std::stof(value), -5.0f, 5.0f);
      }
    } catch (...) {
    }
  }

  MarkGridDirty(state);
}

void LoadSettings(ViewerState & state)
{
  LoadSettingsFile(state, ProfileSettingsPath(state.dataset_profile));
  if (state.dataset_profile == "default") {
    LoadSettingsFile(state, SettingsPath());
  }
}

float * CalibrationValuePtr(ViewerState & state, int slider_index)
{
  switch (slider_index) {
    case 0:
      return &state.calibration.yaw_deg;
    case 1:
      return &state.calibration.pitch_deg;
    case 2:
      return &state.calibration.roll_deg;
    case 3:
      return &state.calibration.z_m;
    default:
      return nullptr;
  }
}

std::pair<float, float> CalibrationRange(int index)
{
  if (index == 5) { // cell size
    return {0.1f, 0.5f};
  }
  if (index == 4) {
    return {1.0f, 8.0f};
  }
  if (index == 3) {
    return {-5.0f, 5.0f};
  }
  return {-180.0f, 180.0f};
}

void UpdateCalibrationFromMouse(ViewerState & state, int slider_index, double mouse_x)
{
  float * value = CalibrationValuePtr(state, slider_index);
  if (slider_index == 4) {
    value = &state.camera.point_size;
  } else if (slider_index == 5) {
    value = &state.algorithm_config.ground_removal.candidate_grid_cell_m;
  }
  if (!value) {
    return;
  }

  const GridButton slider =
    (slider_index == 4) ? PointSizeSliderRect() :
    (slider_index == 5) ? GroundCellSizeSliderRect() :
    CalibrationSliderRect(slider_index);
  const auto [min_value, max_value] = CalibrationRange(slider_index);
  const double normalized = std::clamp((mouse_x - slider.x) / slider.w, 0.0, 1.0);
  *value = static_cast<float>(min_value + normalized * (max_value - min_value));
  if (slider_index == 4) {
    state.frame_dirty = true;
  } else if (slider_index == 5) {
    ApplyCalibration(state);
  } else {
    ApplyCalibration(state);
  }
}

void CommitCalibrationValueInput(ViewerState & state)
{
  float * value = CalibrationValuePtr(state, state.active_value_box);
  if (state.active_value_box == 4) {
    value = &state.camera.point_size;
  } else if (state.active_value_box == 5) {
    value = &state.algorithm_config.ground_removal.candidate_grid_cell_m;
  }
  if (value && !state.value_input_buffer.empty()) {
    try {
      const auto [min_value, max_value] = CalibrationRange(state.active_value_box);
      *value = std::clamp(std::stof(state.value_input_buffer), min_value, max_value);
      if (state.active_value_box == 4) {
        state.frame_dirty = true;
      } else {
        ApplyCalibration(state);
      }
    } catch (...) {
      SetTransientStatus(state, "INVALID VALUE");
    }
  }
  state.active_value_box = -1;
  state.value_input_buffer.clear();
}

void AdvanceFrame(ViewerState & state, int direction)
{
  if (state.frames.empty()) {
    return;
  }

  if (direction >= 0) {
    LoadFrameByIndex(state, (state.current_index + 1) % state.frames.size());
    return;
  }

  const std::size_t next = (state.current_index == 0) ? state.frames.size() - 1 : state.current_index - 1;
  LoadFrameByIndex(state, next);
}

void MouseButtonCallback(GLFWwindow * window, int button, int action, int)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) {
    return;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    if (action == GLFW_PRESS) {
      const GridButton timeline = TimelineTrackRect();
      if (IsInsideRect(mouse_x, mouse_y, timeline.x, timeline.y, timeline.w, timeline.h)) {
        state->timeline_dragging = true;
        ScrubToMouse(*state, mouse_x);
        return;
      }
      const GridButton file_selector = FileSelectorRect();
      if (IsInsideRect(mouse_x, mouse_y, file_selector.x, file_selector.y, file_selector.w, file_selector.h)) {
        state->file_selector_open = !state->file_selector_open;
        return;
      }
      if (state->file_selector_open) {
        const GridButton list_rect = FileSelectorListRect();
        if (IsInsideRect(mouse_x, mouse_y, list_rect.x, list_rect.y, list_rect.w, list_rect.h)) {
          const int clicked_row = static_cast<int>((mouse_y - (list_rect.y + 4.0f)) / kFileSelectorListRowH);
          if (clicked_row >= 0 && clicked_row < static_cast<int>(state->file_groups.size()) &&
            clicked_row < kFileSelectorVisibleRows)
          {
            const auto & group = state->file_groups[static_cast<std::size_t>(clicked_row)];
            state->file_selector_open = false;
            LoadFrameByIndex(*state, group.first_index);
            state->last_frame_advance_time = glfwGetTime();
            return;
          }
        } else {
          state->file_selector_open = false;
        }
      }
      if (IsInsideRect(mouse_x, mouse_y, kPlayButtonX, kPlayButtonY, kPlayButtonW, kPlayButtonH)) {
        state->is_playing = true;
        state->last_frame_advance_time = glfwGetTime();
        return;
      }
      if (IsInsideRect(mouse_x, mouse_y, kPauseButtonX, kPauseButtonY, kPauseButtonW, kPauseButtonH)) {
        state->is_playing = false;
        return;
      }
      const GridButton ground = GroundRemovalButton();
      if (IsInsideRect(mouse_x, mouse_y, ground.x, ground.y, ground.w, ground.h)) {
        state->ground_removal_enabled = !state->ground_removal_enabled;
        ApplyCalibration(*state);
        return;
      }
      const GridButton clustering = ClusteringButton();
      if (IsInsideRect(mouse_x, mouse_y, clustering.x, clustering.y, clustering.w, clustering.h)) {
        state->algorithm_config.clustering.enabled = !state->algorithm_config.clustering.enabled;
        ApplyCalibration(*state);
        return;
      }
      const GridButton detection = DetectionButton();
      if (IsInsideRect(mouse_x, mouse_y, detection.x, detection.y, detection.w, detection.h)) {
        state->algorithm_config.detection.enabled = !state->algorithm_config.detection.enabled;
        ApplyCalibration(*state);
        return;
      }
      const GridButton view2d = View2DButton();
      if (IsInsideRect(mouse_x, mouse_y, view2d.x, view2d.y, view2d.w, view2d.h)) {
        SetProjectionMode(*state, CameraState::ProjectionMode::View2D);
        return;
      }
      const GridButton view3d = View3DButton();
      if (IsInsideRect(mouse_x, mouse_y, view3d.x, view3d.y, view3d.w, view3d.h)) {
        SetProjectionMode(*state, CameraState::ProjectionMode::View3D);
        return;
      }
      const GridButton toggle = GridToggleButton();
      if (IsInsideRect(mouse_x, mouse_y, toggle.x, toggle.y, toggle.w, toggle.h)) {
        state->grid_settings.enabled = !state->grid_settings.enabled;
        return;
      }
      const GridButton save = GridSaveButton();
      if (IsInsideRect(mouse_x, mouse_y, save.x, save.y, save.w, save.h)) {
        try {
          SaveSettings(*state);
          SetTransientStatus(*state, "SETTINGS SAVED");
        } catch (const std::exception & e) {
          std::cerr << e.what() << '\n';
          SetTransientStatus(*state, "SAVE FAILED");
        }
        return;
      }
      for (int value_box = 0; value_box < 6; ++value_box) {
        const GridButton box =
          (value_box == 4) ? PointSizeValueBoxRect() :
          (value_box == 5) ? GroundCellSizeValueBoxRect() :
          CalibrationValueBoxRect(value_box);
        if (IsInsideRect(mouse_x, mouse_y, box.x, box.y, box.w, box.h)) {
          state->active_value_box = value_box;
          state->value_input_buffer.clear();
          return;
        }
      }
      for (int slider = 0; slider < 6; ++slider) {
        const GridButton slider_rect =
          (slider == 4) ? PointSizeSliderRect() :
          (slider == 5) ? GroundCellSizeSliderRect() :
          CalibrationSliderRect(slider);
        if (IsInsideRect(mouse_x, mouse_y, slider_rect.x, slider_rect.y, slider_rect.w, slider_rect.h)) {
          state->active_value_box = -1;
          state->value_input_buffer.clear();
          state->active_slider = slider;
          UpdateCalibrationFromMouse(*state, slider, mouse_x);
          return;
        }
      }
      for (int row = 0; row < 3; ++row) {
        const GridButton minus = GridMinusButton(row);
        const GridButton plus = GridPlusButton(row);
        if (IsInsideRect(mouse_x, mouse_y, minus.x, minus.y, minus.w, minus.h)) {
          if (row == 0) {
            state->grid_settings.spacing -= 0.5f;
          } else if (row == 1) {
            state->grid_settings.thickness -= 1.0f;
          } else {
            state->grid_settings.count -= 1;
          }
          MarkGridDirty(*state);
          return;
        }
        if (IsInsideRect(mouse_x, mouse_y, plus.x, plus.y, plus.w, plus.h)) {
          if (row == 0) {
            state->grid_settings.spacing += 0.5f;
          } else if (row == 1) {
            state->grid_settings.thickness += 1.0f;
          } else {
            state->grid_settings.count += 1;
          }
          MarkGridDirty(*state);
          return;
        }
      }
      state->active_value_box = -1;
      state->value_input_buffer.clear();
    }
    if (action == GLFW_RELEASE) {
      state->timeline_dragging = false;
      state->active_slider = -1;
    }
    state->camera.dragging_left = (action == GLFW_PRESS);
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    state->camera.dragging_right = (action == GLFW_PRESS);
  }

  glfwGetCursorPos(window, &state->camera.last_x, &state->camera.last_y);
}

void CursorCallback(GLFWwindow * window, double x, double y)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) {
    return;
  }

  const double dx = x - state->camera.last_x;
  const double dy = y - state->camera.last_y;
  state->camera.last_x = x;
  state->camera.last_y = y;

  if (state->timeline_dragging) {
    ScrubToMouse(*state, x);
    return;
  }

  if (state->active_slider >= 0) {
    UpdateCalibrationFromMouse(*state, state->active_slider, x);
    return;
  }

  if (state->camera.dragging_left) {
    if (state->camera.projection_mode == CameraState::ProjectionMode::View2D) {
      const float scale = std::max(kPanBaseScale * state->camera.distance, 0.01f);
      state->camera.target.x -= static_cast<float>(dx) * scale;
      state->camera.target.y += static_cast<float>(dy) * scale;
    } else {
      // Turntable-style orbit: moving the mouse right rotates the view right,
      // moving the mouse up tilts the camera upward.
      state->camera.yaw += static_cast<float>(dx) * kOrbitSensitivity;
      state->camera.pitch += static_cast<float>(dy) * kOrbitSensitivity;
      state->camera.pitch = std::clamp(state->camera.pitch, -1.5f, 1.5f);
    }
  }

  if (state->camera.dragging_right) {
    const float scale = std::max(kPanBaseScale * state->camera.distance, 0.01f);
    if (state->camera.projection_mode == CameraState::ProjectionMode::View2D) {
      state->camera.target.x -= static_cast<float>(dx) * scale;
      state->camera.target.y += static_cast<float>(dy) * scale;
    } else {
      const Vec3 eye = CameraPosition(state->camera);
      const Vec3 forward = Normalize(state->camera.target - eye);
      const Vec3 right = Normalize(Cross(forward, {0.0f, 0.0f, 1.0f}));
      const Vec3 up = Normalize(Cross(right, forward));
      state->camera.target = state->camera.target - right * static_cast<float>(dx) * scale +
        up * static_cast<float>(dy) * scale;
    }
  }
}

void ScrollCallback(GLFWwindow * window, double, double yoffset)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) {
    return;
  }

  state->camera.distance *= (yoffset > 0.0) ? 0.9f : 1.1f;
  state->camera.distance = std::clamp(state->camera.distance, 1.0f, 5000.0f);
}

void KeyCallback(GLFWwindow * window, int key, int, int action, int)
{
  if (action != GLFW_PRESS && action != GLFW_REPEAT) {
    return;
  }

  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state) {
    return;
  }

  if (state->active_value_box >= 0) {
    if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
      CommitCalibrationValueInput(*state);
      return;
    }
    if (key == GLFW_KEY_ESCAPE) {
      state->active_value_box = -1;
      state->value_input_buffer.clear();
      return;
    }
    if (key == GLFW_KEY_BACKSPACE && !state->value_input_buffer.empty()) {
      state->value_input_buffer.pop_back();
      return;
    }
    return;
  }

  switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_N:
      if (state->frames.size() > 1) {
        AdvanceFrame(*state, 1);
        state->last_frame_advance_time = glfwGetTime();
      }
      break;
    case GLFW_KEY_LEFT:
    case GLFW_KEY_P:
      if (state->frames.size() > 1) {
        AdvanceFrame(*state, -1);
        state->last_frame_advance_time = glfwGetTime();
      }
      break;
    case GLFW_KEY_F:
      state->fit_view_requested = true;
      break;
    case GLFW_KEY_C:
      state->color_mode = 1 - state->color_mode;
      break;
    case GLFW_KEY_LEFT_BRACKET:
      state->camera.point_size = std::max(1.0f, state->camera.point_size - 1.0f);
      break;
    case GLFW_KEY_RIGHT_BRACKET:
      state->camera.point_size = std::min(8.0f, state->camera.point_size + 1.0f);
      break;
    case GLFW_KEY_SPACE:
      state->is_playing = !state->is_playing;
      state->last_frame_advance_time = glfwGetTime();
      break;
    default:
      break;
  }
}

void CharCallback(GLFWwindow * window, unsigned int codepoint)
{
  auto * state = static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!state || state->active_value_box < 0 || state->value_input_buffer.size() >= 10) {
    return;
  }

  const char ch = static_cast<char>(codepoint);
  if ((ch >= '0' && ch <= '9') || ch == '.' || ch == '-') {
    state->value_input_buffer.push_back(ch);
  }
}

void BuildHudVertices(const ViewerState & state, std::vector<UiVertex> & vertices)
{
  vertices.clear();

  const Color panel_bg{0.05f, 0.07f, 0.10f, 0.82f};
  const Color panel_border{0.26f, 0.33f, 0.41f, 0.95f};
  const Color accent{0.26f, 0.70f, 0.96f, 1.0f};
  const Color title{0.93f, 0.96f, 0.98f, 1.0f};
  const Color text{0.72f, 0.79f, 0.86f, 1.0f};
  const Color hotkey_bg{0.15f, 0.19f, 0.24f, 0.95f};
  const Color hotkey_text{0.98f, 0.99f, 1.0f, 1.0f};
  const Color success{0.18f, 0.74f, 0.46f, 0.95f};
  const Color neutral{0.22f, 0.26f, 0.31f, 0.95f};
  const Color paused{0.87f, 0.42f, 0.27f, 0.95f};
  const Color value_bg{0.10f, 0.13f, 0.17f, 0.95f};
  const Color subtext{0.55f, 0.63f, 0.72f, 1.0f};
  const Color timeline_bg{0.05f, 0.07f, 0.10f, 0.80f};
  const Color timeline_track{0.18f, 0.22f, 0.28f, 1.0f};
  const Color timeline_fill{0.26f, 0.70f, 0.96f, 1.0f};
  const Color timeline_handle{0.96f, 0.98f, 1.0f, 1.0f};

  AppendRect(vertices, kTimelineX, kTimelineY, kTimelineX + kTimelineW, kTimelineY + kTimelineH, timeline_bg);
  AppendRect(vertices, kTimelineX, kTimelineY, kTimelineX + kTimelineW, kTimelineY + 1.0f, panel_border);
  AppendRect(vertices, kTimelineX, kTimelineY + kTimelineH - 1.0f, kTimelineX + kTimelineW, kTimelineY + kTimelineH, panel_border);
  AppendRect(vertices, kTimelineX, kTimelineY, kTimelineX + 1.0f, kTimelineY + kTimelineH, panel_border);
  AppendRect(vertices, kTimelineX + kTimelineW - 1.0f, kTimelineY, kTimelineX + kTimelineW, kTimelineY + kTimelineH, panel_border);
  AppendText(vertices, kTimelineX + 16.0f, kTimelineY + 8.0f, 1.2f, "CURRENT FILE", subtext);
  const std::string current_file_name = state.file_groups.empty()
    ? std::string("NO FILE")
    : state.file_groups[state.current_file_group].display_name;
  AppendText(
    vertices,
    kTimelineX + 16.0f,
    kTimelineY + 22.0f,
    1.35f,
    EllipsizeMiddle(current_file_name, 42),
    title);
  AppendText(vertices, kTimelineX + 16.0f, kTimelineY + 36.0f, 1.1f, "FRAME TIMELINE", subtext);
  AppendRect(vertices, kTimelineTrackX, kTimelineTrackY, kTimelineTrackX + kTimelineTrackW, kTimelineTrackY + kTimelineTrackH, timeline_track);

  float progress = 0.0f;
  std::size_t local_frame_index = 0;
  std::size_t local_frame_count = 1;
  if (!state.file_groups.empty()) {
    const auto & group = state.file_groups[state.current_file_group];
    local_frame_index = state.current_index - group.first_index;
    local_frame_count = group.last_index - group.first_index + 1;
    progress = (local_frame_count <= 1)
      ? 0.0f
      : static_cast<float>(local_frame_index) / static_cast<float>(local_frame_count - 1);
  }
  const float fill_w = kTimelineTrackW * progress;
  AppendRect(vertices, kTimelineTrackX, kTimelineTrackY, kTimelineTrackX + fill_w, kTimelineTrackY + kTimelineTrackH, timeline_fill);
  const float handle_x = kTimelineTrackX + fill_w;
  AppendRect(vertices, handle_x - 4.0f, kTimelineTrackY - 4.0f, handle_x + 4.0f, kTimelineTrackY + kTimelineTrackH + 4.0f, timeline_handle);

  std::ostringstream timeline_label;
  timeline_label << (local_frame_index + 1) << " / " << local_frame_count;
  AppendText(vertices, kTimelineTrackX + kTimelineTrackW - 86.0f, kTimelineY + 36.0f, 1.1f, timeline_label.str(), subtext);

  const GridButton file_selector = FileSelectorRect();
  AppendText(vertices, file_selector.x, kTimelineY + 8.0f, 1.2f, "BIN FILE", subtext);
  AppendRect(
    vertices,
    file_selector.x,
    file_selector.y,
    file_selector.x + file_selector.w,
    file_selector.y + file_selector.h,
    hotkey_bg);
  AppendText(
    vertices,
    file_selector.x + 8.0f,
    file_selector.y + 5.0f,
    1.15f,
    EllipsizeMiddle(current_file_name, 18),
    hotkey_text);
  AppendText(vertices, file_selector.x + file_selector.w - 18.0f, file_selector.y + 5.0f, 1.15f, "V", hotkey_text);

  if (state.file_selector_open) {
    const GridButton list_rect = FileSelectorListRect();
    AppendRect(
      vertices,
      list_rect.x,
      list_rect.y,
      list_rect.x + list_rect.w,
      list_rect.y + list_rect.h,
      panel_bg);
    AppendRect(vertices, list_rect.x, list_rect.y, list_rect.x + list_rect.w, list_rect.y + 1.0f, panel_border);
    AppendRect(vertices, list_rect.x, list_rect.y + list_rect.h - 1.0f, list_rect.x + list_rect.w, list_rect.y + list_rect.h, panel_border);
    AppendRect(vertices, list_rect.x, list_rect.y, list_rect.x + 1.0f, list_rect.y + list_rect.h, panel_border);
    AppendRect(vertices, list_rect.x + list_rect.w - 1.0f, list_rect.y, list_rect.x + list_rect.w, list_rect.y + list_rect.h, panel_border);
    const int visible_rows = std::min(static_cast<int>(state.file_groups.size()), kFileSelectorVisibleRows);
    for (int row = 0; row < visible_rows; ++row) {
      const float row_y = list_rect.y + 4.0f + static_cast<float>(row) * kFileSelectorListRowH;
      const bool selected = static_cast<std::size_t>(row) == state.current_file_group;
      AppendRect(
        vertices,
        list_rect.x + 4.0f,
        row_y,
        list_rect.x + list_rect.w - 4.0f,
        row_y + kFileSelectorListRowH - 2.0f,
        selected ? accent : value_bg);
      AppendText(
        vertices,
        list_rect.x + 10.0f,
        row_y + 4.0f,
        1.05f,
        EllipsizeMiddle(state.file_groups[static_cast<std::size_t>(row)].display_name, 20),
        selected ? hotkey_text : text);
    }
  }

  AppendRect(vertices, kHudPanelX, kHudPanelY, kHudPanelX + kHudPanelW, kHudPanelY + kHudPanelH, panel_bg);
  AppendRect(vertices, kHudPanelX, kHudPanelY, kHudPanelX + kHudPanelW, kHudPanelY + 2.0f, panel_border);
  AppendRect(vertices, kHudPanelX, kHudPanelY + kHudPanelH - 1.0f, kHudPanelX + kHudPanelW, kHudPanelY + kHudPanelH, panel_border);
  AppendRect(vertices, kHudPanelX, kHudPanelY, kHudPanelX + 1.0f, kHudPanelY + kHudPanelH, panel_border);
  AppendRect(vertices, kHudPanelX + kHudPanelW - 1.0f, kHudPanelY, kHudPanelX + kHudPanelW, kHudPanelY + kHudPanelH, panel_border);
  AppendRect(vertices, kHudPanelX, kHudPanelY, kHudPanelX + 58.0f, kHudPanelY + 3.0f, accent);

  const float title_scale = 2.1f;
  const float body_scale = 1.8f;
  const float small_scale = 1.4f;

  AppendText(vertices, kHudPanelX + 16.0f, kHudPanelY + 14.0f, title_scale, "LIDAR VIEWER", title);
  AppendText(vertices, kHudPanelX + 16.0f, kHudPanelY + 30.0f, 1.2f, "OPENGL POINT CLOUD INSPECTOR", subtext);
  AppendText(vertices, kHudPanelX + 214.0f, kHudPanelY + 30.0f, 1.2f, "PROFILE " + state.dataset_profile, subtext);

  std::ostringstream frame_stream;
  frame_stream << "FRAME " << (state.current_index + 1) << "/" << state.frames.size();
  AppendText(vertices, kHudPanelX + 16.0f, kHudPanelY + 48.0f, body_scale, frame_stream.str(), accent);

  AppendRect(
    vertices,
    kPlayButtonX,
    kPlayButtonY,
    kPlayButtonX + kPlayButtonW,
    kPlayButtonY + kPlayButtonH,
    state.is_playing ? success : neutral);
  AppendRect(
    vertices,
    kPauseButtonX,
    kPauseButtonY,
    kPauseButtonX + kPauseButtonW,
    kPauseButtonY + kPauseButtonH,
    state.is_playing ? neutral : paused);
  AppendText(vertices, kPlayButtonX + 13.0f, kPlayButtonY + 4.0f, small_scale, "PLAY", hotkey_text);
  AppendText(vertices, kPauseButtonX + 10.0f, kPauseButtonY + 4.0f, small_scale, "PAUSE", hotkey_text);
  const GridButton ground = GroundRemovalButton();
  AppendRect(
    vertices,
    ground.x,
    ground.y,
    ground.x + ground.w,
    ground.y + ground.h,
    state.ground_removal_enabled ? success : neutral);
  AppendText(vertices, ground.x + 8.0f, ground.y + 4.0f, 1.2f, "GROUND", hotkey_text);
  const GridButton clustering = ClusteringButton();
  AppendRect(
    vertices,
    clustering.x,
    clustering.y,
    clustering.x + clustering.w,
    clustering.y + clustering.h,
    state.algorithm_config.clustering.enabled ? success : neutral);
  AppendText(vertices, clustering.x + 7.0f, clustering.y + 4.0f, 1.15f, "CLUST", hotkey_text);
  const GridButton detection = DetectionButton();
  AppendRect(
    vertices,
    detection.x,
    detection.y,
    detection.x + detection.w,
    detection.y + detection.h,
    state.algorithm_config.detection.enabled ? paused : neutral);
  AppendText(vertices, detection.x + 8.0f, detection.y + 4.0f, 1.15f, "DETECT", hotkey_text);
  const GridButton view2d = View2DButton();
  const GridButton view3d = View3DButton();
  AppendRect(
    vertices,
    view2d.x,
    view2d.y,
    view2d.x + view2d.w,
    view2d.y + view2d.h,
    state.camera.projection_mode == CameraState::ProjectionMode::View2D ? accent : neutral);
  AppendRect(
    vertices,
    view3d.x,
    view3d.y,
    view3d.x + view3d.w,
    view3d.y + view3d.h,
    state.camera.projection_mode == CameraState::ProjectionMode::View3D ? accent : neutral);
  AppendText(vertices, view2d.x + 15.0f, view2d.y + 4.0f, 1.15f, "2D", hotkey_text);
  AppendText(vertices, view3d.x + 15.0f, view3d.y + 4.0f, 1.15f, "3D", hotkey_text);

  std::ostringstream status_stream;
  status_stream << (state.is_playing ? "STATUS PLAYING " : "STATUS PAUSED ")
                << std::fixed << std::setprecision(1) << state.playback_fps << " FPS";
  AppendText(vertices, kHudPanelX + 142.0f, kHudPanelY + 84.0f, 1.2f, status_stream.str(), text);
  std::ostringstream fps_stream;
  fps_stream << "CUR FPS " << std::fixed << std::setprecision(1) << state.current_playback_fps;
  if (state.input_fps_estimate.has_value()) {
    fps_stream << " / INPUT " << std::fixed << std::setprecision(1) << *state.input_fps_estimate;
  } else {
    fps_stream << " / INPUT N/A";
  }
  AppendText(vertices, kHudPanelX + 142.0f, kHudPanelY + 98.0f, 1.2f, fps_stream.str(), subtext);
  std::ostringstream render_fps_stream;
  render_fps_stream << "RENDER " << std::fixed << std::setprecision(1) << state.current_view_fps;
  AppendText(vertices, kHudPanelX + 142.0f, kHudPanelY + 112.0f, 1.15f, render_fps_stream.str(), subtext);
  if (state.ground_removal_enabled) {
    AppendText(
      vertices,
      kHudPanelX + 180.0f,
      kHudPanelY + 84.0f,
      1.2f,
      "GROUND " + std::to_string(state.ground_removed_count),
      text);
  }
  if (state.algorithm_config.clustering.enabled) {
    AppendText(
      vertices,
      kHudPanelX + 180.0f,
      kHudPanelY + 126.0f,
      1.2f,
      "CLUSTERS " + std::to_string(state.clusters.size()),
      subtext);
  }
  if (state.algorithm_config.detection.enabled) {
    AppendText(
      vertices,
      kHudPanelX + 16.0f,
      kHudPanelY + 126.0f,
      1.2f,
      "DETECTIONS " + std::to_string(state.detections.size()),
      subtext);
    AppendText(vertices, kHudPanelX + 16.0f, kHudPanelY + 140.0f, 1.15f, "CLASSES", subtext);
    const std::array<int, 7> legend_classes{{
      lidar_viewer::algorithms::DetectionClassCar,
      lidar_viewer::algorithms::DetectionClassBus,
      lidar_viewer::algorithms::DetectionClassTruck,
      lidar_viewer::algorithms::DetectionClassPerson,
      lidar_viewer::algorithms::DetectionClassCyclist,
      lidar_viewer::algorithms::DetectionClassTree,
      lidar_viewer::algorithms::DetectionClassUnknown,
    }};
    AppendRect(
      vertices,
      kHudPanelX + 16.0f,
      kHudPanelY + 154.0f,
      kHudPanelX + kHudPanelW - 16.0f,
      kHudPanelY + 216.0f,
      value_bg);
    for (std::size_t legend_index = 0; legend_index < legend_classes.size(); ++legend_index) {
      const int class_id = legend_classes[legend_index];
      const float legend_x = kHudPanelX + 28.0f + 118.0f * static_cast<float>(legend_index % 3);
      const float legend_y = kHudPanelY + 164.0f + 16.0f * static_cast<float>(legend_index / 3);
      const auto [fill_color, line_color] = DetectionClassColors(class_id);
      AppendRect(vertices, legend_x, legend_y, legend_x + 12.0f, legend_y + 12.0f, line_color);
      AppendText(vertices, legend_x + 18.0f, legend_y + 1.0f, 1.0f, DetectionClassLabel(class_id), text);
    }
  }

  const float file_block_y = state.algorithm_config.detection.enabled ? (kHudPanelY + 228.0f) : (kHudPanelY + 140.0f);
  AppendText(vertices, kHudPanelX + 16.0f, file_block_y, 1.2f, "CURRENT FILE", subtext);
  AppendRect(vertices, kHudPanelX + 16.0f, file_block_y + 14.0f, kHudPanelX + kHudPanelW - 16.0f, file_block_y + 34.0f, value_bg);
  AppendText(
    vertices,
    kHudPanelX + 22.0f,
    file_block_y + 20.0f,
    1.35f,
    EllipsizeMiddle(state.frames[state.current_index].label, 38),
    title);
  AppendText(
    vertices,
    kHudPanelX + 16.0f,
    file_block_y + 48.0f,
    small_scale,
    state.algorithm_config.detection.enabled ? "MODE DETECTION" :
      (state.ground_removal_enabled ? "MODE GROUND MASK" : "MODE WHITE"),
    text);
  AppendText(
    vertices,
    kHudPanelX + 170.0f,
    file_block_y + 48.0f,
    small_scale,
    "POINTS " + std::to_string(state.frame.points.size()),
    text);

  struct ControlRow
  {
    const char * key;
    const char * action;
  };

  const std::array<ControlRow, 9> controls{{
    {"SPACE", "PLAY / PAUSE"},
    {"N", "NEXT FRAME"},
    {"P", "PREV FRAME"},
    {"F", "FIT VIEW"},
    {"C", "COLOR MODE"},
    {"2D/3D", "PROJECTION"},
    {"[ ]", "POINT"},
    {"ESC", "QUIT"},
    {"LMB", "ORBIT"},
  }};

  const float controls_y = state.algorithm_config.detection.enabled ? (kHudPanelY + 274.0f) : (kHudPanelY + 186.0f);
  AppendText(vertices, kHudPanelX + 16.0f, controls_y, 1.2f, "CONTROLS", subtext);

  float row_y = controls_y + 16.0f;
  for (const auto & row : controls) {
    const float key_box_width = (std::string(row.key).size() > 3) ? 76.0f : 50.0f;
    AppendRect(vertices, kHudPanelX + 16.0f, row_y - 2.0f, kHudPanelX + 16.0f + key_box_width, row_y + 13.0f, hotkey_bg);
    AppendText(vertices, kHudPanelX + 23.0f, row_y + 1.0f, small_scale, row.key, hotkey_text);
    AppendText(vertices, kHudPanelX + 16.0f + key_box_width + 12.0f, row_y + 1.0f, small_scale, row.action, text);
    row_y += 15.0f;
  }

  AppendText(vertices, kHudPanelX + 16.0f, kHudPanelY + kHudPanelH - 16.0f, 1.3f, "RMB PAN  WHEEL ZOOM", subtext);

  AppendRect(vertices, kGridPanelX, kGridPanelY, kGridPanelX + kGridPanelW, kGridPanelY + kGridPanelH, panel_bg);
  AppendRect(vertices, kGridPanelX, kGridPanelY, kGridPanelX + kGridPanelW, kGridPanelY + 2.0f, panel_border);
  AppendRect(vertices, kGridPanelX, kGridPanelY + kGridPanelH - 1.0f, kGridPanelX + kGridPanelW, kGridPanelY + kGridPanelH, panel_border);
  AppendRect(vertices, kGridPanelX, kGridPanelY, kGridPanelX + 1.0f, kGridPanelY + kGridPanelH, panel_border);
  AppendRect(vertices, kGridPanelX + kGridPanelW - 1.0f, kGridPanelY, kGridPanelX + kGridPanelW, kGridPanelY + kGridPanelH, panel_border);
  AppendRect(vertices, kGridPanelX, kGridPanelY, kGridPanelX + 40.0f, kGridPanelY + 3.0f, accent);
  AppendText(vertices, kGridPanelX + 16.0f, kGridPanelY + 14.0f, title_scale, "GRID", title);
  AppendText(vertices, kGridPanelX + 16.0f, kGridPanelY + 30.0f, 1.2f, "WORLD REFERENCE", subtext);

  const GridButton toggle = GridToggleButton();
  AppendRect(vertices, toggle.x, toggle.y, toggle.x + toggle.w, toggle.y + toggle.h, state.grid_settings.enabled ? success : neutral);
  AppendText(vertices, toggle.x + 17.0f, toggle.y + 4.0f, small_scale, state.grid_settings.enabled ? "ON" : "OFF", hotkey_text);

  struct GridRow
  {
    const char * label;
    std::string value;
  };

  std::ostringstream spacing_stream;
  spacing_stream << std::fixed << std::setprecision(1) << state.grid_settings.spacing;
  std::ostringstream thick_stream;
  thick_stream << std::fixed << std::setprecision(0) << state.grid_settings.thickness;
  std::ostringstream count_stream;
  count_stream << state.grid_settings.count;

  const std::array<GridRow, 3> grid_rows{{
    {"SPACING", spacing_stream.str()},
    {"THICK", thick_stream.str()},
    {"COUNT", count_stream.str()},
  }};

  for (int row = 0; row < 3; ++row) {
    const float row_y = kGridPanelY + 34.0f + 26.0f * static_cast<float>(row);
    const GridButton minus = GridMinusButton(row);
    const GridButton plus = GridPlusButton(row);
    AppendText(vertices, kGridPanelX + 16.0f, row_y + 4.0f, small_scale, grid_rows[row].label, text);
    AppendRect(vertices, kGridPanelX + 170.0f, row_y, kGridPanelX + 242.0f, row_y + 16.0f, value_bg);
    AppendText(vertices, kGridPanelX + 188.0f, row_y + 4.0f, small_scale, grid_rows[row].value, title);
    AppendRect(vertices, minus.x, minus.y, minus.x + minus.w, minus.y + minus.h, hotkey_bg);
    AppendRect(vertices, plus.x, plus.y, plus.x + plus.w, plus.y + plus.h, hotkey_bg);
    AppendText(vertices, minus.x + 6.0f, minus.y + 4.0f, small_scale, "-", hotkey_text);
    AppendText(vertices, plus.x + 6.0f, plus.y + 4.0f, small_scale, "+", hotkey_text);
  }

  AppendText(vertices, kGridPanelX + 16.0f, kGridPanelY + 138.0f, 1.2f, "CALIBRATION", subtext);
  struct CalibRow
  {
    const char * label;
    float value;
    int precision;
  };
  const std::array<CalibRow, 4> calib_rows{{
    {"YAW", state.calibration.yaw_deg, 1},
    {"PITCH", state.calibration.pitch_deg, 1},
    {"ROLL", state.calibration.roll_deg, 1},
    {"Z", state.calibration.z_m, 3},
  }};
  for (int row = 0; row < 4; ++row) {
    const float row_y = kGridPanelY + 152.0f + 28.0f * static_cast<float>(row);
    const GridButton slider = CalibrationSliderRect(row);
    const GridButton value_box = CalibrationValueBoxRect(row);
    const auto [min_value, max_value] = CalibrationRange(row);
    const float normalized = std::clamp(
      (calib_rows[row].value - min_value) / (max_value - min_value),
      0.0f,
      1.0f);
    const float handle_x = slider.x + slider.w * normalized;
    std::ostringstream value_stream;
    if (state.active_value_box == row && !state.value_input_buffer.empty()) {
      value_stream << state.value_input_buffer;
    } else {
      value_stream << std::fixed << std::setprecision(calib_rows[row].precision) << calib_rows[row].value;
    }
    AppendText(vertices, kGridPanelX + 16.0f, row_y + 2.0f, small_scale, calib_rows[row].label, text);
    AppendRect(vertices, slider.x, slider.y + 4.0f, slider.x + slider.w, slider.y + 8.0f, hotkey_bg);
    AppendRect(vertices, slider.x, slider.y + 4.0f, handle_x, slider.y + 8.0f, accent);
    AppendRect(vertices, handle_x - 4.0f, slider.y, handle_x + 4.0f, slider.y + slider.h, title);
    AppendRect(
      vertices,
      value_box.x,
      value_box.y,
      value_box.x + value_box.w,
      value_box.y + value_box.h,
      state.active_value_box == row ? hotkey_bg : value_bg);
    AppendText(vertices, value_box.x + 7.0f, row_y + 2.0f, small_scale, value_stream.str(), title);
  }

  const float point_row_y = kGridPanelY + 270.0f;
  const GridButton point_slider = PointSizeSliderRect();
  const auto [point_min, point_max] = CalibrationRange(4);
  const float point_normalized = std::clamp(
    (state.camera.point_size - point_min) / (point_max - point_min),
    0.0f,
    1.0f);
  const float point_handle_x = point_slider.x + point_slider.w * point_normalized;
  std::ostringstream point_size_stream;
  if (state.active_value_box == 4 && !state.value_input_buffer.empty()) {
    point_size_stream << state.value_input_buffer;
  } else {
    point_size_stream << std::fixed << std::setprecision(1) << state.camera.point_size;
  }
  const GridButton point_value_box = PointSizeValueBoxRect();
  AppendText(vertices, kGridPanelX + 16.0f, point_row_y + 2.0f, small_scale, "POINT", text);
  AppendRect(vertices, point_slider.x, point_slider.y + 4.0f, point_slider.x + point_slider.w, point_slider.y + 8.0f, hotkey_bg);
  AppendRect(vertices, point_slider.x, point_slider.y + 4.0f, point_handle_x, point_slider.y + 8.0f, accent);
  AppendRect(vertices, point_handle_x - 4.0f, point_slider.y, point_handle_x + 4.0f, point_slider.y + point_slider.h, title);
  AppendRect(
    vertices,
    point_value_box.x,
    point_value_box.y,
    point_value_box.x + point_value_box.w,
    point_value_box.y + point_value_box.h,
    state.active_value_box == 4 ? hotkey_bg : value_bg);
  AppendText(vertices, kGridPanelX + 241.0f, point_row_y + 2.0f, small_scale, point_size_stream.str(), title);

  const float ground_cell_row_y = kGridPanelY + 294.0f;
  const GridButton ground_cell_slider = GroundCellSizeSliderRect();
  const auto [cell_min, cell_max] = CalibrationRange(5);
  const float cell_normalized = std::clamp(
    (state.algorithm_config.ground_removal.candidate_grid_cell_m - cell_min) / (cell_max - cell_min),
    0.0f,
    1.0f);
  const float cell_handle_x = ground_cell_slider.x + ground_cell_slider.w * cell_normalized;
  std::ostringstream ground_cell_stream;
  if (state.active_value_box == 5 && !state.value_input_buffer.empty()) {
    ground_cell_stream << state.value_input_buffer;
  } else {
    ground_cell_stream << std::fixed << std::setprecision(2) << state.algorithm_config.ground_removal.candidate_grid_cell_m;
  }
  const GridButton ground_cell_value_box = GroundCellSizeValueBoxRect();
  AppendText(vertices, kGridPanelX + 16.0f, ground_cell_row_y + 2.0f, small_scale, "CELL", text);
  AppendRect(vertices, ground_cell_slider.x, ground_cell_slider.y + 4.0f, ground_cell_slider.x + ground_cell_slider.w, ground_cell_slider.y + 8.0f, hotkey_bg);
  AppendRect(vertices, ground_cell_slider.x, ground_cell_slider.y + 4.0f, cell_handle_x, ground_cell_slider.y + 8.0f, accent);
  AppendRect(vertices, cell_handle_x - 4.0f, ground_cell_slider.y, cell_handle_x + 4.0f, ground_cell_slider.y + ground_cell_slider.h, title);
  AppendRect(
    vertices,
    ground_cell_value_box.x,
    ground_cell_value_box.y,
    ground_cell_value_box.x + ground_cell_value_box.w,
    ground_cell_value_box.y + ground_cell_value_box.h,
    state.active_value_box == 5 ? hotkey_bg : value_bg);
  AppendText(vertices, kGridPanelX + 237.0f, ground_cell_row_y + 2.0f, small_scale, ground_cell_stream.str(), title);

  const float axis_y = kGridPanelY + 318.0f;
  AppendText(vertices, kGridPanelX + 16.0f, axis_y, 1.2f, "AXES", subtext);
  AppendRect(vertices, kGridPanelX + 66.0f, axis_y - 1.0f, kGridPanelX + 76.0f, axis_y + 9.0f, {0.94f, 0.29f, 0.26f, 1.0f});
  AppendText(vertices, kGridPanelX + 82.0f, axis_y, small_scale, "X", title);
  AppendRect(vertices, kGridPanelX + 106.0f, axis_y - 1.0f, kGridPanelX + 116.0f, axis_y + 9.0f, {0.21f, 0.82f, 0.46f, 1.0f});
  AppendText(vertices, kGridPanelX + 122.0f, axis_y, small_scale, "Y", title);
  AppendRect(vertices, kGridPanelX + 146.0f, axis_y - 1.0f, kGridPanelX + 156.0f, axis_y + 9.0f, {0.30f, 0.62f, 0.98f, 1.0f});
  AppendText(vertices, kGridPanelX + 162.0f, axis_y, small_scale, "Z", title);

  const GridButton save = GridSaveButton();
  AppendRect(vertices, save.x, save.y, save.x + save.w, save.y + save.h, accent);
  AppendText(vertices, save.x + 12.0f, save.y + 4.0f, small_scale, "SAVE", hotkey_text);
  if (!state.transient_status.empty()) {
    AppendText(vertices, kGridPanelX + 16.0f, kGridPanelY + kGridPanelH - 22.0f, 1.1f, state.transient_status, subtext);
  }
}

void RenderHud(const ViewerState & state, int viewport_w, int viewport_h)
{
  std::vector<UiVertex> vertices;
  BuildHudVertices(state, vertices);

  glUseProgram(state.ui.program);
  glUniform2f(state.ui.viewport_loc, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
  glBindVertexArray(state.ui.vao);
  glBindBuffer(GL_ARRAY_BUFFER, state.ui.vbo);
  glBufferData(
    GL_ARRAY_BUFFER,
    static_cast<GLsizeiptr>(vertices.size() * sizeof(UiVertex)),
    vertices.data(),
    GL_DYNAMIC_DRAW);

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
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
  const float axis_length = std::max(8.0f, std::min(state.frame.radius * 0.45f, 30.0f));
  const std::array<std::pair<Vec3, Color>, 3> axes{{
    {{axis_length, 0.0f, 0.0f}, {0.94f, 0.29f, 0.26f, 1.0f}},
    {{0.0f, axis_length, 0.0f}, {0.21f, 0.82f, 0.46f, 1.0f}},
    {{0.0f, 0.0f, axis_length}, {0.30f, 0.62f, 0.98f, 1.0f}},
  }};

  glUseProgram(state.grid.program);
  glUniformMatrix4fv(state.grid.mvp_loc, 1, GL_FALSE, mvp.m.data());
  glBindVertexArray(state.grid.vao);
  glBindBuffer(GL_ARRAY_BUFFER, state.grid.vbo);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glLineWidth(3.0f);

  for (const auto & axis : axes) {
    const std::array<Vec3, 2> vertices{{{0.0f, 0.0f, 0.0f}, axis.first}};
    glBufferData(
      GL_ARRAY_BUFFER,
      static_cast<GLsizeiptr>(vertices.size() * sizeof(Vec3)),
      vertices.data(),
      GL_DYNAMIC_DRAW);
    glUniform4f(state.grid.color_loc, axis.second.r, axis.second.g, axis.second.b, axis.second.a);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));
  }

  glLineWidth(1.0f);
  state.grid.dirty = true;
}
}  // namespace

int RunViewer(int argc, char ** argv)
{
  try {
    fs::path input_path = kDefaultInputPath;
    if (argc >= 2) {
      input_path = argv[1];
    }

    ViewerState state;
    state.dataset_profile = DetectDatasetProfile(input_path);
    state.algorithm_config = lidar_viewer::LoadPipelineConfig(AlgorithmsConfigPath());
    LoadSettings(state);
    state.frames = ResolveInputFrames(input_path);
    state.input_fps_estimate = EstimateInputFps(state);
    SyncPlaybackFpsToInput(state);
    BuildFileGroups(state);
    LoadFrameByIndex(state, 0);
    state.last_frame_advance_time = 0.0;

    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow * window = glfwCreateWindow(kWindowWidth, kWindowHeight, "LiDAR OpenGL Viewer", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

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
    glfwSetCharCallback(window, CharCallback);

    glGenVertexArrays(1, &state.vao);
    glGenBuffers(1, &state.vbo);

    glBindVertexArray(state.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(KittiPoint), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(KittiPoint), reinterpret_cast<void *>(offsetof(KittiPoint, intensity)));

    state.ui.program = CreateUiProgram();
    state.ui.viewport_loc = glGetUniformLocation(state.ui.program, "u_viewport");
    glGenVertexArrays(1, &state.ui.vao);
    glGenBuffers(1, &state.ui.vbo);
    glBindVertexArray(state.ui.vao);
    glBindBuffer(GL_ARRAY_BUFFER, state.ui.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(UiVertex), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(UiVertex), reinterpret_cast<void *>(offsetof(UiVertex, r)));

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

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
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

      int fb_width = 0;
      int fb_height = 0;
      glfwGetFramebufferSize(window, &fb_width, &fb_height);
      glViewport(0, 0, fb_width, fb_height);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      const float aspect = (fb_height > 0) ? static_cast<float>(fb_width) / static_cast<float>(fb_height) : 1.0f;
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
        ? Vec3{0.0f, 1.0f, 0.0f}
        : Vec3{0.0f, 0.0f, 1.0f};
      const Mat4 view = LookAt(eye, state.camera.target, up);
      const Mat4 mvp = Multiply(projection, view);

      glUseProgram(program);
      glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp.m.data());
      glUniform1f(point_size_loc, state.camera.point_size);
      glUniform1i(color_mode_loc, state.color_mode);
      glUniform1f(min_height_loc, state.frame.min_z);
      glUniform1f(max_height_loc, state.frame.max_z);
      if (state.grid.dirty) {
        RebuildGrid(state);
      }
      glBindVertexArray(state.vao);
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(state.frame.points.size()));
      RenderGrid(state, mvp);
      RenderClusterBoxes(state, mvp);
      RenderAxes(state, mvp);
      RenderHud(state, fb_width, fb_height);

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

    glDeleteProgram(program);
    glDeleteProgram(state.ui.program);
    glDeleteProgram(state.grid.program);
    glDeleteBuffers(1, &state.vbo);
    glDeleteBuffers(1, &state.ui.vbo);
    glDeleteBuffers(1, &state.grid.vbo);
    glDeleteBuffers(1, &state.cluster_boxes.fill_vbo);
    glDeleteBuffers(1, &state.cluster_boxes.line_vbo);
    glDeleteVertexArrays(1, &state.vao);
    glDeleteVertexArrays(1, &state.ui.vao);
    glDeleteVertexArrays(1, &state.grid.vao);
    glDeleteVertexArrays(1, &state.cluster_boxes.vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
