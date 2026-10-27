#pragma once

#include <algorithm>
#include <cmath>

namespace lidar_viewer
{

// Window/logical coordinates, not framebuffer pixels. ImGui's renderer owns
// the conversion to the framebuffer's content scale.
struct UiRect
{
  float x{0.0f};
  float y{0.0f};
  float w{0.0f};
  float h{0.0f};
};

struct UiLayout
{
  UiRect header;
  UiRect sidebar;
  UiRect viewport;
  UiRect transport;
  UiRect status;
  float scale{1.0f};
  bool compact{false};
};

inline UiLayout ComputeUiLayout(
  float window_width, float window_height, float requested_scale = 1.0f)
{
  const float width = std::isfinite(window_width) ? std::max(0.0f, window_width) : 0.0f;
  const float height = std::isfinite(window_height) ? std::max(0.0f, window_height) : 0.0f;
  const float requested = std::isfinite(requested_scale) && requested_scale > 0.0f ?
    requested_scale : 1.0f;
  // Large DPI/user scales must not consume a small window's whole canvas.
  // Below the usable desktop minimum, keep controls readable and let the
  // sidebar scroll; geometry is still clipped to valid, nonnegative bounds.
  const float budget_scale = std::min(width / 1024.0f, height / 720.0f);
  const float scale = std::clamp(std::min(requested, budget_scale), 0.75f, 2.0f);
  UiLayout layout;
  layout.scale = scale;
  layout.compact = width < 1280.0f * scale || height < 800.0f * scale;
  if (width == 0.0f || height == 0.0f) {
    return layout;
  }

  const float gap = std::min({16.0f * scale, width * 0.05f, height * 0.05f});
  const float status_height = std::min(28.0f * scale, height);
  const float header_height = std::min(56.0f * scale, height - status_height);
  layout.header = {0.0f, 0.0f, width, header_height};
  layout.status = {0.0f, height - status_height, width, status_height};

  const float body_y = std::min(header_height + gap, layout.status.y);
  const float body_bottom = std::max(body_y, layout.status.y - gap);
  const float body_height = body_bottom - body_y;
  const float content_width = std::max(0.0f, width - 3.0f * gap);
  const float sidebar_width = std::min(320.0f * scale, content_width * 0.40f);
  layout.sidebar = {gap, body_y, sidebar_width, body_height};

  const float main_x = std::min(width, layout.sidebar.x + sidebar_width + gap);
  const float main_width = std::max(0.0f, width - gap - main_x);
  const float transport_height = std::min(110.0f * scale, body_height);
  layout.transport = {main_x, body_bottom - transport_height, main_width, transport_height};
  const float viewport_height = std::max(0.0f, layout.transport.y - gap - body_y);
  layout.viewport = {main_x, body_y, main_width, viewport_height};
  return layout;
}

}  // namespace lidar_viewer
