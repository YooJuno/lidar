#include "lidar_viewer/ui_layout.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
using lidar_viewer::ComputeUiLayout;
using lidar_viewer::UiLayout;
using lidar_viewer::UiRect;

void Check(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool Overlaps(const UiRect & a, const UiRect & b)
{
  constexpr float epsilon = 0.01f;
  return a.w > epsilon && a.h > epsilon && b.w > epsilon && b.h > epsilon &&
    a.x < b.x + b.w - epsilon && b.x < a.x + a.w - epsilon &&
    a.y < b.y + b.h - epsilon && b.y < a.y + a.h - epsilon;
}

void CheckGeometry(float requested_width, float requested_height, float scale)
{
  const float width = std::isfinite(requested_width) ? std::max(0.0f, requested_width) : 0.0f;
  const float height = std::isfinite(requested_height) ? std::max(0.0f, requested_height) : 0.0f;
  const UiLayout layout = ComputeUiLayout(requested_width, requested_height, scale);
  const std::array<UiRect, 5> regions{
    layout.header, layout.sidebar, layout.viewport, layout.transport, layout.status};
  Check(std::isfinite(layout.scale) && layout.scale >= 0.75f && layout.scale <= 2.0f,
    "Effective scale must remain readable and bounded");
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto & rect = regions[i];
    Check(std::isfinite(rect.x) && std::isfinite(rect.y) &&
      std::isfinite(rect.w) && std::isfinite(rect.h), "Nonfinite rectangle");
    Check(rect.x >= 0 && rect.y >= 0 && rect.w >= 0 && rect.h >= 0,
      "Negative layout geometry");
    Check(rect.x + rect.w <= width + 0.01f && rect.y + rect.h <= height + 0.01f,
      "A panel extends beyond the logical window");
    for (std::size_t j = i + 1; j < regions.size(); ++j) {
      Check(!Overlaps(rect, regions[j]), "Panel interiors overlap");
    }
  }
  if (width >= 800 && height >= 600) {
    Check(layout.viewport.w > width * 0.50f && layout.viewport.h > 100,
      "Desktop layout must preserve a useful point-cloud canvas");
    Check(layout.sidebar.h > 200 && layout.transport.h >= 70,
      "Desktop settings and transport regions disappeared");
    Check(layout.sidebar.w <= width * 0.40f + 0.01f,
      "Sidebar dominates a narrow desktop window");
  }
}

}  // namespace

int main()
{
  try {
    const std::array<std::pair<float, float>, 16> sizes{{
      {0, 0}, {0, 720}, {1280, 0}, {1, 1}, {24, 16}, {320, 200},
      {640, 480}, {800, 600}, {1024, 640}, {1280, 720},
      {1600, 960}, {1920, 1080}, {2560, 1440}, {3840, 2160},
      {-10, 720}, {1280, -10}}};
    for (const auto & size : sizes) {
      for (const float scale : {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f}) {
        CheckGeometry(size.first, size.second, scale);
      }
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CheckGeometry(nan, 720, 1);
    CheckGeometry(1280, inf, 1);
    CheckGeometry(1600, 960, nan);
    CheckGeometry(1600, 960, inf);
    CheckGeometry(1600, 960, -1);
    const auto normal = ComputeUiLayout(1600, 960);
    Check(normal.header.h == 56 && normal.sidebar.w == 320 &&
      normal.transport.h == 110 && normal.status.h == 28,
      "Normal desktop dimensions changed unexpectedly");
    Check(!normal.compact && ComputeUiLayout(1024, 640).compact,
      "Compact mode must reflect available logical space");
    std::cout << "ui_layout_smoke: responsive layout checks passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "ui_layout_smoke: " << error.what() << '\n';
    return 1;
  }
}
