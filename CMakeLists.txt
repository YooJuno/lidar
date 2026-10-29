cmake_minimum_required(VERSION 3.16)
# PCL's VTK package exposes an MPI C target, so enable C as well as C++.
project(lidar_opengl_viewer VERSION 0.2.0 LANGUAGES C CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
endif()
function(lidar_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
  endif()
endfunction()
option(LIDAR_VIEWER_USER_SETTINGS "Save profiles in the user's settings directory (packaged builds)" OFF)

option(LIDAR_VIEWER_ENABLE_CUDA "Build optional CUDA acceleration backend" ON)
set(LIDAR_VIEWER_CUDA_ARCHITECTURES "89" CACHE STRING
  "Semicolon-separated CUDA architectures (89 is native for the RTX 4060)")

set(LIDAR_VIEWER_CUDA_SOURCES
  src/algorithms/clustering_cuda_stub.cpp
  src/algorithms/ground_removal_cuda_stub.cpp
)
if(LIDAR_VIEWER_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES ${LIDAR_VIEWER_CUDA_ARCHITECTURES})
    set(LIDAR_VIEWER_CUDA_SOURCES
      src/algorithms/clustering_cuda.cu
      src/algorithms/ground_removal_cuda.cu
    )
    add_compile_definitions(LIDAR_VIEWER_HAS_CUDA=1)
    # Older nvcc frontends cannot parse glibc's fortified level-3 builtins.
    # Keep host C++ hardening untouched; use supported level 2 for CUDA only.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
      add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-U_FORTIFY_SOURCE>
        $<$<COMPILE_LANGUAGE:CUDA>:-D_FORTIFY_SOURCE=2>)
    endif()
    message(STATUS "CUDA backend enabled: ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA compiler not found; using CPU fallback backend")
  endif()
endif()

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(dear_imgui
  URL https://codeload.github.com/ocornut/imgui/tar.gz/f5befd2d29e66809cd1110a152e375a7f1981f06
  URL_HASH SHA256=85f4ce357df05bcc331b587f01976f47fb55f19fadf477a7907289686bc3f4c8
)
FetchContent_MakeAvailable(dear_imgui)
add_library(lidar_viewer_imgui STATIC
  ${dear_imgui_SOURCE_DIR}/imgui.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_draw.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_tables.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${dear_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${dear_imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(lidar_viewer_imgui PUBLIC
  ${dear_imgui_SOURCE_DIR} ${dear_imgui_SOURCE_DIR}/backends)
target_link_libraries(lidar_viewer_imgui PUBLIC glfw ${CMAKE_DL_LIBS})
find_package(PCL REQUIRED COMPONENTS common filters kdtree segmentation)
find_package(nanoflann REQUIRED)

# Optional offline bag reader; no ROS node, DDS connection or sensor driver.
# Source /opt/ros/jazzy/setup.bash before configuring to enable this backend.
option(LIDAR_VIEWER_ENABLE_ROSBAG "Read ROS 2 PointCloud2 bags directly" ON)
if(LIDAR_VIEWER_ENABLE_ROSBAG)
  find_package(rosbag2_cpp QUIET)
  find_package(sensor_msgs QUIET)
  find_package(tf2_msgs QUIET)
  find_package(rclcpp QUIET)
endif()
add_library(lidar_viewer_bag STATIC src/bag_io.cpp src/bag_catalog.cpp src/imu_alignment.cpp)
target_include_directories(lidar_viewer_bag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
lidar_warnings(lidar_viewer_bag)
if(LIDAR_VIEWER_ENABLE_ROSBAG AND rosbag2_cpp_FOUND AND sensor_msgs_FOUND)
  target_compile_definitions(lidar_viewer_bag PUBLIC LIDAR_VIEWER_HAS_ROSBAG=1)
  target_link_libraries(lidar_viewer_bag PUBLIC
    rosbag2_cpp::rosbag2_cpp sensor_msgs::sensor_msgs__rosidl_typesupport_cpp)
  if(tf2_msgs_FOUND)
    target_compile_definitions(lidar_viewer_bag PUBLIC LIDAR_VIEWER_HAS_STATIC_TF=1)
    target_link_libraries(lidar_viewer_bag PUBLIC tf2_msgs::tf2_msgs__rosidl_typesupport_cpp)
  endif()
  message(STATUS "ROS 2 bag input enabled (offline reader)")
else()
  message(STATUS "ROS 2 bag input unavailable; .bin input remains supported")
endif()

# GSeg3D is a header-only core library.  We intentionally include the pinned
# core source only; its ROS 2 wrapper (TF, IMU, PointCloud2) is not part of
# this viewer.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ground_segmentation/include/ground_detection.hpp")
  message(FATAL_ERROR "Missing GSeg3D submodule. Run: git submodule update --init --recursive")
endif()
add_library(lidar_viewer_gseg3d INTERFACE)
target_include_directories(lidar_viewer_gseg3d INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ground_segmentation/include"
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(lidar_viewer_gseg3d INTERFACE ${PCL_LIBRARIES} nanoflann::nanoflann)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/patchwork_plusplus/cpp/patchworkpp/src/patchworkpp.cpp")
  message(FATAL_ERROR "Missing Patchwork++ submodule. Run: git submodule update --init --recursive")
endif()
# PCL may provide Eigen 3 or Eigen 5 under the same Eigen3 package/target.
# A requested 3.x version rejects compatible Eigen 5 installations.
find_package(Eigen3 REQUIRED NO_MODULE)
add_library(lidar_viewer_patchworkpp STATIC
  third_party/patchwork_plusplus/cpp/patchworkpp/src/patchworkpp.cpp
  third_party/patchwork_plusplus/cpp/common/src/plane_fit.cpp)
target_include_directories(lidar_viewer_patchworkpp SYSTEM PUBLIC
  third_party/patchwork_plusplus/cpp/patchworkpp/include
  third_party/patchwork_plusplus/cpp/common/include)
target_link_libraries(lidar_viewer_patchworkpp PUBLIC Eigen3::Eigen)

add_library(lidar_viewer_roi STATIC src/roi.cpp)
target_include_directories(lidar_viewer_roi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
lidar_warnings(lidar_viewer_roi)

if(WIN32)
  add_library(lidar_viewer_file_dialog STATIC src/file_dialog_win.cpp)
  target_link_libraries(lidar_viewer_file_dialog PUBLIC comdlg32 ole32)
else()
  add_library(lidar_viewer_file_dialog STATIC src/file_dialog.cpp)
endif()
target_include_directories(lidar_viewer_file_dialog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
lidar_warnings(lidar_viewer_file_dialog)

add_library(lidar_viewer_live STATIC src/live_io.cpp)
target_include_directories(lidar_viewer_live PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar_viewer_live PUBLIC lidar_viewer_bag)
lidar_warnings(lidar_viewer_live)
if(LIDAR_VIEWER_ENABLE_ROSBAG AND rosbag2_cpp_FOUND AND sensor_msgs_FOUND AND tf2_msgs_FOUND AND rclcpp_FOUND)
  target_compile_definitions(lidar_viewer_live PUBLIC LIDAR_VIEWER_HAS_LIVE_ROS=1)
  target_link_libraries(lidar_viewer_live PUBLIC rclcpp::rclcpp)
  message(STATUS "Live ROS 2 input and raw MCAP recording enabled")
endif()

find_package(Threads REQUIRED)
add_library(lidar_viewer_detector STATIC src/learned_detector.cpp src/algorithms/detection.cpp)
target_include_directories(lidar_viewer_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar_viewer_detector PUBLIC Threads::Threads)
lidar_warnings(lidar_viewer_detector)

add_executable(lidar_viewer
  src/main.cpp
  src/viewer_app.cpp
  src/data_io.cpp
  src/config.cpp
  src/algorithms/ground_removal.cpp
  src/algorithms/ground_removal_gseg3d.cpp
  src/algorithms/ground_removal_patchworkpp.cpp
  src/algorithms/ground_removal_fast.cpp
  src/algorithms/clustering.cpp
  ${LIDAR_VIEWER_CUDA_SOURCES}
  src/algorithms/pipeline.cpp
)

lidar_warnings(lidar_viewer)
target_compile_definitions(lidar_viewer PRIVATE LIDAR_VIEWER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(LIDAR_VIEWER_USER_SETTINGS)
  target_compile_definitions(lidar_viewer PRIVATE LIDAR_VIEWER_USER_SETTINGS=1)
endif()

target_include_directories(lidar_viewer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lidar_viewer PRIVATE
  glfw
  GLEW::GLEW
  OpenGL::GL
  lidar_viewer_gseg3d
  lidar_viewer_patchworkpp
  lidar_viewer_bag
  lidar_viewer_imgui
  lidar_viewer_roi
  lidar_viewer_file_dialog
  lidar_viewer_live
  lidar_viewer_detector
)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_executable(learned_detector_smoke_test tests/learned_detector_smoke.cpp)
    target_link_libraries(learned_detector_smoke_test PRIVATE lidar_viewer_detector)
    add_test(NAME learned_detector_smoke_test COMMAND learned_detector_smoke_test ${Python3_EXECUTABLE})
    set_tests_properties(learned_detector_smoke_test PROPERTIES TIMEOUT 45)
  endif()
  add_executable(live_io_smoke_test tests/live_io_smoke.cpp)
  target_link_libraries(live_io_smoke_test PRIVATE lidar_viewer_live)
  add_test(NAME live_io_smoke_test COMMAND live_io_smoke_test)
  set_tests_properties(live_io_smoke_test PROPERTIES TIMEOUT 45
    ENVIRONMENT "ROS_DOMAIN_ID=187;ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(file_dialog_smoke_test tests/file_dialog_smoke.cpp)
    target_link_libraries(file_dialog_smoke_test PRIVATE lidar_viewer_file_dialog)
    add_test(NAME file_dialog_smoke_test COMMAND file_dialog_smoke_test)
  endif()
  add_executable(roi_smoke_test tests/roi_smoke.cpp)
  target_link_libraries(roi_smoke_test PRIVATE lidar_viewer_roi)
  add_test(NAME roi_smoke_test COMMAND roi_smoke_test)
  add_executable(imu_alignment_smoke_test tests/imu_alignment_smoke.cpp)
  target_link_libraries(imu_alignment_smoke_test PRIVATE lidar_viewer_bag)
  add_test(NAME imu_alignment_smoke_test COMMAND imu_alignment_smoke_test)
  add_executable(config_data_smoke_test tests/config_data_smoke.cpp src/config.cpp src/data_io.cpp)
  target_include_directories(config_data_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(config_data_smoke_test PRIVATE LIDAR_VIEWER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(config_data_smoke_test PRIVATE lidar_viewer_bag)
  add_test(NAME config_data_smoke_test COMMAND config_data_smoke_test)
  add_executable(settings_io_smoke_test tests/settings_io_smoke.cpp)
  target_include_directories(settings_io_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME settings_io_smoke_test COMMAND settings_io_smoke_test)
  add_executable(ui_layout_smoke_test tests/ui_layout_smoke.cpp)
  target_include_directories(ui_layout_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME ui_layout_smoke_test COMMAND ui_layout_smoke_test)
  add_executable(bag_io_smoke_test tests/bag_io_smoke.cpp)
  target_link_libraries(bag_io_smoke_test PRIVATE lidar_viewer_bag)
  add_test(NAME bag_io_smoke_test COMMAND bag_io_smoke_test)
  add_executable(bag_catalog_smoke_test tests/bag_catalog_smoke.cpp src/bag_catalog.cpp)
  target_include_directories(bag_catalog_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME bag_catalog_smoke_test COMMAND bag_catalog_smoke_test)
  add_executable(algorithm_smoke_test
    tests/algorithm_smoke.cpp
    tests/algorithm_regression.cpp
    src/algorithms/ground_removal.cpp
    src/algorithms/ground_removal_gseg3d.cpp
    src/algorithms/ground_removal_patchworkpp.cpp
    src/algorithms/ground_removal_fast.cpp
    src/algorithms/clustering.cpp
    ${LIDAR_VIEWER_CUDA_SOURCES}
  )
  lidar_warnings(algorithm_smoke_test)
  target_include_directories(algorithm_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(algorithm_smoke_test PRIVATE lidar_viewer_gseg3d lidar_viewer_patchworkpp lidar_viewer_bag)
  add_test(NAME algorithm_smoke_test COMMAND algorithm_smoke_test)
endif()

include(cmake/Packaging.cmake)
