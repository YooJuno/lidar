cmake_minimum_required(VERSION 3.16)
project(lidar_opengl_viewer LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIDAR_VIEWER_ENABLE_CUDA "Build optional CUDA acceleration backend" ON)

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
    set(CMAKE_CUDA_ARCHITECTURES 60)
    set(LIDAR_VIEWER_CUDA_SOURCES
      src/algorithms/clustering_cuda.cu
      src/algorithms/ground_removal_cuda.cu
    )
    add_compile_definitions(LIDAR_VIEWER_HAS_CUDA=1)
    message(STATUS "CUDA backend enabled: ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA compiler not found; using CPU fallback backend")
  endif()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)
pkg_check_modules(GLEW REQUIRED glew)

add_executable(lidar_viewer
  src/main.cpp
  src/viewer_app.cpp
  src/data_io.cpp
  src/config.cpp
  src/algorithms/ground_removal.cpp
  src/algorithms/clustering.cpp
  ${LIDAR_VIEWER_CUDA_SOURCES}
  src/algorithms/detection.cpp
  src/algorithms/pipeline.cpp
)

target_compile_options(lidar_viewer PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

target_include_directories(lidar_viewer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${GLFW3_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
)

target_link_directories(lidar_viewer PRIVATE
  ${GLFW3_LIBRARY_DIRS}
  ${GLEW_LIBRARY_DIRS}
)

target_link_libraries(lidar_viewer PRIVATE
  ${GLFW3_LIBRARIES}
  ${GLEW_LIBRARIES}
  GL
)
