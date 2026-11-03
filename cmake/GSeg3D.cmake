# Keep the pinned upstream checkout untouched. Initialize its cell state in a
# generated header: sparse cells have no estimated normal, and the virtual
# seed must have a defined centroid before neighbor/floating-cell checks.
set(gseg_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ground_segmentation/include")
file(READ "${gseg_root}/ground_detection_types.hpp" gseg_types)
file(READ "${gseg_root}/ground_detection.hpp" gseg_core)
set(gseg_init_anchor "    explored = false;")
set(gseg_seed_anchor "  robot_cell.terrain_type = TerrainType::GROUND;")
set(gseg_plane_anchor "  Eigen::Vector3d n = gcell.normal;")
foreach(pair IN ITEMS gseg_init_anchor gseg_seed_anchor gseg_plane_anchor)
  if(pair STREQUAL "gseg_init_anchor")
    set(gseg_text "${gseg_types}")
  else()
    set(gseg_text "${gseg_core}")
  endif()
  string(FIND "${gseg_text}" "${${pair}}" gseg_anchor_index)
  if(gseg_anchor_index EQUAL -1)
    message(FATAL_ERROR "Pinned GSeg3D initialization adapter no longer matches its upstream header")
  endif()
endforeach()
string(REPLACE "${gseg_init_anchor}" [=[    explored = false;
    // LiDAR Studio: no evidence must not become a plane through uninitialized memory.
    terrain_type = TerrainType::UNDEFINED;
    primitive_type = PrimitiveType::NOISE;
    centroid.setZero();
    normal.setZero();
    eigenvalues.setZero();
    eigenvectors.setIdentity();]=] gseg_types "${gseg_types}")
string(REPLACE "${gseg_seed_anchor}" [=[  robot_cell.terrain_type = TerrainType::GROUND;
  if (robot_cell.points->size() < 3) {
    robot_cell.centroid << 0.0, 0.0, grid_config.distToGround, 1.0;
  }]=] gseg_core "${gseg_core}")
string(REPLACE "${gseg_plane_anchor}" [=[  // A horizontal line/sparse cell is not evidence for an arbitrary plane.
  if (gcell.primitive_type != PrimitiveType::PLANE || gcell.inliers->indices.empty()) return false;
  Eigen::Vector3d n = gcell.normal;]=] gseg_core "${gseg_core}")
set(gseg_generated "${CMAKE_CURRENT_BINARY_DIR}/gseg3d_adapter")
file(MAKE_DIRECTORY "${gseg_generated}")
file(WRITE "${gseg_generated}/ground_detection_types.hpp" "${gseg_types}")
file(WRITE "${gseg_generated}/ground_detection.hpp" "${gseg_core}")
add_library(lidar_viewer_gseg3d INTERFACE)
target_include_directories(lidar_viewer_gseg3d INTERFACE "${gseg_generated}" "${gseg_root}" ${PCL_INCLUDE_DIRS})
target_link_libraries(lidar_viewer_gseg3d INTERFACE ${PCL_LIBRARIES} nanoflann::nanoflann)
