# Original FugSeg core, pinned in the submodule. ROS/dataset/Boost I/O is not
# needed. Inject a small sensor configuration API into a generated header;
# the upstream segmentation source itself remains unchanged.
set(fugseg_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fugseg/src/fugseg")
if(NOT EXISTS "${fugseg_root}/include/fugseg/fugseg.h")
  message(FATAL_ERROR "Missing FugSeg submodule. Run: git submodule update --init --recursive")
endif()
file(READ "${fugseg_root}/include/fugseg/fugseg.h" fugseg_header)
set(fugseg_anchor "        bool ExtractGround(")
string(FIND "${fugseg_header}" "${fugseg_anchor}" fugseg_anchor_index)
if(fugseg_anchor_index EQUAL -1)
  message(FATAL_ERROR "Pinned FugSeg sensor adapter no longer matches the upstream header")
endif()
set(fugseg_sensor_api [[
        // LiDAR Studio integration: sensor origin/noise, no KITTI vehicle crop,
        // and dynamic buffers for scans larger than the upstream 134400 limit.
        void ConfigureSensor(float height, float horizontal_deg, float vertical_deg,
                             float range_sigma, size_t capacity) {
            _lidarHeight = height;
            _maxHeightSeed = 0.3f - height;
            _close_region_boundary_x_neg = 0;
            _close_region_boundary_x_pos = 0;
            _close_region_boundary_y = 0;
            _m2_theta = horizontal_deg * static_cast<float>(PI/180.0);
            _m2_theta *= _m2_theta;
            _m2_phi = vertical_deg * static_cast<float>(PI/180.0);
            _m2_phi *= _m2_phi;
            _m2_ra = range_sigma * range_sigma;
            if (capacity > static_cast<size_t>(_range2d.size())) {
                _range2d.resize(static_cast<Eigen::Index>(capacity));
                _yaws.resize(static_cast<Eigen::Index>(capacity));
            }
        }
        bool ExtractGround(]])
string(REPLACE "${fugseg_anchor}" "${fugseg_sensor_api}" fugseg_header "${fugseg_header}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fugseg_adapter/fugseg")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fugseg_adapter/fugseg/fugseg.h" "${fugseg_header}")
add_library(lidar_viewer_fugseg STATIC "${fugseg_root}/src/fugseg.cpp" "${fugseg_root}/src/cloud.cpp")
target_include_directories(lidar_viewer_fugseg SYSTEM PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/fugseg_adapter" "${fugseg_root}/include")
target_link_libraries(lidar_viewer_fugseg PUBLIC Eigen3::Eigen)
