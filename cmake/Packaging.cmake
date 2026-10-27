set(CPACK_PACKAGE_NAME "LiDARViewer")
set(CPACK_PACKAGE_VENDOR "YooJuno")
set(CPACK_PACKAGE_CONTACT "https://github.com/YooJuno/lidar/issues")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "LiDAR viewer with yaw ROI, ground segmentation and optional ROS 2 live input")
if(NOT CPACK_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()
set(CPACK_PACKAGE_FILE_NAME "LiDARViewer-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages")

if(APPLE)
  set_target_properties(lidar_viewer PROPERTIES MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "LiDAR Viewer"
    MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.yoojuno.lidar"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${CPACK_PACKAGE_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${CPACK_PACKAGE_VERSION}")
  install(TARGETS lidar_viewer BUNDLE DESTINATION .)
  set(resource_destination "lidar_viewer.app/Contents/Resources")
  set(CPACK_GENERATOR "ZIP;DragNDrop")
  set(CPACK_DMG_VOLUME_NAME "LiDAR Viewer")
elseif(WIN32)
  target_sources(lidar_viewer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/resources/app.manifest")
  install(TARGETS lidar_viewer RUNTIME DESTINATION .)
  set(resource_destination ".")
  if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/" DESTINATION .
      FILES_MATCHING PATTERN "*.dll")
    install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/" DESTINATION third-party-licenses
      FILES_MATCHING PATTERN "copyright")
  endif()
  # Include the redistributable runtime beside the executable.
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
  include(InstallRequiredSystemLibraries)
  set(CPACK_GENERATOR ZIP)
else()
  install(TARGETS lidar_viewer RUNTIME DESTINATION lib/lidar_viewer)
  install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux/lidar-viewer" DESTINATION bin)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/resources/lidar-viewer.desktop" DESTINATION share/applications)
  set(resource_destination "share/lidar_viewer")
  set(CPACK_GENERATOR "TGZ;DEB")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "YooJuno")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "zenity")
  if(LIDAR_VIEWER_ENABLE_ROSBAG AND rosbag2_cpp_FOUND)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "zenity, ros-jazzy-rclcpp, ros-jazzy-rosbag2-cpp, ros-jazzy-sensor-msgs, ros-jazzy-tf2-msgs, ros-jazzy-rmw-fastrtps-cpp, ros-jazzy-rosbag2-storage-mcap, ros-jazzy-rosbag2-storage-sqlite3")
    set_target_properties(lidar_viewer PROPERTIES INSTALL_RPATH "/opt/ros/jazzy/lib")
  endif()
endif()
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config/" DESTINATION "${resource_destination}/config"
  FILES_MATCHING PATTERN "*.cfg" PATTERN "*.yaml")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md" DESTINATION "${resource_destination}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ground_segmentation/LICENSE.txt"
  DESTINATION "${resource_destination}/third-party-licenses" RENAME GSeg3D-BSD-3-Clause.txt)
install(FILES "${dear_imgui_SOURCE_DIR}/LICENSE.txt"
  DESTINATION "${resource_destination}/third-party-licenses" RENAME Dear-ImGui-MIT.txt)
if(APPLE)
  install(CODE "set(lidar_fixup_dirs \"${PCL_LIBRARY_DIRS};${CMAKE_PREFIX_PATH}/lib\")")
  install(CODE [[
    include(BundleUtilities)
    set(BU_CHMOD_BUNDLE_ITEMS ON)
    fixup_bundle("${CMAKE_INSTALL_PREFIX}/lidar_viewer.app" "" "${lidar_fixup_dirs}")
    execute_process(COMMAND codesign --force --deep --sign - "${CMAKE_INSTALL_PREFIX}/lidar_viewer.app"
      COMMAND_ERROR_IS_FATAL ANY)
  ]])
endif()
include(CPack)
