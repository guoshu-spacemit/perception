cmake_minimum_required(VERSION 3.16)
project(stereo_matching VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(STEREO_MATCHING_BUILD_DEMO "Build the image-file inference demo" OFF)
option(STEREO_MATCHING_BUILD_EXTERNAL_INPUT "Build the LAS2 NV12 DMA-BUF source backend" OFF)
option(LAS2_ENABLE_RVV_RECTIFICATION "Use RVV map operations when supported by the compiler" ON)
set(STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY "" CACHE FILEPATH
  "Matching NV12 external-input backend; not interchangeable with libstereo_matching.so")

add_library(stereo_matching SHARED IMPORTED GLOBAL)
set_target_properties(stereo_matching PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/libstereo_matching.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_library(stereo_matching::stereo_matching ALIAS stereo_matching)

if(STEREO_MATCHING_BUILD_DEMO)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  add_executable(stereo_matching_demo demo/demo.cpp)
  target_compile_features(stereo_matching_demo PRIVATE cxx_std_17)
  target_compile_options(stereo_matching_demo PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_include_directories(stereo_matching_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(stereo_matching_demo PRIVATE stereo_matching ${OpenCV_LIBS})
  set_target_properties(stereo_matching_demo PROPERTIES BUILD_RPATH "$ORIGIN/../lib")
  install(TARGETS stereo_matching_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES lib/libstereo_matching.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching)
set(STEREO_MATCHING_HAS_EXTERNAL_INPUT FALSE)
if(STEREO_MATCHING_BUILD_EXTERNAL_INPUT AND STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY)
  message(FATAL_ERROR "Choose source ExternalStereo or an explicit prebuilt backend, not both")
endif()
if(STEREO_MATCHING_BUILD_EXTERNAL_INPUT)
  include(cmake/ExternalInput.cmake)
  set(STEREO_MATCHING_HAS_EXTERNAL_INPUT TRUE)
elseif(STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY)
  if(NOT EXISTS "${STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY}")
    message(FATAL_ERROR "STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY does not exist")
  endif()
  set(STEREO_MATCHING_HAS_EXTERNAL_INPUT TRUE)
  include(cmake/PrebuiltExternalInput.cmake)
endif()
configure_package_config_file(cmake/stereo_matchingConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/stereo_matchingConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stereo_matching)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/stereo_matchingConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stereo_matchingConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/stereo_matchingConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stereo_matching)
