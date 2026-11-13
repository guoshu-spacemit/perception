# The public header is include/stereo_pipeline.h. Compile only this TU:
# its private runtime and preprocessing headers contain the shared implementation.
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
  "External matched SpaceMIT ORT/EP development and runtime artifact")
set(_las2_use_embedded_runtime FALSE)
if(ONNXRUNTIME_ROOT)
  find_path(LAS2_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(LAS2_ORT_LIBRARY onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(LAS2_EP_LIBRARY spacemit_ep
    PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    if(NOT EXISTS "${LAS2_ORT_INCLUDE_DIR}/spacemit_ort_env.h")
      message(FATAL_ERROR "ONNXRUNTIME_ROOT is missing the matched spacemit_ort_env.h")
    endif()
    find_library(LAS2_TCM_LIBRARY spine_tcm
      PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  endif()
  set(_las2_runtime_libraries "${LAS2_ORT_LIBRARY}" "${LAS2_EP_LIBRARY}")
  set(_las2_runtime_build_rpath "${ONNXRUNTIME_ROOT}/lib")
else()
  set(_las2_use_embedded_runtime TRUE)
  include("${CMAKE_CURRENT_LIST_DIR}/EmbeddedRuntime.cmake")
endif()

get_filename_component(_las2_sdk_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(DEFINED ENV{SROBOTIS_ROOT} AND NOT "$ENV{SROBOTIS_ROOT}" STREQUAL "")
  set(_las2_sdk_root "$ENV{SROBOTIS_ROOT}")
endif()
set(MPP_ROOT "${_las2_sdk_root}/components/multimedia/mpp" CACHE PATH "SDK MPP source headers")
find_path(LAS2_MPP_INCLUDE_DIR sys/vb_type.h
  PATHS "${MPP_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
set(DMABUFHEAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/dmabufheap" CACHE PATH
  "Compatible allocator source, using the minimal licensed snapshot by default")
if(NOT EXISTS "${DMABUFHEAP_ROOT}/src/BufferAllocator.cpp" OR
   NOT EXISTS "${DMABUFHEAP_ROOT}/src/BufferAllocatorWrapper.cpp")
  message(FATAL_ERROR "DMABUFHEAP_ROOT must contain the compatible allocator sources")
endif()

add_library(stereo_matching_pipeline SHARED
  src/stereo_pipeline.cpp
  "${DMABUFHEAP_ROOT}/src/BufferAllocator.cpp"
  "${DMABUFHEAP_ROOT}/src/BufferAllocatorWrapper.cpp")
add_library(stereo_matching::external_input ALIAS stereo_matching_pipeline)
target_compile_features(stereo_matching_pipeline PUBLIC cxx_std_17)
target_include_directories(stereo_matching_pipeline
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE src "${DMABUFHEAP_ROOT}/include" "${LAS2_ORT_INCLUDE_DIR}"
    "${LAS2_MPP_INCLUDE_DIR}" "${MPP_ROOT}/include/sys"
    ${OpenCV_INCLUDE_DIRS})
if(NOT LAS2_ENABLE_RVV_RECTIFICATION)
  target_compile_definitions(stereo_matching_pipeline PRIVATE LAS2_DISABLE_RVV_RECTIFICATION=1)
endif()
target_compile_options(stereo_matching_pipeline PRIVATE -O3 -Wall -Wextra -ffunction-sections -fdata-sections)
target_link_options(stereo_matching_pipeline PRIVATE -Wl,--gc-sections -Wl,--no-undefined)
target_link_libraries(stereo_matching_pipeline PRIVATE
  ${OpenCV_LIBS} OpenCL::OpenCL ${_las2_runtime_libraries}
  Threads::Threads yaml-cpp)
set_target_properties(stereo_matching_pipeline PROPERTIES
  CXX_EXTENSIONS OFF VERSION 1.0.0 SOVERSION 1
  BUILD_RPATH "${_las2_runtime_build_rpath}" INSTALL_RPATH "$ORIGIN")
install(TARGETS stereo_matching_pipeline LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES thirdparty/dmabufheap/LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/licenses RENAME dmabufheap-LICENSE)
include("${CMAKE_CURRENT_LIST_DIR}/ExternalRuntime.cmake")
