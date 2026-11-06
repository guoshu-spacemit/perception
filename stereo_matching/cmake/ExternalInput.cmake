# The public ABI is the delivered las2_usb_stereo.h. Compile only this TU:
# it includes the USB and common implementation .cpp files itself.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LAS2_LIBDRM REQUIRED IMPORTED_TARGET libdrm)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
  "External matched SpaceMIT ORT/EP development and runtime artifact")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "Source ExternalStereo requires an explicit ONNXRUNTIME_ROOT with the matched "
    "SpaceMIT ORT/EP headers and libraries; private runtime/model files are not in this source package")
endif()
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

get_filename_component(_las2_sdk_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(DEFINED ENV{SROBOTIS_ROOT} AND NOT "$ENV{SROBOTIS_ROOT}" STREQUAL "")
  set(_las2_sdk_root "$ENV{SROBOTIS_ROOT}")
endif()
set(MPP_ROOT "${_las2_sdk_root}/components/multimedia/mpp" CACHE PATH "SDK MPP source headers")
find_path(LAS2_MPP_INCLUDE_DIR sys/sys_api.h
  PATHS "${MPP_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
set(DMABUFHEAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dmabufheap" CACHE PATH
  "Compatible allocator source, using the minimal licensed snapshot by default")
if(NOT EXISTS "${DMABUFHEAP_ROOT}/src/BufferAllocator.cpp" OR
   NOT EXISTS "${DMABUFHEAP_ROOT}/src/BufferAllocatorWrapper.cpp")
  message(FATAL_ERROR "DMABUFHEAP_ROOT must contain the compatible allocator sources")
endif()

add_library(las2_usb_stereo SHARED
  src/las2_usb_stereo.cpp
  "${DMABUFHEAP_ROOT}/src/BufferAllocator.cpp"
  "${DMABUFHEAP_ROOT}/src/BufferAllocatorWrapper.cpp")
add_library(stereo_matching::external_input ALIAS las2_usb_stereo)
target_compile_features(las2_usb_stereo PUBLIC cxx_std_17)
target_include_directories(las2_usb_stereo
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE src "${DMABUFHEAP_ROOT}/include" "${LAS2_ORT_INCLUDE_DIR}"
    "${LAS2_MPP_INCLUDE_DIR}" "${MPP_ROOT}/include/sys" "${MPP_ROOT}/include/uvc"
    "${MPP_ROOT}/include/vdec" "${MPP_ROOT}/include/stream"
    "${MPP_ROOT}/utils/include" "${MPP_ROOT}/al/include"
    ${OpenCV_INCLUDE_DIRS} ${LAS2_LIBDRM_INCLUDE_DIRS})
if(NOT LAS2_ENABLE_RVV_RECTIFICATION)
  target_compile_definitions(las2_usb_stereo PRIVATE LAS2_DISABLE_RVV_RECTIFICATION=1)
endif()
target_compile_options(las2_usb_stereo PRIVATE -O3 -Wall -Wextra -ffunction-sections -fdata-sections)
target_link_options(las2_usb_stereo PRIVATE -Wl,--gc-sections -Wl,--no-undefined)
target_link_libraries(las2_usb_stereo PRIVATE
  ${OpenCV_LIBS} OpenCL::OpenCL "${LAS2_ORT_LIBRARY}" "${LAS2_EP_LIBRARY}"
  Threads::Threads yaml-cpp)
set_target_properties(las2_usb_stereo PROPERTIES
  CXX_EXTENSIONS OFF VERSION 1.0.0 SOVERSION 1
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib" INSTALL_RPATH "$ORIGIN")
install(TARGETS las2_usb_stereo LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES SOURCE_PROVENANCE.json RUNTIME_CONTRACT.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching)
install(FILES third_party/dmabufheap/LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/licenses RENAME dmabufheap-LICENSE)
include("${CMAKE_CURRENT_LIST_DIR}/ExternalRuntime.cmake")
