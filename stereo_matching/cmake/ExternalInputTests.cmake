# These CPU tests do not instantiate inference, capture or display. The shared
# rectification probe uses the existing tensor facade declarations only so it
# needs no private ORT headers or RISC-V vendor binary on a development host.
enable_testing()
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
get_filename_component(_las2_test_sdk_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(DEFINED ENV{SROBOTIS_ROOT} AND NOT "$ENV{SROBOTIS_ROOT}" STREQUAL "")
  set(_las2_test_sdk_root "$ENV{SROBOTIS_ROOT}")
endif()
set(MPP_ROOT "${_las2_test_sdk_root}/components/multimedia/mpp" CACHE PATH "SDK MPP source headers")

add_executable(test_rectify_map_ops tests/test_rectify_map_ops.cpp)
target_include_directories(test_rectify_map_ops PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_compile_features(test_rectify_map_ops PRIVATE cxx_std_17)
target_link_libraries(test_rectify_map_ops PRIVATE ${OpenCV_LIBS} Threads::Threads)
add_test(NAME rectify_map_ops COMMAND test_rectify_map_ops)

add_executable(test_shared_rectification tests/test_shared_rectification.cpp)
target_include_directories(test_shared_rectification PRIVATE
  src include thirdparty/dmabufheap/include ${OpenCV_INCLUDE_DIRS}
  "${MPP_ROOT}/include" "${MPP_ROOT}/include/sys")
target_compile_definitions(test_shared_rectification PRIVATE LAS2_WITH_STEREO_MATCHING=1)
target_compile_features(test_shared_rectification PRIVATE cxx_std_17)
target_compile_options(test_shared_rectification PRIVATE -ffunction-sections -fdata-sections)
target_link_options(test_shared_rectification PRIVATE -Wl,--gc-sections)
target_link_libraries(test_shared_rectification PRIVATE
  ${OpenCV_LIBS} OpenCL::OpenCL Threads::Threads yaml-cpp ${CMAKE_DL_LIBS})
add_test(NAME shared_rectification COMMAND test_shared_rectification)

add_executable(test_opencl_profiling tests/test_opencl_profiling.cpp)
target_include_directories(test_opencl_profiling PRIVATE src)
target_compile_features(test_opencl_profiling PRIVATE cxx_std_17)
target_link_libraries(test_opencl_profiling PRIVATE OpenCL::OpenCL)
add_test(NAME opencl_profiling COMMAND test_opencl_profiling)
set_tests_properties(rectify_map_ops shared_rectification opencl_profiling
  PROPERTIES LABELS "cpu" TIMEOUT 120)

# Explicit GPU probes are compiled on request but never run by ordinary CTest.
# Run them on the target with src/stereo_preprocessing.h as their argument.
foreach(probe imagenet_opencl fp16_opencl preview_opencl)
  add_executable(test_${probe} tests/test_${probe}.cpp)
  target_compile_features(test_${probe} PRIVATE cxx_std_17)
  target_link_libraries(test_${probe} PRIVATE OpenCL::OpenCL)
endforeach()
