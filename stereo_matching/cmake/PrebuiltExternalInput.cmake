# Keep the installed linker name and the ELF loader's SONAME available. Resolve
# input symlinks first so a selected .so link cannot become a dangling install.
get_filename_component(_las2_prebuilt_backend
  "${STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY}" REALPATH)
if(CMAKE_READELF)
  set(_las2_prebuilt_readelf "${CMAKE_READELF}")
else()
  find_program(_las2_prebuilt_readelf NAMES readelf llvm-readelf)
endif()
if(NOT _las2_prebuilt_readelf)
  message(FATAL_ERROR "A readelf tool is required to inspect the prebuilt ExternalStereo ELF SONAME")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env LC_ALL=C "${_las2_prebuilt_readelf}"
    -d "${_las2_prebuilt_backend}"
  RESULT_VARIABLE _las2_prebuilt_readelf_result
  OUTPUT_VARIABLE _las2_prebuilt_dynamic
  ERROR_VARIABLE _las2_prebuilt_readelf_error)
if(NOT _las2_prebuilt_readelf_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect prebuilt ExternalStereo ELF: ${_las2_prebuilt_readelf_error}")
endif()
string(REGEX MATCH "\\(SONAME\\)[^\n]*\\[([^]]+)\\]"
  _las2_prebuilt_soname_entry "${_las2_prebuilt_dynamic}")
set(_las2_prebuilt_soname "${CMAKE_MATCH_1}")
if(_las2_prebuilt_soname AND NOT
   _las2_prebuilt_soname MATCHES "^libstereo_matching_pipeline\\.so(\\.[0-9]+)*$")
  message(FATAL_ERROR
    "Prebuilt ExternalStereo SONAME must use the libstereo_matching_pipeline.so family; "
    "found ${_las2_prebuilt_soname}. Renaming a file does not change its ELF SONAME.")
endif()
install(FILES "${_las2_prebuilt_backend}"
  DESTINATION ${CMAKE_INSTALL_LIBDIR} RENAME libstereo_matching_pipeline.so)
if(_las2_prebuilt_soname AND NOT
   _las2_prebuilt_soname STREQUAL "libstereo_matching_pipeline.so")
  install(FILES "${_las2_prebuilt_backend}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR} RENAME "${_las2_prebuilt_soname}")
endif()
