# Install the selected runtime without exporting build-machine paths into the
# package. The default is one embedded library; ONNXRUNTIME_ROOT selects an
# explicitly supplied development/runtime artifact instead.
option(STEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME
  "Install the selected external ORT/EP runtime alongside ExternalStereo" ON)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _las2_runtime_processor)
set(_las2_target_requires_tcm FALSE)
if(_las2_runtime_processor MATCHES "riscv")
  set(_las2_target_requires_tcm TRUE)
endif()
option(STEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM
  "Require TCM for the selected runtime, including dlopen dependencies; always required on RISC-V"
  ${_las2_target_requires_tcm})
set(STEREO_MATCHING_EXTERNAL_INPUT_MODEL "" CACHE FILEPATH
  "Optional matching ExternalStereo model, installed separately from the tensor model")

if(STEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME)
  set(_las2_runtime_files)
  if(_las2_use_embedded_runtime)
    set(_las2_runtime_files "${LAS2_EMBEDDED_RUNTIME_LIBRARY}")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime/LICENSE")
      message(FATAL_ERROR "The bundled LAS2 runtime must include the ONNX Runtime license")
    endif()
    install(FILES thirdparty/onnxruntime/LICENSE
      DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/licenses
      RENAME onnxruntime-MIT.txt)
  else()
    set(_las2_required_runtime_libraries onnxruntime onnxruntime_providers_shared spacemit_ep)
    if(_las2_target_requires_tcm OR STEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM)
      list(APPEND _las2_required_runtime_libraries spine_tcm)
    endif()
    # Preserve a supplied TCM family even when optional. Its absence is valid on
    # generic runtimes, but declared dynamic dependencies must remain complete.
    foreach(_las2_runtime_library onnxruntime onnxruntime_providers_shared spacemit_ep spine_tcm)
      file(GLOB _las2_runtime_family LIST_DIRECTORIES FALSE
        "${ONNXRUNTIME_ROOT}/lib/lib${_las2_runtime_library}.so"
        "${ONNXRUNTIME_ROOT}/lib/lib${_las2_runtime_library}.so.*")
      if(NOT _las2_runtime_family)
        if(NOT _las2_runtime_library IN_LIST _las2_required_runtime_libraries)
          continue()
        endif()
        message(FATAL_ERROR
          "ExternalStereo runtime deployment requires lib${_las2_runtime_library}.so* in ONNXRUNTIME_ROOT/lib; "
          "select the complete matched runtime artifact or disable STEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME")
      endif()
      foreach(_las2_runtime_file IN LISTS _las2_runtime_family)
        if(NOT EXISTS "${_las2_runtime_file}")
          message(FATAL_ERROR "Broken external runtime symlink: ${_las2_runtime_file}")
        endif()
        if(IS_SYMLINK "${_las2_runtime_file}")
          file(READ_SYMLINK "${_las2_runtime_file}" _las2_runtime_link)
          if(IS_ABSOLUTE "${_las2_runtime_link}")
            message(FATAL_ERROR
              "The external runtime artifact must use relocatable relative symlinks: ${_las2_runtime_file}")
          endif()
        endif()
      endforeach()
      list(APPEND _las2_runtime_files ${_las2_runtime_family})
    endforeach()
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/LICENSE")
      message(FATAL_ERROR "The selected external runtime artifact must include LICENSE")
    endif()
  endif()
  # FILES preserves the artifact's relative ABI and dlopen-name symlinks.
  install(FILES ${_las2_runtime_files} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  set(_las2_deployed_checksums "${CMAKE_CURRENT_BINARY_DIR}/DEPLOYED_RUNTIME_SHA256SUMS")
  file(WRITE "${_las2_deployed_checksums}" "")
  foreach(_las2_runtime_file IN LISTS _las2_runtime_files)
    file(SHA256 "${_las2_runtime_file}" _las2_runtime_sha256)
    get_filename_component(_las2_runtime_name "${_las2_runtime_file}" NAME)
    file(APPEND "${_las2_deployed_checksums}"
      "${_las2_runtime_sha256}  ${CMAKE_INSTALL_LIBDIR}/${_las2_runtime_name}\n")
  endforeach()
  install(FILES "${_las2_deployed_checksums}"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/external_input/runtime)
  if(NOT _las2_use_embedded_runtime)
    install(FILES "${ONNXRUNTIME_ROOT}/LICENSE"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/licenses
      RENAME external-ort-LICENSE)
    install(FILES "${ONNXRUNTIME_ROOT}/SHA256SUMS"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/external_input/runtime
      RENAME SOURCE_SHA256SUMS OPTIONAL)
    install(FILES "${ONNXRUNTIME_ROOT}/README.vendor.md"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/external_input/runtime OPTIONAL)
  endif()
endif()

if(STEREO_MATCHING_EXTERNAL_INPUT_MODEL)
  if(NOT EXISTS "${STEREO_MATCHING_EXTERNAL_INPUT_MODEL}")
    message(FATAL_ERROR "STEREO_MATCHING_EXTERNAL_INPUT_MODEL does not exist")
  endif()
  install(FILES "${STEREO_MATCHING_EXTERNAL_INPUT_MODEL}"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/stereo_matching/external_input/model)
endif()
