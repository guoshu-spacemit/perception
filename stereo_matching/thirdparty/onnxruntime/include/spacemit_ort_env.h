// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include <string>
#include <unordered_map>

#include "onnxruntime_cxx_api.h"
// clang-format off
#ifndef SpaceMITPROVIDER_VERSION
#define SpaceMITPROVIDER_VERSION "2.0.4"
#define SpaceMITPROVIDER_BUILD_DATE "2026-06-17"
#endif

namespace Ort {
Status SessionOptionsSpaceMITEnvInit(
    SessionOptions & options,
    const std::unordered_map<std::string, std::string> provider_options = {});

}  // namespace Ort

// clang-format on
