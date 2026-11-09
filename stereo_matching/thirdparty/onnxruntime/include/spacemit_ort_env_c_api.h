// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime_c_api.h"

// clang-format off
#ifndef SpaceMITPROVIDER_VERSION
#define SpaceMITPROVIDER_VERSION "2.0.4"
#define SpaceMITPROVIDER_BUILD_DATE "2026-06-17"
#endif

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus * ORT_API_CALL OrtSessionOptionsSpaceMITEnvInit(
                OrtSessionOptions *                       options,
                _In_reads_(num_keys) const char * const * provider_options_keys,
                _In_reads_(num_keys) const char * const * provider_options_values,
                size_t                                    num_keys);

#ifdef __cplusplus
}
#endif
// clang-format on
