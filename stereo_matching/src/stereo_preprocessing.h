#ifndef STEREO_PREPROCESSING_H
#define STEREO_PREPROCESSING_H

#include "stereo_runtime.h"

extern "C"
{
#include "sys/sys_type.h"
#include "sys/vb_type.h"
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

#include "opencl_profiling.h"
#include "rectification_map_ops.h"
#include "shared_initialization_cache.h"

namespace
{

struct StereoPipelineOptions
{
    std::string dev;
    std::string model;
    std::string calib;
    std::string affinities = "8,9,10,11,12,13,14,15";
    std::string uvc_format = "yuyv";
    int instances = 8;
    int width = 4000;
    int height = 1200;
    int fps = 30;
    int crop_x = 160;
    int source_eye_width = 0, source_eye_height = 0;
    int left_y = 0, right_y = 0;
    int left_x = -1;
    int right_x = -1;
    int rect_w = 1920;
    int rect_h = 1200;
    int roi_mask_x = 0;
    double rectify_alpha = 0.0;
    bool nv12_image_sampler = false;
    bool swap_lr = true;
    bool use_remap = true;
};

bool usb_format_is_yuyv(const StereoPipelineOptions& opt)
{
    return opt.uvc_format == "yuyv" || opt.uvc_format == "YUYV";
}

struct RectifyMaps
{
    int width = 0;
    int height = 0;
    double focal_px = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline_m = 0.0;
    std::array<double, 3> depth_origin_in_left{};
    bool full_precision = false;
    std::array<double, 9> rectified_from_left{};
    std::vector<float> left_x;
    std::vector<float> left_y;
    std::vector<float> right_x;
    std::vector<float> right_y;
};

enum class KalibrDistortionModel
{
    kRadtan,
    kEquidistant,
};

const char* kalibr_distortion_model_name(KalibrDistortionModel model)
{
    switch (model)
    {
    case KalibrDistortionModel::kRadtan:
        return "pinhole+radtan";
    case KalibrDistortionModel::kEquidistant:
        return "pinhole+equidistant";
    }
    return "unknown";
}

std::vector<double> yaml_vector(
    const YAML::Node& node, const std::string& key, size_t expected_size)
{
    const YAML::Node values = node[key];
    check(!values || !values.IsSequence(), "missing calibration sequence: " + key);
    check(values.size() != expected_size,
        "unexpected calibration sequence size for " + key + ": " + std::to_string(values.size()));
    return values.as<std::vector<double>>();
}

cv::Mat kalibr_camera_matrix(const YAML::Node& camera, const std::string& name)
{
    const std::vector<double> intrinsics = yaml_vector(camera, "intrinsics", 4);
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = intrinsics[0];
    K.at<double>(1, 1) = intrinsics[1];
    K.at<double>(0, 2) = intrinsics[2];
    K.at<double>(1, 2) = intrinsics[3];
    check(camera["camera_model"].as<std::string>() != "pinhole",
        name + " camera_model must be pinhole");
    return K;
}

KalibrDistortionModel kalibr_distortion_model(const YAML::Node& camera, const std::string& name)
{
    check(!camera["distortion_model"], name + " is missing distortion_model");
    const std::string model = camera["distortion_model"].as<std::string>();
    if (model == "radtan")
    {
        return KalibrDistortionModel::kRadtan;
    }
    if (model == "equidistant")
    {
        return KalibrDistortionModel::kEquidistant;
    }
    throw std::runtime_error(name + " unsupported Kalibr distortion_model: " + model +
        " (expected radtan or equidistant)");
}

cv::Mat kalibr_distortion(const YAML::Node& camera)
{
    const std::vector<double> values = yaml_vector(camera, "distortion_coeffs", 4);
    cv::Mat distortion(1, 4, CV_64F);
    std::memcpy(distortion.ptr<double>(), values.data(), values.size() * sizeof(double));
    return distortion;
}

cv::Mat kalibr_transform(const YAML::Node& camera, const std::string& key)
{
    const YAML::Node rows = camera[key];
    check(!rows || !rows.IsSequence() || rows.size() != 4,
        "missing 4x4 calibration transform: " + key);
    cv::Mat transform(4, 4, CV_64F);
    for (size_t row = 0; row < 4; ++row)
    {
        check(!rows[row].IsSequence() || rows[row].size() != 4,
            "invalid row in calibration transform: " + key);
        for (size_t col = 0; col < 4; ++col)
        {
            transform.at<double>(static_cast<int>(row), static_cast<int>(col)) =
                rows[row][col].as<double>();
        }
    }
    return transform;
}

using las2::detail::inspect_remap;
using las2::detail::RemapValidity;

RectifyMaps make_rectify_maps(
    const YAML::Node& root, const std::string& path, int rect_w, int rect_h, double rectify_alpha)
{
    const YAML::Node cam0 = root["cam0"];
    const YAML::Node cam1 = root["cam1"];
    check(!cam0 || !cam1, "Kalibr YAML must contain cam0 and cam1");

    cv::Mat K1 = kalibr_camera_matrix(cam0, "cam0");
    cv::Mat K2 = kalibr_camera_matrix(cam1, "cam1");
    const KalibrDistortionModel cam0_model = kalibr_distortion_model(cam0, "cam0");
    const KalibrDistortionModel cam1_model = kalibr_distortion_model(cam1, "cam1");
    check(cam0_model != cam1_model, "cam0 and cam1 distortion models must match");
    const KalibrDistortionModel distortion_model = cam0_model;
    cv::Mat D1 = kalibr_distortion(cam0);
    cv::Mat D2 = kalibr_distortion(cam1);

    const std::vector<double> resolution0 = yaml_vector(cam0, "resolution", 2);
    const std::vector<double> resolution1 = yaml_vector(cam1, "resolution", 2);
    const int calib_w = static_cast<int>(std::lround(resolution0[0]));
    const int calib_h = static_cast<int>(std::lround(resolution0[1]));
    check(resolution0 != resolution1, "cam0 and cam1 calibration resolutions differ");

    // Kalibr defines T_cn_cnm1 as the transform from cam0 to cam1.
    const cv::Mat transform = kalibr_transform(cam1, "T_cn_cnm1");
    cv::Mat R = transform(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat T = transform(cv::Rect(3, 0, 1, 3)).clone();
    check(calib_w != rect_w || calib_h != rect_h,
        "calibration resolution is " + std::to_string(calib_w) + "x" + std::to_string(calib_h) +
            ", but rect size is " + std::to_string(rect_w) + "x" + std::to_string(rect_h));

    cv::Mat R1, R2, P1, P2, Q;
    const cv::Size image_size(rect_w, rect_h);
    cv::Mat map1x, map1y, map2x, map2y;
    double fisheye_fov_scale = 1.0;
    if (distortion_model == KalibrDistortionModel::kEquidistant)
    {
        check(rectify_alpha < 0.0 || rectify_alpha > 1.0,
            "equidistant rectify-alpha must be in [0, 1]");
        RemapValidity left_validity;
        RemapValidity right_validity;
        auto generate_fisheye_maps = [&](double fov_scale)
        {
            cv::fisheye::stereoRectify(K1, D1, K2, D2, image_size, R, T, R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, image_size, rectify_alpha, fov_scale);
            las2::detail::fisheye_maps(K1, D1, R1, P1, image_size, map1x, map1y);
            las2::detail::fisheye_maps(K2, D2, R2, P2, image_size, map2x, map2y);
            left_validity = inspect_remap(map1x, map1y, rect_w, rect_h);
            right_validity = inspect_remap(map2x, map2y, rect_w, rect_h);
            return left_validity.invalid == 0 && right_validity.invalid == 0;
        };

        bool fully_valid = generate_fisheye_maps(fisheye_fov_scale);

        // fisheye::stereoRectify does not return valid-pixel ROIs and
        // balance=0 does not guarantee that every pixel of both rectified
        // images maps inside its source.  Find the largest fov_scale that
        // makes both maps fully valid, then pass that value back through the
        // official fisheye API.  This retains the widest possible common FOV
        // instead of clamping invalid coordinates in the OpenCL kernel.
        if (!fully_valid)
        {
            double valid_scale = fisheye_fov_scale;
            double invalid_scale = fisheye_fov_scale;
            bool bracketed = false;
            for (int attempt = 0; attempt < 64; ++attempt)
            {
                valid_scale *= 0.98;
                if (generate_fisheye_maps(valid_scale))
                {
                    bracketed = true;
                    break;
                }
            }
            check(!bracketed, "could not find a fully valid equidistant stereo fov_scale");

            for (int iteration = 0; iteration < 16; ++iteration)
            {
                const double middle = (valid_scale + invalid_scale) * 0.5;
                if (generate_fisheye_maps(middle))
                {
                    valid_scale = middle;
                }
                else
                {
                    invalid_scale = middle;
                }
            }
            // A slightly smaller fov_scale means a slightly larger focal
            // length. Leave a numerical guard band for OpenCV version and
            // RG16F conversion differences across host and board builds.
            fisheye_fov_scale = valid_scale * 0.9995;
            check(!generate_fisheye_maps(fisheye_fov_scale),
                "automatic equidistant fov_scale did not produce fully "
                "valid maps");
        }

        std::printf("[CALIB_MAP] left_invalid=%zu right_invalid=%zu "
                    "left_range=[%.3f,%.3f]x[%.3f,%.3f] "
                    "right_range=[%.3f,%.3f]x[%.3f,%.3f] fov_scale=%.7f\n",
            left_validity.invalid, right_validity.invalid, left_validity.min_x, left_validity.max_x,
            left_validity.min_y, left_validity.max_y, right_validity.min_x, right_validity.max_x,
            right_validity.min_y, right_validity.max_y, fisheye_fov_scale);
    }
    else
    {
        cv::stereoRectify(K1, D1, K2, D2, image_size, R, T, R1, R2, P1, P2, Q,
            cv::CALIB_ZERO_DISPARITY, rectify_alpha, image_size);
        cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32FC1, map2x, map2y);
    }

    RectifyMaps maps;
    maps.width = rect_w;
    maps.height = rect_h;
    maps.focal_px = P1.at<double>(0, 0);
    maps.fx = P1.at<double>(0, 0);
    maps.fy = P1.at<double>(1, 1);
    maps.cx = P1.at<double>(0, 2);
    maps.cy = P1.at<double>(1, 2);
    maps.baseline_m = cv::norm(T);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            maps.rectified_from_left[static_cast<size_t>(row) * 3 + column] =
                R1.at<double>(row, column);
        }
    }
    const size_t n = static_cast<size_t>(rect_w) * rect_h;
    maps.left_x.assign(map1x.ptr<float>(), map1x.ptr<float>() + n);
    maps.left_y.assign(map1y.ptr<float>(), map1y.ptr<float>() + n);
    maps.right_x.assign(map2x.ptr<float>(), map2x.ptr<float>() + n);
    maps.right_y.assign(map2y.ptr<float>(), map2y.ptr<float>() + n);
    check(maps.left_x.size() != n || maps.left_y.size() != n || maps.right_x.size() != n ||
            maps.right_y.size() != n,
        "invalid remap table size");
    std::printf("[CALIB] %s model=%s rect=%dx%d alpha=%.3f maps=%zu "
                "fx=%.6fpx baseline=%.9fm\n",
        path.c_str(), kalibr_distortion_model_name(distortion_model), rect_w, rect_h, rectify_alpha,
        n, maps.focal_px, maps.baseline_m);
    return maps;
}

std::shared_ptr<const RectifyMaps> load_shared_rectify_maps(
    const std::string& path, int rect_w, int rect_h, double rectify_alpha)
{
    check(!std::isfinite(rectify_alpha), "rectify-alpha must be finite");
    std::ifstream file(path, std::ios::binary);
    check(!file, "could not open calibration: " + path);
    const std::string contents((std::istreambuf_iterator<char>(file)), {});
    check(file.bad(), "could not read calibration: " + path);
    // Key and parser use the SAME file snapshot: editing a file in place or
    // replacing it at the same path must never reuse the previous calibration.
    using Key = std::tuple<std::string, int, int, double>;
    static las2::detail::SharedInitializationCache<Key, RectifyMaps> cache;
    return cache.get(Key{contents, rect_w, rect_h, rectify_alpha}, [&]
        { return make_rectify_maps(YAML::Load(contents), path, rect_w, rect_h, rectify_alpha); });
}

uint16_t float_to_half_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0)
    {
        if (exp < -10)
        {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const uint32_t shifted = mant >> (static_cast<uint32_t>(1 - exp) + 13u);
        return static_cast<uint16_t>(sign | shifted);
    }
    if (exp >= 31)
    {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10u) | (mant >> 13u));
}

const char* kUsbPreprocessKernel = R"CLC(
inline float clamp255(float x) {
    return clamp(x, 0.0f, 255.0f);
}

inline float sample_y(__global const uchar* nv12, float fx, float fy, int y_stride, int max_x, int max_y) {
    fx = clamp(fx, 0.0f, (float)max_x);
    fy = clamp(fy, 0.0f, (float)max_y);
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, max_x);
    int y1 = min(y0 + 1, max_y);
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;
    float v00 = (float)nv12[y0 * y_stride + x0];
    float v01 = (float)nv12[y0 * y_stride + x1];
    float v10 = (float)nv12[y1 * y_stride + x0];
    float v11 = (float)nv12[y1 * y_stride + x1];
    return mix(mix(v00, v01, ax), mix(v10, v11, ax), ay);
}

inline float2 sample_uv(__global const uchar* nv12,
                        float fx, float fy,
                        int uv_stride, int uv_offset,
                        int frame_w, int frame_h) {
    int uv_w = frame_w >> 1;
    int uv_h = frame_h >> 1;
    float ux = clamp(fx * 0.5f, 0.0f, (float)(uv_w - 1));
    float uy = clamp(fy * 0.5f, 0.0f, (float)(uv_h - 1));
    int x0 = (int)floor(ux);
    int y0 = (int)floor(uy);
    int x1 = min(x0 + 1, uv_w - 1);
    int y1 = min(y0 + 1, uv_h - 1);
    float ax = ux - (float)x0;
    float ay = uy - (float)y0;
    int i00 = uv_offset + y0 * uv_stride + x0 * 2;
    int i01 = uv_offset + y0 * uv_stride + x1 * 2;
    int i10 = uv_offset + y1 * uv_stride + x0 * 2;
    int i11 = uv_offset + y1 * uv_stride + x1 * 2;
    float2 v00 = convert_float2(vload2(0, nv12 + i00));
    float2 v01 = convert_float2(vload2(0, nv12 + i01));
    float2 v10 = convert_float2(vload2(0, nv12 + i10));
    float2 v11 = convert_float2(vload2(0, nv12 + i11));
    return mix(mix(v00, v01, ax), mix(v10, v11, ax), ay);
}

inline void sample_nv12_rgb(__global const uchar* nv12,
                            float fx,
                            float fy,
                            int y_stride,
                            int uv_stride,
                            int uv_offset,
                            int max_x,
                            int max_y,
                            float* r,
                            float* g,
                            float* b) {
    fx = clamp(fx, 0.0f, (float)max_x);
    fy = clamp(fy, 0.0f, (float)max_y);
    float yv = sample_y(nv12, fx, fy, y_stride, max_x, max_y);
    float2 uv = sample_uv(nv12, fx, fy, uv_stride, uv_offset, max_x + 1, max_y + 1);
    float u = uv.x;
    float v = uv.y;
    float c = yv - 16.0f;
    float d = u - 128.0f;
    float e = v - 128.0f;
    *r = clamp255(1.16438356f * c + 1.79274107f * e);
    *g = clamp255(1.16438356f * c - 0.21324861f * d - 0.53290933f * e);
    *b = clamp255(1.16438356f * c + 2.11240179f * d);
}

inline void sample_nv12_image(read_only image2d_t y_image,
                                read_only image2d_t uv_image,
                                float fx, float fy,
                                float* r, float* g, float* b) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    float yv = read_imagef(y_image, sampler, (float2)(fx + 0.5f, fy + 0.5f)).x * 255.0f;
    float2 uv = read_imagef(uv_image, sampler,
                            (float2)(fx * 0.5f + 0.5f, fy * 0.5f + 0.5f)).xy * 255.0f;
    float c = yv - 16.0f;
    float d = uv.x - 128.0f;
    float e = uv.y - 128.0f;
    *r = clamp255(1.16438356f * c + 1.79274107f * e);
    *g = clamp255(1.16438356f * c - 0.21324861f * d - 0.53290933f * e);
    *b = clamp255(1.16438356f * c + 2.11240179f * d);
}

inline void yuyv_pixel_rgb(__global const uchar* yuyv,
                            int x,
                            int y,
                            int stride,
                            int max_x,
                            int max_y,
                            float* r,
                            float* g,
                            float* b) {
    x = clamp(x, 0, max_x);
    y = clamp(y, 0, max_y);
    int pair_x = x & ~1;
    int base = y * stride + pair_x * 2;
    float yv = (float)yuyv[base + ((x & 1) ? 2 : 0)];
    float u = (float)yuyv[base + 1];
    float v = (float)yuyv[base + 3];
    float c = yv - 16.0f;
    float d = u - 128.0f;
    float e = v - 128.0f;
    *r = clamp255(1.16438356f * c + 1.79274107f * e);
    *g = clamp255(1.16438356f * c - 0.21324861f * d - 0.53290933f * e);
    *b = clamp255(1.16438356f * c + 2.11240179f * d);
}

inline void sample_yuyv_rgb(__global const uchar* yuyv,
                            float fx,
                            float fy,
                            int stride,
                            int max_x,
                            int max_y,
                            float* r,
                            float* g,
                            float* b) {
    fx = clamp(fx, 0.0f, (float)max_x);
    fy = clamp(fy, 0.0f, (float)max_y);
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, max_x);
    int y1 = min(y0 + 1, max_y);
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;
    float r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
    yuyv_pixel_rgb(yuyv, x0, y0, stride, max_x, max_y, &r00, &g00, &b00);
    yuyv_pixel_rgb(yuyv, x1, y0, stride, max_x, max_y, &r01, &g01, &b01);
    yuyv_pixel_rgb(yuyv, x0, y1, stride, max_x, max_y, &r10, &g10, &b10);
    yuyv_pixel_rgb(yuyv, x1, y1, stride, max_x, max_y, &r11, &g11, &b11);
    *r = mix(mix(r00, r01, ax), mix(r10, r11, ax), ay);
    *g = mix(mix(g00, g01, ax), mix(g10, g11, ax), ay);
    *b = mix(mix(b00, b01, ax), mix(b10, b11, ax), ay);
}

#ifdef LAS2_TENSOR_FP16
typedef half tensor_scalar;
inline void store_tensor(float value, int index, __global tensor_scalar* output) {
    // Half storage and vstore_half are core OpenCL functionality; arithmetic
    // stays float32 and does not require cl_khr_fp16 on the device.
    vstore_half_rte(value, index, output);
}
#else
typedef float tensor_scalar;
inline void store_tensor(float value, int index, __global tensor_scalar* output) {
    output[index] = value;
}
#endif

__kernel void usb_nv12_rectify_to_las2(__global const uchar* nv12,
                                        read_only image2d_t y_image,
                                        read_only image2d_t uv_image,
                                        int use_image_sampler,
                                        __global tensor_scalar* left_out,
                                        __global tensor_scalar* right_out,
                                        __global uint* left_xrgb,
                                        __global uint* right_xrgb,
                                        read_only image2d_t left_map,
                                        read_only image2d_t right_map,
                                        int frame_w,
                                        int frame_h,
                                        int y_stride,
                                        int uv_stride,
                                        int uv_offset,
                                        int input_format,
                                        int left_src_x,
                                        int right_src_x,
                                        int left_src_y, int right_src_y,
                                        int source_eye_w, int source_eye_h,
                                        int rect_w,
                                        int rect_h,
                                        int roi_mask_x,
                                        int model_w,
                                        int model_h,
                                        int scaled_w,
                                        int scaled_h,
                                        int pad_left,
                                        int pad_top,
                                        int nhwc,
                                        int swap_lr,
                                        int use_remap) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= model_w || y >= model_h) return;

    float lr = 0.0f, lg = 0.0f, lb = 0.0f;
    float rr = 0.0f, rg = 0.0f, rb = 0.0f;
    // Preserve subpixel filtering of the RG16F remap texture in FP32.
    float rx = 0.0f;
    float ry = 0.0f;
    int valid = x >= pad_left && x < pad_left + scaled_w &&
                y >= pad_top && y < pad_top + scaled_h;
    if (valid) {
        rx = clamp((((float)(x - pad_left) + 0.5f) * (float)rect_w / (float)scaled_w) - 0.5f,
                    0.0f, (float)(rect_w - 1));
        ry = clamp((((float)(y - pad_top) + 0.5f) * (float)rect_h / (float)scaled_h) - 0.5f,
                    0.0f, (float)(rect_h - 1));
        if (rx < roi_mask_x || rx >= rect_w - roi_mask_x) {
            valid = 0;
        }
    }
    if (valid) {
        const sampler_t map_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
        int left_base = swap_lr ? right_src_x : left_src_x;
        int right_base = swap_lr ? left_src_x : right_src_x;
        float2 map_coord = (float2)(rx + 0.5f, ry + 0.5f);
        float4 lmap = use_remap ? read_imagef(left_map, map_sampler, map_coord)
                                : (float4)(rx, ry, 0.0f, 0.0f);
        float4 rmap = use_remap ? read_imagef(right_map, map_sampler, map_coord)
                                : (float4)(rx, ry, 0.0f, 0.0f);
        int lvalid = lmap.x >= 0.0f && lmap.x <= (float)(source_eye_w - 1) &&
                        lmap.y >= 0.0f && lmap.y <= (float)(source_eye_h - 1);
        int rvalid = rmap.x >= 0.0f && rmap.x <= (float)(source_eye_w - 1) &&
                        rmap.y >= 0.0f && rmap.y <= (float)(source_eye_h - 1);
        if (lvalid) {
            float lsx = (float)left_base + lmap.x;
            float lsy = lmap.y + (swap_lr ? right_src_y : left_src_y);
            if (input_format == 1) {
                sample_yuyv_rgb(nv12, lsx, lsy, y_stride, frame_w - 1, frame_h - 1, &lr, &lg, &lb);
            } else if (use_image_sampler) {
                sample_nv12_image(y_image, uv_image, lsx, lsy, &lr, &lg, &lb);
            } else {
                sample_nv12_rgb(nv12, lsx, lsy, y_stride, uv_stride, uv_offset, frame_w - 1, frame_h - 1,
                                &lr, &lg, &lb);
            }
        }
        if (rvalid) {
            float rsx = (float)right_base + rmap.x;
            float rsy = rmap.y + (swap_lr ? left_src_y : right_src_y);
            if (input_format == 1) {
                sample_yuyv_rgb(nv12, rsx, rsy, y_stride, frame_w - 1, frame_h - 1, &rr, &rg, &rb);
            } else if (use_image_sampler) {
                sample_nv12_image(y_image, uv_image, rsx, rsy, &rr, &rg, &rb);
            } else {
                sample_nv12_rgb(nv12, rsx, rsy, y_stride, uv_stride, uv_offset, frame_w - 1, frame_h - 1,
                                &rr, &rg, &rb);
            }
        }
    }

    int idx = y * model_w + x;
    int plane = model_w * model_h;
    // Preview stays in RGB byte range; normalize only inference tensors.
    uint lp = 0xff000000u | (((uint)(lr + 0.5f)) << 16) | (((uint)(lg + 0.5f)) << 8) | ((uint)(lb + 0.5f));
    uint rp = 0xff000000u | (((uint)(rr + 0.5f)) << 16) | (((uint)(rg + 0.5f)) << 8) | ((uint)(rb + 0.5f));
    lr = (lr / 255.0f - 0.485f) / 0.229f;
    lg = (lg / 255.0f - 0.456f) / 0.224f;
    lb = (lb / 255.0f - 0.406f) / 0.225f;
    rr = (rr / 255.0f - 0.485f) / 0.229f;
    rg = (rg / 255.0f - 0.456f) / 0.224f;
    rb = (rb / 255.0f - 0.406f) / 0.225f;
    if (nhwc) {
        int o = idx * 3;
        store_tensor(lr, o, left_out);
        store_tensor(lg, o + 1, left_out);
        store_tensor(lb, o + 2, left_out);
        store_tensor(rr, o, right_out);
        store_tensor(rg, o + 1, right_out);
        store_tensor(rb, o + 2, right_out);
    } else {
        store_tensor(lr, idx, left_out);
        store_tensor(lg, idx + plane, left_out);
        store_tensor(lb, idx + plane * 2, left_out);
        store_tensor(rr, idx, right_out);
        store_tensor(rg, idx + plane, right_out);
        store_tensor(rb, idx + plane * 2, right_out);
    }
    left_xrgb[idx] = lp;
    right_xrgb[idx] = rp;
}
)CLC";

const char* kDepthPreviewKernel = R"CLC(
inline float turbo_channel(float x, float c0, float c1, float c2,
                            float c3, float c4, float c5) {
    return clamp((((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0),
                    0.0f, 1.0f);
}

inline float3 unpack_xrgb(uint packed) {
    return (float3)((float)((packed >> 16) & 255u),
                    (float)((packed >> 8) & 255u),
                    (float)(packed & 255u));
}

inline float3 depth_rgb(float value, float depth_min_m, float depth_max_m,
                        int disparity_mode) {
    if (!isfinite(value) ||
        (disparity_mode ? value <= 0.0f :
                            (value <= depth_min_m || value > depth_max_m))) {
        return (float3)(0.0f);
    }
    const float t = clamp((value - depth_min_m) / (depth_max_m - depth_min_m), 0.0f, 1.0f);
    return (float3)(
        turbo_channel(t, 0.13572138f, 4.61539260f, -42.66032258f,
                        132.13108234f, -152.94239396f, 59.28637943f),
        turbo_channel(t, 0.09140261f, 2.19418839f, 4.84296658f,
                        -14.18503333f, 4.27729857f, 2.82956604f),
        turbo_channel(t, 0.10667330f, 12.64194608f, -60.58204836f,
                        110.36276771f, -89.90310912f, 27.34824973f)) * 255.0f;
}

inline uchar rgb_y(float3 rgb) {
    return convert_uchar_sat_rte(0.257f * rgb.x + 0.504f * rgb.y +
                                    0.098f * rgb.z + 16.0f);
}

inline uchar rgb_u(float3 rgb) {
    return convert_uchar_sat_rte(-0.148f * rgb.x - 0.291f * rgb.y +
                                    0.439f * rgb.z + 128.0f);
}

inline uchar rgb_v(float3 rgb) {
    return convert_uchar_sat_rte(0.439f * rgb.x - 0.368f * rgb.y -
                                    0.071f * rgb.z + 128.0f);
}

__kernel void compose_nv12_depth(__global const uint* left_xrgb,
                                    __global const float* depth,
                                    __global uchar* output,
                                    int width, int height, int y_stride,
                                    int uv_stride, float depth_min_m,
                                    float depth_max_m, int disparity_mode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int input_index = y * width + x;
    const float3 left = unpack_xrgb(left_xrgb[input_index]);
    const float3 colored_depth = depth_rgb(depth[input_index], depth_min_m, depth_max_m,
                                            disparity_mode);
    output[y * y_stride + x] = rgb_y(left);
    output[y * y_stride + width + x] = rgb_y(colored_depth);
    if ((x & 1) == 0 && (y & 1) == 0) {
        float3 left_sum = (float3)(0.0f);
        float3 depth_sum = (float3)(0.0f);
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int sample = (y + dy) * width + x + dx;
                left_sum += unpack_xrgb(left_xrgb[sample]);
                depth_sum += depth_rgb(depth[sample], depth_min_m, depth_max_m,
                                        disparity_mode);
            }
        }
        const int uv_base = y_stride * height + (y / 2) * uv_stride;
        const float3 left_avg = left_sum * 0.25f;
        const float3 depth_avg = depth_sum * 0.25f;
        output[uv_base + x] = rgb_u(left_avg);
        output[uv_base + x + 1] = rgb_v(left_avg);
        output[uv_base + width + x] = rgb_u(depth_avg);
        output[uv_base + width + x + 1] = rgb_v(depth_avg);
    }
}
__kernel void pack_rgb(__global const uint* input, __global uchar* output,
                        int width, int height, int stride) {
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=width || y>=height) return;
    uint p=input[y*width+x]; int offset=y*stride+3*x;
    output[offset]=(uchar)(p>>16); output[offset+1]=(uchar)(p>>8); output[offset+2]=(uchar)p;
}
)CLC";

class StereoPreprocessorOpenCl
{
    struct Nv12Images
    {
        cl_mem y = nullptr;
        cl_mem uv_sub = nullptr;
        cl_mem uv = nullptr;
        int width = 0, height = 0, y_stride = 0, uv_stride = 0, uv_offset = 0;
    };

    static void release_nv12_images(Nv12Images& images)
    {
        if (images.uv)
            clReleaseMemObject(images.uv);
        if (images.uv_sub)
            clReleaseMemObject(images.uv_sub);
        if (images.y)
            clReleaseMemObject(images.y);
        images = {};
    }

public:
    explicit StereoPreprocessorOpenCl(const RectifyMaps& maps, bool tensor_fp16 = false)
    {
        cl_int err = clGetPlatformIDs(1, &platform_, nullptr);
        check(err != CL_SUCCESS, "clGetPlatformIDs failed");
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
        check(err != CL_SUCCESS, "clGetDeviceIDs GPU failed");
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        check(!context_ || err != CL_SUCCESS, "clCreateContext failed");
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
        check(!queue_ || err != CL_SUCCESS, "clCreateCommandQueueWithProperties failed");
        import_memory_ = reinterpret_cast<PFN_clImportMemoryARM>(
            clGetExtensionFunctionAddressForPlatform(platform_, "clImportMemoryARM"));
        acquire_ = reinterpret_cast<PFN_clEnqueueAcquireExternalMemObjectsKHR>(
            clGetExtensionFunctionAddressForPlatform(
                platform_, "clEnqueueAcquireExternalMemObjectsKHR"));
        release_ = reinterpret_cast<PFN_clEnqueueReleaseExternalMemObjectsKHR>(
            clGetExtensionFunctionAddressForPlatform(
                platform_, "clEnqueueReleaseExternalMemObjectsKHR"));
        check(!import_memory_, "clImportMemoryARM not available");
        check(!acquire_ || !release_, "external memory acquire/release not available");

        const char* sources[] = {kUsbPreprocessKernel, kDepthPreviewKernel};
        const size_t lengths[] = {
            std::strlen(kUsbPreprocessKernel), std::strlen(kDepthPreviewKernel)};
        program_ = clCreateProgramWithSource(context_, 2, sources, lengths, &err);
        check(!program_ || err != CL_SUCCESS, "clCreateProgramWithSource failed");
        err = clBuildProgram(
            program_, 1, &device_, tensor_fp16 ? "-DLAS2_TENSOR_FP16=1" : "", nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(
                program_, device_, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
            throw std::runtime_error("clBuildProgram failed:\n" + log);
        }
        kernel_ = clCreateKernel(program_, "usb_nv12_rectify_to_las2", &err);
        check(!kernel_ || err != CL_SUCCESS, "clCreateKernel failed");
        depth_preview_kernel_ = clCreateKernel(program_, "compose_nv12_depth", &err);
        check(!depth_preview_kernel_ || err != CL_SUCCESS,
            "clCreateKernel compose_nv12_depth failed");
        rgb_kernel_ = clCreateKernel(program_, "pack_rgb", &err);
        check(!rgb_kernel_ || err != CL_SUCCESS, "pack_rgb kernel failed");
        left_map_ = create_map_image(
            maps.left_x, maps.left_y, maps.width, maps.height, maps.full_precision);
        right_map_ = create_map_image(
            maps.right_x, maps.right_y, maps.width, maps.height, maps.full_precision);
        dummy_y_ = create_dummy_image(CL_R);
        dummy_uv_ = create_dummy_image(CL_RG);
        unused_input_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, 16, nullptr, &err);
        check(!unused_input_ || err != CL_SUCCESS, "create unused buffer argument failed");
    }

    ~StereoPreprocessorOpenCl()
    {
        if (queue_)
            clFinish(queue_);
        if (disparity_preview_)
            clReleaseMemObject(disparity_preview_);
        for (auto& item : image_cache_)
            release_nv12_images(item.second);
        for (auto& item : input_cache_)
        {
            if (item.second)
                clReleaseMemObject(item.second);
        }
        if (dummy_y_)
            clReleaseMemObject(dummy_y_);
        if (dummy_uv_)
            clReleaseMemObject(dummy_uv_);
        if (unused_input_)
            clReleaseMemObject(unused_input_);
        if (left_map_)
            clReleaseMemObject(left_map_);
        if (right_map_)
            clReleaseMemObject(right_map_);
        if (depth_preview_kernel_)
            clReleaseKernel(depth_preview_kernel_);
        if (rgb_kernel_)
            clReleaseKernel(rgb_kernel_);
        if (kernel_)
            clReleaseKernel(kernel_);
        if (program_)
            clReleaseProgram(program_);
        if (queue_)
            clReleaseCommandQueue(queue_);
        if (context_)
            clReleaseContext(context_);
    }

    cl_mem import_dmabuf(int fd, size_t size)
    {
        const cl_import_properties_arm props[] = {
            CL_IMPORT_TYPE_ARM,
            CL_IMPORT_TYPE_DMA_BUF_ARM,
            0,
        };
        cl_int err = CL_SUCCESS;
        cl_mem mem = import_memory_(context_, CL_MEM_READ_WRITE, props, &fd, size, &err);
        check(!mem || err != CL_SUCCESS, "clImportMemoryARM failed: " + std::to_string(err));
        return mem;
    }

    cl_mem import_input_dmabuf_cached(int fd, size_t size)
    {
        auto it = input_cache_.find(fd);
        if (it != input_cache_.end())
        {
            return it->second;
        }
        cl_mem mem = import_dmabuf(fd, size);
        input_cache_.emplace(fd, mem);
        std::printf("[OpenCL] cache input dmabuf fd=%d size=%zu cached=%zu\n", fd, size,
            input_cache_.size());
        return mem;
    }

    void release_input_dmabuf_cached(int fd)
    {
        auto it = input_cache_.find(fd);
        if (it == input_cache_.end())
        {
            return;
        }
        clFinish(queue_);
        auto images = image_cache_.find(it->second);
        if (images != image_cache_.end())
        {
            release_nv12_images(images->second);
            image_cache_.erase(images);
        }
        if (it->second)
        {
            clReleaseMemObject(it->second);
        }
        input_cache_.erase(it);
    }

    double run(cl_mem input, cl_mem left_tensor, cl_mem right_tensor, cl_mem left_preview,
        cl_mem right_preview, const VideoFrameInfo& frame, const StereoPipelineOptions& opt,
        int model_w, int model_h, bool nhwc, int scaled_w, int scaled_h, int pad_left, int pad_top)
    {
        const auto wall_start = std::chrono::steady_clock::now();
        const bool is_yuyv = usb_format_is_yuyv(opt);
        const int frame_w = is_yuyv ?
            static_cast<int>(frame.stCommFrameInfo.u32Width) :
            static_cast<int>(frame.stVdecFrameInfo.stCommFrameInfo.u32Width);
        const int frame_h = is_yuyv ?
            static_cast<int>(frame.stCommFrameInfo.u32Height) :
            static_cast<int>(frame.stVdecFrameInfo.stCommFrameInfo.u32Height);
        const int raw_y_stride = static_cast<int>(frame.stVFrame.u32PlaneStride[0]);
        const int y_stride =
            is_yuyv ? ((raw_y_stride >= frame_w * 2) ? raw_y_stride : frame_w * 2) : raw_y_stride;
        const int uv_stride = is_yuyv ?
            0 :
            (frame.stVFrame.u32PlaneStride[1] ? static_cast<int>(frame.stVFrame.u32PlaneStride[1]) :
                                                y_stride);
        const int uv_offset = is_yuyv ?
            0 :
            (frame.stVFrame.ulPlaneVirAddr[1] ?
                    static_cast<int>(reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(
                                            frame.stVFrame.ulPlaneVirAddr[1])) -
                        reinterpret_cast<uintptr_t>(
                            reinterpret_cast<void*>(frame.stVFrame.ulPlaneVirAddr[0]))) :
                    y_stride * frame_h);
        const int nhwc_i = nhwc ? 1 : 0;
        const int swap_i = opt.swap_lr ? 1 : 0;
        const int use_remap_i = opt.use_remap ? 1 : 0;
        const int input_format = is_yuyv ? 1 : 0;
        cl_mem y_image = dummy_y_;
        cl_mem uv_image = dummy_uv_;
        int use_image_sampler = 0;
        if (!is_yuyv && opt.nv12_image_sampler && image_path_supported_)
        {
            const auto& images =
                cached_nv12_images(input, frame_w, frame_h, y_stride, uv_stride, uv_offset);
            if (images.y && images.uv)
            {
                y_image = images.y;
                uv_image = images.uv;
                use_image_sampler = 1;
            }
        }

        cl_mem mems[5] = {input, left_tensor, right_tensor, left_preview, right_preview};
        cl_int err = acquire_(queue_, 5, mems, 0, nullptr, nullptr);
        check(err != CL_SUCCESS, "acquire external memory failed: " + std::to_string(err));
        const auto acquired_at = std::chrono::steady_clock::now();
        int arg = 0;
        // Do not bind the same imported allocation as both a buffer argument
        // and image aliases. On K3 that retriggers ~1.6 s driver synchronization
        // every dispatch, even though the buffer branch is not executed.
        // Ownership transfer below still covers the actual imported parent.
        const cl_mem buffer_argument = use_image_sampler ? unused_input_ : input;
        err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &buffer_argument);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &y_image);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &uv_image);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &use_image_sampler);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &left_tensor);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &right_tensor);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &left_preview);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &right_preview);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &left_map_);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &right_map_);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &frame_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &frame_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &y_stride);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &uv_stride);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &uv_offset);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &input_format);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.left_x);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.right_x);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.left_y);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.right_y);
        const int source_eye_w = opt.source_eye_width ? opt.source_eye_width : opt.rect_w;
        const int source_eye_h = opt.source_eye_height ? opt.source_eye_height : opt.rect_h;
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &source_eye_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &source_eye_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.rect_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.rect_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &opt.roi_mask_x);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &model_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &model_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &scaled_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &scaled_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &pad_left);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &pad_top);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &nhwc_i);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &swap_i);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &use_remap_i);
        check(err != CL_SUCCESS, "clSetKernelArg failed");
        size_t global[2] = {static_cast<size_t>(model_w), static_cast<size_t>(model_h)};
        cl_event event = nullptr;
        err = clEnqueueNDRangeKernel(
            queue_, kernel_, 2, nullptr, global, nullptr, 0, nullptr, &event);
        check(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed");
        err = release_(queue_, 5, mems, 0, nullptr, nullptr);
        check(err != CL_SUCCESS, "release external memory failed: " + std::to_string(err));
        check(clFinish(queue_) != CL_SUCCESS, "preprocess finish failed");
        const auto finished_at = std::chrono::steady_clock::now();
        const double elapsed_ms = las2::detail::opencl_event_ms(event);
        clReleaseEvent(event);
        if ((use_image_sampler || profile_preprocess_) && image_debug_count_++ < 16)
        {
            const double setup_ms =
                std::chrono::duration<double, std::milli>(acquired_at - wall_start).count();
            const double execute_ms =
                std::chrono::duration<double, std::milli>(finished_at - acquired_at).count();
            std::printf("[OpenCL][TENSOR_TIMING] image_sampler=%d setup_acquire=%.3fms "
                        "execute_release_finish=%.3fms kernel=%.3fms\n",
                use_image_sampler, setup_ms, execute_ms, elapsed_ms);
        }
        return elapsed_ms;
    }

    void pack_rgb(cl_mem input, int output_fd, size_t bytes, int width, int height, int stride)
    {
        cl_mem output = import_dmabuf(output_fd, bytes);
        cl_mem buffers[] = {input, output};
        cl_int err = acquire_(queue_, 2, buffers, 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseMemObject(output);
            throw std::runtime_error("RGB acquire failed");
        }
        err = clSetKernelArg(rgb_kernel_, 0, sizeof(input), &input);
        err |= clSetKernelArg(rgb_kernel_, 1, sizeof(output), &output);
        err |= clSetKernelArg(rgb_kernel_, 2, sizeof(width), &width);
        err |= clSetKernelArg(rgb_kernel_, 3, sizeof(height), &height);
        err |= clSetKernelArg(rgb_kernel_, 4, sizeof(stride), &stride);
        size_t global[] = {static_cast<size_t>(width), static_cast<size_t>(height)};
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(
                queue_, rgb_kernel_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        const auto release_error = release_(queue_, 2, buffers, 0, nullptr, nullptr);
        const auto finish_error = clFinish(queue_);
        clReleaseMemObject(output);
        check(err != CL_SUCCESS || release_error != CL_SUCCESS || finish_error != CL_SUCCESS,
            "RGB packing failed");
    }

    void compose_nv12_depth(cl_mem left_xrgb, cl_mem depth, cl_mem output, int width, int height,
        int y_stride_bytes, int uv_stride_bytes, float depth_min_m, float depth_max_m,
        const float* disparity = nullptr)
    {
        check(y_stride_bytes < width * 2 || uv_stride_bytes < width * 2,
            "NV12 preview stride is too small");
        // ORT exposes a host disparity view. Reuse one GPU allocation; upload
        // the scalar map, not CPU-colored RGB/NV12, and finish before returning.
        const int disparity_mode = disparity != nullptr;
        if (disparity)
        {
            const size_t bytes = static_cast<size_t>(width) * height * sizeof(float);
            if (!disparity_preview_ || disparity_preview_bytes_ != bytes)
            {
                cl_int status = CL_SUCCESS;
                cl_mem replacement =
                    clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes, nullptr, &status);
                check(status != CL_SUCCESS || !replacement, "allocate disparity preview failed");
                if (disparity_preview_)
                    clReleaseMemObject(disparity_preview_);
                disparity_preview_ = replacement;
                disparity_preview_bytes_ = bytes;
            }
            check(clEnqueueWriteBuffer(queue_, disparity_preview_, CL_FALSE, 0, bytes, disparity, 0,
                        nullptr, nullptr) != CL_SUCCESS,
                "upload disparity preview failed");
        }
        cl_mem values = disparity ? disparity_preview_ : depth;
        // Only imported external allocations participate in acquire/release.
        cl_mem mems[3] = {left_xrgb, output, depth};
        const cl_uint count = disparity ? 2 : 3;
        cl_int err = acquire_(queue_, count, mems, 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
            (void)clFinish(queue_);
        check(err != CL_SUCCESS, "acquire preview memory failed: " + std::to_string(err));
        int arg = 0;
        err = clSetKernelArg(depth_preview_kernel_, arg++, sizeof(cl_mem), &left_xrgb);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(cl_mem), &values);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(cl_mem), &output);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(int), &width);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(int), &height);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(int), &y_stride_bytes);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(int), &uv_stride_bytes);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(float), &depth_min_m);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(float), &depth_max_m);
        err |= clSetKernelArg(depth_preview_kernel_, arg++, sizeof(int), &disparity_mode);
        const size_t global[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(
                queue_, depth_preview_kernel_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        const cl_int released = release_(queue_, count, mems, 0, nullptr, nullptr);
        const cl_int finished = clFinish(queue_);
        check(err != CL_SUCCESS || released != CL_SUCCESS || finished != CL_SUCCESS,
            "compose NV12 preview failed");
    }

private:
    const Nv12Images& cached_nv12_images(
        cl_mem input, int width, int height, int y_stride, int uv_stride, int uv_offset)
    {
        auto& cached = image_cache_[input];
        if (cached.y && cached.width == width && cached.height == height &&
            cached.y_stride == y_stride && cached.uv_stride == uv_stride &&
            cached.uv_offset == uv_offset)
            return cached;
        release_nv12_images(cached);
        cached = create_nv12_images(input, width, height, y_stride, uv_stride, uv_offset);
        cached.width = width;
        cached.height = height;
        cached.y_stride = y_stride;
        cached.uv_stride = uv_stride;
        cached.uv_offset = uv_offset;
        return cached;
    }

    cl_mem create_dummy_image(cl_channel_order order)
    {
        cl_image_format format{};
        format.image_channel_order = order;
        format.image_channel_data_type = CL_UNORM_INT8;
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = 1;
        desc.image_height = 1;
        const uint8_t pixels[2] = {0, 128};
        cl_int err = CL_SUCCESS;
        cl_mem image = clCreateImage(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
            &desc, const_cast<uint8_t*>(pixels), &err);
        check(!image || err != CL_SUCCESS, "clCreateImage dummy failed: " + std::to_string(err));
        return image;
    }

    Nv12Images create_nv12_images(
        cl_mem input, int width, int height, int y_stride, int uv_stride, int uv_offset)
    {
        Nv12Images images;
        cl_int err = CL_SUCCESS;
        cl_image_format y_format{};
        y_format.image_channel_order = CL_R;
        y_format.image_channel_data_type = CL_UNORM_INT8;
        cl_image_desc y_desc{};
        y_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        y_desc.image_width = static_cast<size_t>(width);
        y_desc.image_height = static_cast<size_t>(height);
        y_desc.image_row_pitch = static_cast<size_t>(y_stride);
        y_desc.buffer = input;
        images.y = clCreateImage(context_, CL_MEM_READ_ONLY, &y_format, &y_desc, nullptr, &err);
        if (!images.y || err != CL_SUCCESS)
        {
            std::printf("[OpenCL][IMAGE_FALLBACK] Y image-from-buffer failed err=%d\n", err);
            image_path_supported_ = false;
            return images;
        }

        const cl_buffer_region uv_region{static_cast<size_t>(uv_offset),
            static_cast<size_t>(uv_stride) * static_cast<size_t>(height / 2)};
        images.uv_sub = clCreateSubBuffer(input, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
            const_cast<cl_buffer_region*>(&uv_region), &err);
        if (!images.uv_sub || err != CL_SUCCESS)
        {
            std::printf(
                "[OpenCL][IMAGE_FALLBACK] UV sub-buffer failed err=%d offset=%d\n", err, uv_offset);
            clReleaseMemObject(images.y);
            images.y = nullptr;
            image_path_supported_ = false;
            return images;
        }

        cl_image_format uv_format{};
        uv_format.image_channel_order = CL_RG;
        uv_format.image_channel_data_type = CL_UNORM_INT8;
        cl_image_desc uv_desc{};
        uv_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        uv_desc.image_width = static_cast<size_t>(width / 2);
        uv_desc.image_height = static_cast<size_t>(height / 2);
        uv_desc.image_row_pitch = static_cast<size_t>(uv_stride);
        uv_desc.buffer = images.uv_sub;
        images.uv = clCreateImage(context_, CL_MEM_READ_ONLY, &uv_format, &uv_desc, nullptr, &err);
        if (!images.uv || err != CL_SUCCESS)
        {
            std::printf("[OpenCL][IMAGE_FALLBACK] UV RG8 image-from-buffer failed err=%d\n", err);
            clReleaseMemObject(images.uv_sub);
            clReleaseMemObject(images.y);
            images.uv_sub = nullptr;
            images.y = nullptr;
            image_path_supported_ = false;
            return images;
        }
        if (!image_enabled_logged_)
        {
            std::printf("[OpenCL][IMAGE_SAMPLER] NV12 R8/RG8 image2d enabled %dx%d strides=%d/%d\n",
                width, height, y_stride, uv_stride);
            image_enabled_logged_ = true;
        }
        return images;
    }

    cl_mem create_map_image(const std::vector<float>& map_x, const std::vector<float>& map_y,
        int width, int height, bool full_precision = false)
    {
        check(map_x.size() != map_y.size() || map_x.size() != static_cast<size_t>(width) * height,
            "invalid RG remap image size");
        std::vector<uint16_t> rg(static_cast<size_t>(width) * height * 2);
        for (size_t i = 0; i < map_x.size(); ++i)
        {
            rg[i * 2 + 0] = float_to_half_bits(map_x[i]);
            rg[i * 2 + 1] = float_to_half_bits(map_y[i]);
        }
        std::vector<float> precise;
        if (full_precision)
        {
            precise.resize(map_x.size() * 2);
            for (size_t i = 0; i < map_x.size(); ++i)
            {
                precise[2 * i] = map_x[i];
                precise[2 * i + 1] = map_y[i];
            }
        }
        cl_image_format format;
        std::memset(&format, 0, sizeof(format));
        format.image_channel_order = CL_RG;
        format.image_channel_data_type = full_precision ? CL_FLOAT : CL_HALF_FLOAT;
        cl_image_desc desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = static_cast<size_t>(width);
        desc.image_height = static_cast<size_t>(height);
        cl_int err = CL_SUCCESS;
        cl_mem mem =
            clCreateImage(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc,
                full_precision ? static_cast<void*>(precise.data()) : static_cast<void*>(rg.data()),
                &err);
        check(!mem || err != CL_SUCCESS,
            "clCreateImage RG fp16 remap failed: " + std::to_string(err));
        return mem;
    }

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_kernel depth_preview_kernel_ = nullptr;
    cl_mem disparity_preview_ = nullptr;
    size_t disparity_preview_bytes_ = 0;
    cl_kernel rgb_kernel_ = nullptr;
    cl_mem left_map_ = nullptr;
    cl_mem right_map_ = nullptr;
    cl_mem dummy_y_ = nullptr;
    cl_mem dummy_uv_ = nullptr;
    cl_mem unused_input_ = nullptr;
    bool image_path_supported_ = true;
    bool image_enabled_logged_ = false;
    int image_debug_count_ = 0;
    bool profile_preprocess_ = std::getenv("LAS2_PROFILE_PREPROCESS") != nullptr;
    std::unordered_map<int, cl_mem> input_cache_;
    std::unordered_map<cl_mem, Nv12Images> image_cache_;
    PFN_clImportMemoryARM import_memory_ = nullptr;
    PFN_clEnqueueAcquireExternalMemObjectsKHR acquire_ = nullptr;
    PFN_clEnqueueReleaseExternalMemObjectsKHR release_ = nullptr;
};

struct PreprocessedStereoFrame
{
    std::vector<uint32_t> left_xrgb;
    std::vector<uint32_t> right_xrgb;
    int image_w = 0;
    int image_h = 0;
    int depth_w = 0;
    int depth_h = 0;
    int pad_left = 0;
    int pad_top = 0;
    int dma_fd = -1;
    double decode_ms = 0.0;
    double opencl_ms = 0.0;
};

struct StereoPipelineSlot
{
    PreprocessedStereoFrame frame;
    int frame_id = -1;
    std::unique_ptr<DmaBuffer> left_tensor_dma;
    std::unique_ptr<DmaBuffer> right_tensor_dma;
    std::unique_ptr<DmaBuffer> left_preview_dma;
    std::unique_ptr<DmaBuffer> right_preview_dma;
    std::unique_ptr<DmaBuffer> depth_dma;
    cl_mem cl_left_tensor = nullptr;
    cl_mem cl_right_tensor = nullptr;
    cl_mem cl_left_preview = nullptr;
    cl_mem cl_right_preview = nullptr;
    cl_mem cl_depth = nullptr;
};

std::vector<uint32_t> read_preview(DmaBuffer& dma, int width, int height)
{
    std::vector<uint32_t> out(static_cast<size_t>(width) * height);
    dma.sync_start(kSyncRead);
    const auto* src = static_cast<const uint32_t*>(dma.data());
    std::copy(src, src + out.size(), out.begin());
    dma.sync_end(kSyncRead);
    return out;
}

template <typename Decoder>
PreprocessedStereoFrame decode_preprocess_usb(Decoder& decoder, StereoPreprocessorOpenCl& cl,
    StereoPipelineSlot& slot, const StereoPipelineOptions& opt, int model_w, int model_h,
    int disp_w, int disp_h, bool nhwc)
{
    VideoFrameInfo mpp_frame;
    std::memset(&mpp_frame, 0, sizeof(mpp_frame));
    int dma_fd = -1;
    double decode_ms = 0.0;
    check(!decoder.get_frame(mpp_frame, dma_fd, decode_ms), "failed to get UVC decoded frame");
    const bool is_yuyv = usb_format_is_yuyv(opt);
    const int frame_w = is_yuyv ?
        static_cast<int>(mpp_frame.stCommFrameInfo.u32Width) :
        static_cast<int>(mpp_frame.stVdecFrameInfo.stCommFrameInfo.u32Width);
    const int frame_h = is_yuyv ?
        static_cast<int>(mpp_frame.stCommFrameInfo.u32Height) :
        static_cast<int>(mpp_frame.stVdecFrameInfo.stCommFrameInfo.u32Height);
    check(frame_w != opt.width || frame_h != opt.height,
        "unexpected decoded frame size: " + std::to_string(frame_w) + "x" +
            std::to_string(frame_h));
    check(dma_fd < 0, "decoded frame has no dma-buf fd");

    const int scaled_w = model_w;
    const int scaled_h = model_h;
    const int pad_left = 0;
    const int pad_top = 0;
    const size_t fallback_size = is_yuyv ?
        static_cast<size_t>(mpp_frame.stVFrame.u32PlaneStride[0] ?
                mpp_frame.stVFrame.u32PlaneStride[0] :
                frame_w * 2) *
            frame_h :
        static_cast<size_t>(mpp_frame.stVFrame.u32PlaneStride[0]) * frame_h * 3 / 2;
    const size_t input_size = mpp_frame.stVFrame.u32TotalSize ?
        static_cast<size_t>(mpp_frame.stVFrame.u32TotalSize) :
        fallback_size;
    cl_mem cl_input = cl.import_input_dmabuf_cached(dma_fd, input_size);
    double opencl_ms = 0.0;
    try
    {
        opencl_ms = cl.run(cl_input, slot.cl_left_tensor, slot.cl_right_tensor,
            slot.cl_left_preview, slot.cl_right_preview, mpp_frame, opt, model_w, model_h, nhwc,
            scaled_w, scaled_h, pad_left, pad_top);
    }
    catch (...)
    {
        decoder.release_frame(mpp_frame);
        throw;
    }
    decoder.release_frame(mpp_frame);

    PreprocessedStereoFrame out;
    out.left_xrgb = read_preview(*slot.left_preview_dma, model_w, model_h);
    out.right_xrgb = read_preview(*slot.right_preview_dma, model_w, model_h);
    out.image_w = model_w;
    out.image_h = model_h;
    out.depth_w = scaled_w;
    out.depth_h = scaled_h;
    out.pad_left = pad_left;
    out.pad_top = pad_top;
    out.dma_fd = dma_fd;
    out.decode_ms = decode_ms;
    out.opencl_ms = opencl_ms;
    return out;
}

void release_slot_cl(StereoPipelineSlot& slot)
{
    if (slot.cl_left_tensor)
        clReleaseMemObject(slot.cl_left_tensor);
    if (slot.cl_right_tensor)
        clReleaseMemObject(slot.cl_right_tensor);
    if (slot.cl_left_preview)
        clReleaseMemObject(slot.cl_left_preview);
    if (slot.cl_right_preview)
        clReleaseMemObject(slot.cl_right_preview);
    if (slot.cl_depth)
        clReleaseMemObject(slot.cl_depth);
    slot.cl_left_tensor = nullptr;
    slot.cl_right_tensor = nullptr;
    slot.cl_left_preview = nullptr;
    slot.cl_right_preview = nullptr;
    slot.cl_depth = nullptr;
}

} // namespace

#endif // STEREO_PREPROCESSING_H
