#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

#if defined(__riscv_vector) && defined(__riscv_v_elen_fp) && __riscv_v_elen_fp >= 64 &&            \
    !defined(LAS2_DISABLE_RVV_RECTIFICATION)
#include <riscv_vector.h>
#define LAS2_HAS_RVV_RECTIFICATION 1
#else
#define LAS2_HAS_RVV_RECTIFICATION 0
#endif

namespace las2::detail {

struct RemapValidity {
    size_t invalid = 0;
    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
};

inline RemapValidity inspect_remap(const cv::Mat &map_x, const cv::Mat &map_y, int width,
                                   int height) {
    CV_Assert(map_x.type() == CV_32FC1 && map_y.type() == CV_32FC1 && map_x.size() == map_y.size());
    RemapValidity result;
    const float max_x = static_cast<float>(width - 1);
    const float max_y = static_cast<float>(height - 1);
    for (int row = 0; row < map_x.rows; ++row) {
        const float *px = map_x.ptr<float>(row);
        const float *py = map_y.ptr<float>(row);
        int column = 0;
#if LAS2_HAS_RVV_RECTIFICATION
        while (column < map_x.cols) {
            const size_t vl = __riscv_vsetvl_e32m4(map_x.cols - column);
            const auto x = __riscv_vle32_v_f32m4(px + column, vl);
            const auto y = __riscv_vle32_v_f32m4(py + column, vl);
            const auto finite_x = __riscv_vmflt_vf_f32m4_b8(
                __riscv_vfabs_v_f32m4(x, vl), std::numeric_limits<float>::infinity(), vl);
            const auto finite_y = __riscv_vmflt_vf_f32m4_b8(
                __riscv_vfabs_v_f32m4(y, vl), std::numeric_limits<float>::infinity(), vl);
            const auto finite = __riscv_vmand_mm_b8(finite_x, finite_y, vl);
            auto valid = __riscv_vmand_mm_b8(finite, __riscv_vmfge_vf_f32m4_b8(x, 0, vl), vl);
            valid = __riscv_vmand_mm_b8(valid, __riscv_vmfge_vf_f32m4_b8(y, 0, vl), vl);
            valid = __riscv_vmand_mm_b8(valid, __riscv_vmfle_vf_f32m4_b8(x, max_x, vl), vl);
            valid = __riscv_vmand_mm_b8(valid, __riscv_vmfle_vf_f32m4_b8(y, max_y, vl), vl);
            result.invalid += vl - __riscv_vcpop_m_b8(valid, vl);
            result.min_x = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m4_f32m1_m(
                finite, x, __riscv_vfmv_v_f_f32m1(result.min_x, 1), vl));
            result.max_x = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1_m(
                finite, x, __riscv_vfmv_v_f_f32m1(result.max_x, 1), vl));
            result.min_y = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m4_f32m1_m(
                finite, y, __riscv_vfmv_v_f_f32m1(result.min_y, 1), vl));
            result.max_y = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1_m(
                finite, y, __riscv_vfmv_v_f_f32m1(result.max_y, 1), vl));
            column += static_cast<int>(vl);
        }
#else
        for (; column < map_x.cols; ++column) {
            const float x = px[column], y = py[column];
            const bool finite = std::isfinite(x) && std::isfinite(y);
            if (finite) {
                result.min_x = std::min(result.min_x, x);
                result.max_x = std::max(result.max_x, x);
                result.min_y = std::min(result.min_y, y);
                result.max_y = std::max(result.max_y, y);
            }
            if (!finite || x < 0 || x > max_x || y < 0 || y > max_y)
                ++result.invalid;
        }
#endif
    }
    return result;
}

#if LAS2_HAS_RVV_RECTIFICATION
// atan on nonnegative radii. Range reduction puts the Taylor argument within
// [-tan(pi/8), tan(pi/8)]. Degree 41 gives a truncation bound < 9e-19 radians;
// this is NOT a bound on total floating-point rounding error. Tests compare
// projection coordinates and validity, including all range-reduction boundaries.
inline vfloat64m2_t atan_radius(vfloat64m2_t radius, size_t vl) {
    const auto reciprocal = __riscv_vmfgt_vf_f64m2_b32(radius, 1.0, vl);
    const auto safe_radius = __riscv_vfmax_vf_f64m2(radius, 1.0, vl);
    auto t = __riscv_vmerge_vvm_f64m2(radius, __riscv_vfrdiv_vf_f64m2(safe_radius, 1.0, vl),
                                      reciprocal, vl);
    const auto shifted = __riscv_vmfgt_vf_f64m2_b32(t, 0.4142135623730950488, vl);
    t = __riscv_vmerge_vvm_f64m2(t,
                                 __riscv_vfdiv_vv_f64m2(__riscv_vfsub_vf_f64m2(t, 1.0, vl),
                                                        __riscv_vfadd_vf_f64m2(t, 1.0, vl), vl),
                                 shifted, vl);
    const auto square = __riscv_vfmul_vv_f64m2(t, t, vl);
    auto polynomial = __riscv_vfmv_v_f_f64m2(1.0 / 41.0, vl);
    for (int n = 19; n >= 0; --n) {
        const double coefficient = (n % 2 ? -1.0 : 1.0) / (2 * n + 1);
        polynomial =
            __riscv_vfadd_vf_f64m2(__riscv_vfmul_vv_f64m2(polynomial, square, vl), coefficient, vl);
    }
    auto angle = __riscv_vfmul_vv_f64m2(t, polynomial, vl);
    angle = __riscv_vmerge_vvm_f64m2(angle, __riscv_vfadd_vf_f64m2(angle, CV_PI / 4.0, vl), shifted,
                                     vl);
    return __riscv_vmerge_vvm_f64m2(angle, __riscv_vfrsub_vf_f64m2(angle, CV_PI / 2.0, vl),
                                    reciprocal, vl);
}
#endif

// Same double-precision equidistant projection as OpenCV fisheye, emitting
// float32 backward maps. Small matrix decomposition remains in OpenCV.
inline void fisheye_maps(const cv::Mat &k, const cv::Mat &d, const cv::Mat &r, const cv::Mat &p,
                         cv::Size size, cv::Mat &map_x, cv::Mat &map_y) {
#if LAS2_HAS_RVV_RECTIFICATION
    CV_Assert(k.type() == CV_64FC1 && k.rows == 3 && k.cols == 3 && d.type() == CV_64FC1 &&
              d.total() == 4 && d.isContinuous() && r.type() == CV_64FC1 && r.rows == 3 &&
              r.cols == 3 && p.type() == CV_64FC1 && p.rows == 3 && p.cols >= 3);
    const cv::Mat transform = (p.colRange(0, 3) * r).inv(cv::DECOMP_SVD);
    const cv::Matx33d inverse(transform);
    const double *coeff = d.ptr<double>();
    const double fx = k.at<double>(0, 0), fy = k.at<double>(1, 1);
    const double cx = k.at<double>(0, 2), cy = k.at<double>(1, 2);
    map_x.create(size, CV_32FC1);
    map_y.create(size, CV_32FC1);
    for (int row = 0; row < size.height; ++row) {
        float *px = map_x.ptr<float>(row);
        float *py = map_y.ptr<float>(row);
        const double origin_x = row * inverse(0, 1) + inverse(0, 2);
        const double origin_y = row * inverse(1, 1) + inverse(1, 2);
        const double origin_w = row * inverse(2, 1) + inverse(2, 2);
        for (int column = 0; column < size.width;) {
            const size_t vl = __riscv_vsetvl_e64m2(size.width - column);
            const auto index = __riscv_vfcvt_f_xu_v_f64m2(
                __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), column, vl), vl);
            const auto xx = __riscv_vfadd_vf_f64m2(__riscv_vfmul_vf_f64m2(index, inverse(0, 0), vl),
                                                   origin_x, vl);
            const auto yy = __riscv_vfadd_vf_f64m2(__riscv_vfmul_vf_f64m2(index, inverse(1, 0), vl),
                                                   origin_y, vl);
            const auto ww = __riscv_vfadd_vf_f64m2(__riscv_vfmul_vf_f64m2(index, inverse(2, 0), vl),
                                                   origin_w, vl);
            const double infinity = std::numeric_limits<double>::infinity();
            const auto invalid_u =
                __riscv_vfmerge_vfm_f64m2(__riscv_vfmv_v_f_f64m2(infinity, vl), -infinity,
                                          __riscv_vmfgt_vf_f64m2_b32(xx, 0, vl), vl);
            const auto invalid_v =
                __riscv_vfmerge_vfm_f64m2(__riscv_vfmv_v_f_f64m2(infinity, vl), -infinity,
                                          __riscv_vmfgt_vf_f64m2_b32(yy, 0, vl), vl);
            const auto behind = __riscv_vmfle_vf_f64m2_b32(ww, 0, vl);
            // Match OpenCV's early exit for rays behind the camera. Avoid
            // paying for division/atan when an entire vector is invalid.
            if (__riscv_vcpop_m_b32(behind, vl) == vl) {
                __riscv_vse32_v_f32m1(px + column, __riscv_vfncvt_f_f_w_f32m1(invalid_u, vl), vl);
                __riscv_vse32_v_f32m1(py + column, __riscv_vfncvt_f_f_w_f32m1(invalid_v, vl), vl);
                column += static_cast<int>(vl);
                continue;
            }
            const auto x = __riscv_vfdiv_vv_f64m2(xx, ww, vl);
            const auto y = __riscv_vfdiv_vv_f64m2(yy, ww, vl);
            const auto radius =
                __riscv_vfsqrt_v_f64m2(__riscv_vfadd_vv_f64m2(__riscv_vfmul_vv_f64m2(x, x, vl),
                                                              __riscv_vfmul_vv_f64m2(y, y, vl), vl),
                                       vl);
            const auto theta = atan_radius(radius, vl);
            const auto t2 = __riscv_vfmul_vv_f64m2(theta, theta, vl);
            const auto t4 = __riscv_vfmul_vv_f64m2(t2, t2, vl);
            const auto t6 = __riscv_vfmul_vv_f64m2(t4, t2, vl);
            const auto t8 = __riscv_vfmul_vv_f64m2(t4, t4, vl);
            auto factor = __riscv_vfadd_vf_f64m2(__riscv_vfmul_vf_f64m2(t2, coeff[0], vl), 1, vl);
            factor = __riscv_vfadd_vv_f64m2(factor, __riscv_vfmul_vf_f64m2(t4, coeff[1], vl), vl);
            factor = __riscv_vfadd_vv_f64m2(factor, __riscv_vfmul_vf_f64m2(t6, coeff[2], vl), vl);
            factor = __riscv_vfadd_vv_f64m2(factor, __riscv_vfmul_vf_f64m2(t8, coeff[3], vl), vl);
            const auto zero = __riscv_vmfeq_vf_f64m2_b32(radius, 0, vl);
            const auto divisor = __riscv_vfmerge_vfm_f64m2(radius, 1.0, zero, vl);
            auto scale =
                __riscv_vfdiv_vv_f64m2(__riscv_vfmul_vv_f64m2(theta, factor, vl), divisor, vl);
            scale = __riscv_vfmerge_vfm_f64m2(scale, 1.0, zero, vl);
            auto u = __riscv_vfadd_vf_f64m2(
                __riscv_vfmul_vv_f64m2(__riscv_vfmul_vf_f64m2(x, fx, vl), scale, vl), cx, vl);
            auto v = __riscv_vfadd_vf_f64m2(
                __riscv_vfmul_vv_f64m2(__riscv_vfmul_vf_f64m2(y, fy, vl), scale, vl), cy, vl);
            u = __riscv_vmerge_vvm_f64m2(u, invalid_u, behind, vl);
            v = __riscv_vmerge_vvm_f64m2(v, invalid_v, behind, vl);
            __riscv_vse32_v_f32m1(px + column, __riscv_vfncvt_f_f_w_f32m1(u, vl), vl);
            __riscv_vse32_v_f32m1(py + column, __riscv_vfncvt_f_f_w_f32m1(v, vl), vl);
            column += static_cast<int>(vl);
        }
    }
#else
    cv::fisheye::initUndistortRectifyMap(k, d, r, p, size, CV_32FC1, map_x, map_y);
#endif
}

} // namespace las2::detail
