#include "rectification_map_ops.h"
#include "shared_initialization_cache.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>

using Clock = std::chrono::steady_clock;

static void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static double compare(const cv::Mat& a, const cv::Mat& b)
{
    require(a.type() == CV_32FC1 && a.size() == b.size(), "map shape mismatch");
    double worst = 0;
    double relative = 0, dot = 0, norm_a = 0, norm_b = 0;
    for (int y = 0; y < a.rows; ++y)
    {
        for (int x = 0; x < a.cols; ++x)
        {
            const float av = a.at<float>(y, x), bv = b.at<float>(y, x);
            if (!std::isfinite(av) || !std::isfinite(bv))
            {
                require((std::isnan(av) && std::isnan(bv)) || av == bv, "non-finite mismatch");
            }
            else
            {
                worst = std::max(worst, std::abs(static_cast<double>(av) - bv));
                relative = std::max(
                    relative, std::abs(double(av) - bv) / std::max(1e-12, std::abs(double(bv))));
                dot += double(av) * bv;
                norm_a += double(av) * av;
                norm_b += double(bv) * bv;
                require(std::abs(av - bv) <= 0.00025F, "map coordinate error > 0.00025px");
            }
        }
    }
    const double cosine = norm_a > 0 && norm_b > 0 ? dot / std::sqrt(norm_a * norm_b) : 1;
    require(cosine >= .9999, "map cosine similarity");
    std::cout << "comparison max_relative_eps1e-12=" << relative << " cosine=" << cosine << '\n';
    return worst;
}

static las2::detail::RemapValidity scalar_validity(
    const cv::Mat& x, const cv::Mat& y, int width, int height)
{
    las2::detail::RemapValidity result;
    for (int row = 0; row < x.rows; ++row)
        for (int col = 0; col < x.cols; ++col)
        {
            const auto a = x.at<float>(row, col), b = y.at<float>(row, col);
            const bool finite = std::isfinite(a) && std::isfinite(b);
            if (finite)
            {
                result.min_x = std::min(result.min_x, a);
                result.max_x = std::max(result.max_x, a);
                result.min_y = std::min(result.min_y, b);
                result.max_y = std::max(result.max_y, b);
            }
            if (!finite || a < 0 || a > width - 1 || b < 0 || b > height - 1)
                ++result.invalid;
        }
    return result;
}

static void compare_validity(const cv::Mat& x, const cv::Mat& y)
{
    const auto a = scalar_validity(x, y, x.cols, x.rows);
    const auto b = las2::detail::inspect_remap(x, y, x.cols, x.rows);
    require(a.invalid == b.invalid && a.min_x == b.min_x && a.max_x == b.max_x &&
            a.min_y == b.min_y && a.max_y == b.max_y,
        "validity mismatch");
}

static void test_validity()
{
    std::mt19937 random(4138);
    std::uniform_real_distribution<float> value(-100, 1200);
    for (int width : {1, 3, 7, 31, 32, 33, 65, 384, 641, 1920})
    {
        cv::Mat x_store(9, width + 13, CV_32FC1), y_store(9, width + 13, CV_32FC1);
        cv::Mat x = x_store(cv::Rect(3, 0, width, 9));
        cv::Mat y = y_store(cv::Rect(3, 0, width, 9));
        for (int row = 0; row < x.rows; ++row)
            for (int col = 0; col < x.cols; ++col)
            {
                x.at<float>(row, col) = value(random);
                y.at<float>(row, col) = value(random);
            }
        x.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
        x.at<float>(1, 0) = std::numeric_limits<float>::infinity();
        x.at<float>(2, 0) = -std::numeric_limits<float>::infinity();
        x.at<float>(3, 0) = 0;
        y.at<float>(3, 0) = 0;
        x.at<float>(4, 0) = width - 1;
        y.at<float>(4, 0) = 8;
        compare_validity(x, y);
        x.setTo(std::numeric_limits<float>::quiet_NaN());
        compare_validity(x, y);
    }
}

static void test_cache()
{
    las2::detail::SharedInitializationCache<std::string, int> cache;
    std::atomic<int> calls{0};
    std::promise<void> begin;
    auto gate = begin.get_future().share();
    std::vector<std::future<std::shared_ptr<const int>>> workers;
    for (int n = 0; n < 8; ++n)
        workers.push_back(std::async(std::launch::async,
            [&]
            {
                gate.wait();
                return cache.get("same",
                    [&]
                    {
                        ++calls;
                        return 42;
                    });
            }));
    begin.set_value();
    std::vector<std::shared_ptr<const int>> owners;
    for (auto& worker : workers)
        owners.push_back(worker.get());
    require(calls == 1, "duplicate cache construction");
    for (auto& owner : owners)
        require(owner.get() == owners[0].get(), "value was copied");
    auto distinct = cache.get("different contents/geometry", [] { return 7; });
    require(*distinct == 7 && distinct.get() != owners[0].get(), "cache key ignored");
    std::weak_ptr<const int> lifetime = owners[0];
    owners.clear();
    require(lifetime.expired(), "cache retains unused large maps");
    auto rebuilt = cache.get("same",
        [&]
        {
            ++calls;
            return 43;
        });
    require(calls == 2 && *rebuilt == 43, "expired entry not rebuilt");
    try
    {
        cache.get("failure", []() -> int { throw std::runtime_error("expected"); });
        require(false, "missing factory error");
    }
    catch (const std::runtime_error& e)
    {
        require(std::string(e.what()) == "expected", "unexpected error");
    }
    require(*cache.get("failure", [] { return 9; }) == 9, "failed entry poisons retry");
}

static void test_atan()
{
#if LAS2_HAS_RVV_RECTIFICATION
    std::vector<double> radii{0, 1e-300, 0.4142135623730950488, 1, 2.4142135623730950488,
        std::numeric_limits<double>::infinity()};
    for (int i = -12000; i <= 12000; ++i)
        radii.push_back(std::exp(i * 0.04));
    for (double boundary : {0.4142135623730950488, 1.0, 2.4142135623730950488})
    {
        radii.push_back(std::nextafter(boundary, 0));
        radii.push_back(std::nextafter(boundary, std::numeric_limits<double>::infinity()));
    }
    double worst = 0;
    std::vector<double> result(radii.size());
    for (size_t i = 0; i < radii.size();)
    {
        const size_t vl = __riscv_vsetvl_e64m2(radii.size() - i);
        __riscv_vse64_v_f64m2(result.data() + i,
            las2::detail::atan_radius(__riscv_vle64_v_f64m2(radii.data() + i, vl), vl), vl);
        i += vl;
    }
    for (size_t i = 0; i < radii.size(); ++i)
        worst = std::max(worst, std::abs(result[i] - std::atan(radii[i])));
    require(worst < 1e-14, "atan error");
    std::cout << "atan max_error_rad=" << worst << '\n';
#endif
}

int main(int argc, char** argv)
{
    try
    {
        test_cache();
        test_validity();
        test_atan();
        const bool write = argc == 3 && std::string(argv[1]) == "--write-reference";
        const bool read = argc == 3 && std::string(argv[1]) == "--verify-reference";
        require(argc == 1 || write || read, "usage: [--write-reference|--verify-reference FILE]");
        std::ofstream golden_out;
        std::ifstream golden_in;
        if (write)
            golden_out.open(argv[2], std::ios::binary);
        if (read)
            golden_in.open(argv[2], std::ios::binary);
        const std::vector<cv::Size> shapes{
            {1, 1}, {7, 5}, {33, 17}, {384, 288}, {641, 401}, {1920, 1200}};
        for (auto size : shapes)
        {
            for (int variant = 0; variant < 4; ++variant)
            {
                cv::Mat k = (cv::Mat_<double>(3, 3) << size.width * .61, 0, size.width * .48, 0,
                    size.width * .62, size.height * .53, 0, 0, 1);
                cv::Mat d = (cv::Mat_<double>(1, 4) << -.08, .015, -.003, .0002);
                if (variant == 1)
                    d.setTo(0);
                cv::Mat r;
                cv::Rodrigues(
                    cv::Vec3d(variant == 3 ? 3.0 : .025 * variant, -.04 * variant, .01), r);
                cv::Mat p = cv::Mat::zeros(3, 4, CV_64F);
                k.copyTo(p.colRange(0, 3));
                p.at<double>(0, 0) *= .7;
                p.at<double>(1, 1) *= .7;
                cv::Mat rx, ry;
                cv::fisheye::initUndistortRectifyMap(k, d, r, p, size, CV_32FC1, rx, ry);
                if (write)
                {
                    golden_out.write(reinterpret_cast<char*>(rx.data), rx.total() * sizeof(float));
                    golden_out.write(reinterpret_cast<char*>(ry.data), ry.total() * sizeof(float));
                    require(golden_out.good(), "reference write failed");
                }
                if (read)
                {
                    golden_in.read(reinterpret_cast<char*>(rx.data), rx.total() * sizeof(float));
                    golden_in.read(reinterpret_cast<char*>(ry.data), ry.total() * sizeof(float));
                    require(golden_in.good(), "reference read failed");
                }
                cv::Mat backing_x(size.height, size.width + 9, CV_32FC1, cv::Scalar(-123));
                cv::Mat backing_y(size.height, size.width + 9, CV_32FC1, cv::Scalar(-123));
                cv::Mat x = backing_x(cv::Rect(4, 0, size.width, size.height));
                cv::Mat y = backing_y(cv::Rect(4, 0, size.width, size.height));
                las2::detail::fisheye_maps(k, d, r, p, size, x, y);
                const double worst = std::max(compare(x, rx), compare(y, ry));
                compare_validity(x, y);
                for (int row = 0; row < size.height; ++row)
                {
                    require(backing_x.at<float>(row, 3) == -123 &&
                            backing_x.at<float>(row, size.width + 4) == -123 &&
                            backing_y.at<float>(row, 3) == -123 &&
                            backing_y.at<float>(row, size.width + 4) == -123,
                        "output overrun");
                }
                double reference_ms = 0, selected_ms = 0;
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    auto start = Clock::now();
                    cv::fisheye::initUndistortRectifyMap(k, d, r, p, size, CV_32FC1, rx, ry);
                    reference_ms +=
                        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    start = Clock::now();
                    las2::detail::fisheye_maps(k, d, r, p, size, x, y);
                    selected_ms +=
                        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                }
                std::cout << "map=" << size.width << 'x' << size.height << " variant=" << variant
                            << " max_error_px=" << worst << " opencv_ms=" << reference_ms / 3
                            << " selected_ms=" << selected_ms / 3 << '\n';
            }
        }
        std::cout << "PASS backend=" << (LAS2_HAS_RVV_RECTIFICATION ? "rvv-fp64" : "opencv")
                    << '\n';
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
