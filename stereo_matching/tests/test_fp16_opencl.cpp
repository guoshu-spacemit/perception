// Exercise the production NV12/remap/sampler kernel, not a copied conversion.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_half.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cl_int code) {
    if (code != CL_SUCCESS)
        throw std::runtime_error("OpenCL: " + std::to_string(code));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    std::cout << std::unitbuf;
    try {
        std::ifstream file(argv[1]);
        const std::string source((std::istreambuf_iterator<char>(file)), {});
        const auto marker = source.find("const char* kUsbPreprocessKernel");
        const auto begin = source.find("R\"CLC(", marker);
        const auto end = source.find(")CLC\"", begin);
        if (begin == std::string::npos || end == std::string::npos)
            return 2;
        const std::string code = source.substr(begin + 6, end - begin - 6);
        cl_platform_id platform;
        cl_device_id device;
        check(clGetPlatformIDs(1, &platform, nullptr));
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
        cl_int error = 0;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        check(error);
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
        check(error);
        auto image = [&](cl_channel_order order, cl_channel_type type, size_t width, size_t height,
                         void* data) {
            cl_image_format format{order, type};
            cl_image_desc desc{};
            desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width = width;
            desc.image_height = height;
            cl_mem result = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
                                          &desc, data, &error);
            check(error);
            return result;
        };
        std::array<cl_half, 32> map{};
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                map[(y * 4 + x) * 2] = cl_half_from_float(float(x), CL_HALF_RTE);
                map[(y * 4 + x) * 2 + 1] = cl_half_from_float(float(y), CL_HALF_RTE);
            }
        }
        cl_mem map_image = image(CL_RG, CL_HALF_FLOAT, 4, 4, map.data());
        size_t cases = 0;
        const char* options[] = {"", "-DLAS2_TENSOR_FP16=1"};
        for (int mode : {0, 1}) {
            const bool half = mode != 0;
            const char* text = code.c_str();
            cl_program program = clCreateProgramWithSource(context, 1, &text, nullptr, &error);
            check(error);
            error = clBuildProgram(program, 1, &device, options[mode], nullptr, nullptr);
            if (error != CL_SUCCESS) {
                char log[16384]{};
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log,
                                      nullptr);
                std::cerr << log;
                check(error);
            }
            cl_kernel kernel = clCreateKernel(program, "usb_nv12_rectify_to_las2", &error);
            check(error);
            for (auto size : {std::array<int, 2>{1, 1}, {7, 5}, {320, 256}, {384, 288}}) {
                const int width = size[0], height = size[1], pixels = width * height;
                const size_t elements = size_t(pixels) * 3;
                const size_t bytes = elements * (half ? 2 : 4);
                for (int nhwc : {0, 1}) {
                    for (int sampler : {0, 1}) {
                        for (auto color : {std::array<int, 3>{81, 90, 240},
                                           {145, 54, 34},
                                           {16, 128, 128},
                                           {235, 128, 128}}) {
                            std::array<unsigned char, 48> nv12{};
                            std::fill(nv12.begin(), nv12.begin() + 32, color[0]);
                            for (int i = 32; i < 48; i += 2) {
                                nv12[i] = color[1];
                                nv12[i + 1] = color[2];
                            }
                            cl_mem y_image = image(CL_R, CL_UNORM_INT8, 8, 4, nv12.data());
                            cl_mem uv_image = image(CL_RG, CL_UNORM_INT8, 4, 2, nv12.data() + 32);
                            std::vector<cl_mem> buffers;
                            auto buffer = [&](size_t length, void* data) {
                                cl_mem result = clCreateBuffer(
                                    context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length, data,
                                    &error);
                                check(error);
                                buffers.push_back(result);
                                return result;
                            };
                            cl_mem input = buffer(nv12.size(), nv12.data());
                            std::vector<unsigned char> initial(bytes + 64, 0xa5);
                            cl_mem left = buffer(initial.size(), initial.data());
                            cl_mem right = buffer(initial.size(), initial.data());
                            std::vector<cl_uint> preview(size_t(pixels) + 16, 0xa5a5a5a5);
                            cl_mem lp = buffer(preview.size() * 4, preview.data());
                            cl_mem rp = buffer(preview.size() * 4, preview.data());
                            int index = 0;
                            auto arg = [&](const auto& value) {
                                check(clSetKernelArg(kernel, index++, sizeof(value), &value));
                            };
                            arg(input);
                            arg(y_image);
                            arg(uv_image);
                            arg(sampler);
                            arg(left);
                            arg(right);
                            arg(lp);
                            arg(rp);
                            arg(map_image);
                            arg(map_image);
                            for (int value :
                                 {8, 4, 8, 8,     32,     0,     0,      4, 0, 0,    4, 4,
                                  4, 4, 0, width, height, width, height, 0, 0, nhwc, 0, 1})
                                arg(value);
                            const size_t global[] = {size_t(width), size_t(height)};
                            check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr,
                                                         0, nullptr, nullptr));
                            const float c = color[0] - 16.0f, d = color[1] - 128.0f,
                                        e = color[2] - 128.0f;
                            const float rgb[] = {
                                std::clamp(1.16438356f * c + 1.79274107f * e, 0.0f, 255.0f),
                                std::clamp(1.16438356f * c - 0.21324861f * d - 0.53290933f * e,
                                           0.0f, 255.0f),
                                std::clamp(1.16438356f * c + 2.11240179f * d, 0.0f, 255.0f)};
                            const float mean[] = {0.485f, 0.456f, 0.406f},
                                        stddev[] = {0.229f, 0.224f, 0.225f};
                            double dot = 0, a2 = 0, b2 = 0, max_error = 0;
                            for (cl_mem tensor : {left, right}) {
                                std::vector<unsigned char> output(initial.size());
                                check(clEnqueueReadBuffer(queue, tensor, CL_TRUE, 0, output.size(),
                                                          output.data(), 0, nullptr, nullptr));
                                for (size_t i = 0; i < elements; ++i) {
                                    const size_t channel = nhwc ? i % 3 : i / pixels;
                                    float expected =
                                        (rgb[channel] / 255.0f - mean[channel]) / stddev[channel];
                                    float actual;
                                    if (half) {
                                        cl_half bits;
                                        std::memcpy(&bits, output.data() + i * 2, 2);
                                        actual = cl_half_to_float(bits);
                                        expected = cl_half_to_float(
                                            cl_half_from_float(expected, CL_HALF_RTE));
                                    } else {
                                        std::memcpy(&actual, output.data() + i * 4, 4);
                                    }
                                    const double delta = std::abs(actual - expected);
                                    // The hardware UNORM sampler has lower filtering
                                    // precision than the buffer arithmetic reference.
                                    const double tolerance =
                                        sampler ? 0.005 : (half ? 0.002 : 1e-5);
                                    if (!std::isfinite(actual) || delta > tolerance) {
                                        std::cerr << "shape=" << width << 'x' << height
                                                  << " mode=" << mode << " nhwc=" << nhwc
                                                  << " sampler=" << sampler << " index=" << i
                                                  << " actual=" << actual
                                                  << " expected=" << expected << '\n';
                                        throw std::runtime_error("tensor numeric mismatch");
                                    }
                                    max_error = std::max(max_error, delta);
                                    dot += actual * expected;
                                    a2 += actual * actual;
                                    b2 += expected * expected;
                                }
                                if (!std::all_of(output.begin() + bytes, output.end(),
                                                 [](unsigned char v) { return v == 0xa5; }))
                                    throw std::runtime_error("tensor overrun");
                            }
                            const cl_uint packed = 0xff000000U | (cl_uint(rgb[0] + 0.5f) << 16) |
                                                   (cl_uint(rgb[1] + 0.5f) << 8) |
                                                   cl_uint(rgb[2] + 0.5f);
                            for (cl_mem output : {lp, rp}) {
                                check(clEnqueueReadBuffer(queue, output, CL_TRUE, 0,
                                                          preview.size() * 4, preview.data(), 0,
                                                          nullptr, nullptr));
                                for (int i = 0; i < pixels; ++i) {
                                    for (int shift : {0, 8, 16, 24}) {
                                        const int error =
                                            std::abs(int((preview[i] >> shift) & 255) -
                                                     int((packed >> shift) & 255));
                                        if (error != 0)
                                            throw std::runtime_error("preview numeric mismatch");
                                    }
                                }
                                for (size_t i = pixels; i < preview.size(); ++i)
                                    if (preview[i] != 0xa5a5a5a5)
                                        throw std::runtime_error("preview overrun");
                            }
                            const double cosine = dot / std::sqrt(a2 * b2);
                            if (cosine < 0.9999)
                                throw std::runtime_error("cosine mismatch");
                            ++cases;
                            for (auto b : buffers)
                                clReleaseMemObject(b);
                            clReleaseMemObject(y_image);
                            clReleaseMemObject(uv_image);
                            std::cout << "PASS " << width << 'x' << height << " mode=" << mode
                                      << " nhwc=" << nhwc << " sampler=" << sampler
                                      << " max_abs=" << max_error << " cosine=" << cosine << '\n';
                        }
                    }
                }
            }
            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }
        clReleaseMemObject(map_image);
        check(clFinish(queue));
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        std::cout << "PASS cases=" << cases << " guards=unchanged preview_rgb_error=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
