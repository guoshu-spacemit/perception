// Compile the production color/NV12 kernel, not a separately maintained copy.
#define CL_TARGET_OPENCL_VERSION 120
#include "preview_reference.hpp"
#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cl_int error) {
    if (error != CL_SUCCESS)
        throw std::runtime_error("OpenCL error " + std::to_string(error));
}
struct Cl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::vector<cl_mem> buffers;
    ~Cl() {
        if (queue)
            clFinish(queue);
        for (auto m : buffers)
            clReleaseMemObject(m);
        if (kernel)
            clReleaseKernel(kernel);
        if (program)
            clReleaseProgram(program);
        if (queue)
            clReleaseCommandQueue(queue);
        if (context)
            clReleaseContext(context);
    }
    cl_mem buffer(size_t bytes, void* data) {
        cl_int error = CL_SUCCESS;
        auto m =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data, &error);
        check(error);
        buffers.push_back(m);
        return m;
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    std::cout << std::unitbuf;
    try {
        std::ifstream file(argv[1]);
        const std::string source((std::istreambuf_iterator<char>(file)), {});
        const auto marker = source.find("const char* kDepthPreviewKernel");
        const auto begin = source.find("R\"CLC(", marker);
        const auto end = source.find(")CLC\"", begin);
        if (begin == std::string::npos || end == std::string::npos)
            throw std::runtime_error("production preview source not found");
        const std::string code = source.substr(begin + 6, end - begin - 6);
        Cl cl;
        cl_platform_id platform;
        cl_device_id device;
        check(clGetPlatformIDs(1, &platform, nullptr));
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
        cl_int error = CL_SUCCESS;
        cl.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        check(error);
        cl.queue = clCreateCommandQueue(cl.context, device, CL_QUEUE_PROFILING_ENABLE, &error);
        check(error);
        const char* text = code.c_str();
        cl.program = clCreateProgramWithSource(cl.context, 1, &text, nullptr, &error);
        check(error);
        error = clBuildProgram(cl.program, 1, &device, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            char log[8192]{};
            clGetProgramBuildInfo(cl.program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log,
                                  nullptr);
            std::cerr << log;
            check(error);
        }
        cl.kernel = clCreateKernel(cl.program, "compose_nv12_depth", &error);
        check(error);
        int cases = 0;
        for (auto shape : {std::array<int, 2>{2, 2}, {8, 6}, {320, 256}, {384, 288}}) {
            const int w = shape[0], h = shape[1];
            for (int pad : {0, 32}) {
                const int ys = 2 * w + pad, uvs = 2 * w + 2 * pad;
                std::vector<std::uint32_t> xrgb(w * h);
                std::vector<std::uint8_t> rgb(w * h * 3);
                std::vector<float> values(w * h);
                const size_t bytes = size_t(ys) * h + size_t(uvs) * (h / 2);
                std::vector<std::uint8_t> expected(bytes + 64, 0xa5), actual(expected);
                for (int i = 0; i < w * h; ++i) {
                    const auto r = (i * 71) % 256, g = (i * 31) % 256, b = (i * 13) % 256;
                    xrgb[i] = 0xff000000U | (r << 16) | (g << 8) | b;
                    rgb[i * 3] = r;
                    rgb[i * 3 + 1] = g;
                    rgb[i * 3 + 2] = b;
                }
                auto left = cl.buffer(xrgb.size() * 4, xrgb.data());
                auto scalar = cl.buffer(values.size() * 4, values.data());
                auto out = cl.buffer(actual.size(), actual.data());
                for (int mode : {0, 1})
                    for (float minimum : {0.0F, 8.0F}) {
                        const float maximum = mode ? 128.0F : 15.0F;
                        const float special[] = {0,
                                                 -1,
                                                 std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::infinity(),
                                                 -std::numeric_limits<float>::infinity(),
                                                 minimum,
                                                 minimum * 0.5F,
                                                 maximum,
                                                 maximum + 1,
                                                 1.0e-7F};
                        for (int i = 0; i < w * h; ++i)
                            values[i] = (i % 19 < 10) ? special[i % 19] : (i % 257) * 0.51F;
                        std::fill(expected.begin(), expected.end(), 0xa5);
                        preview_test::compose(rgb.data(), w * 3, values.data(), w * 4, w, h, mode,
                                              minimum, maximum, expected.data(), ys, uvs);
                        check(clEnqueueWriteBuffer(cl.queue, scalar, CL_TRUE, 0, values.size() * 4,
                                                   values.data(), 0, nullptr, nullptr));
                        int index = 0;
                        const auto arg = [&](const auto& v) {
                            check(clSetKernelArg(cl.kernel, index++, sizeof(v), &v));
                        };
                        arg(left);
                        arg(scalar);
                        arg(out);
                        arg(w);
                        arg(h);
                        arg(ys);
                        arg(uvs);
                        arg(minimum);
                        arg(maximum);
                        arg(mode);
                        const size_t global[] = {size_t(w + 3), size_t(h + 3)};
                        check(clEnqueueNDRangeKernel(cl.queue, cl.kernel, 2, nullptr, global,
                                                     nullptr, 0, nullptr, nullptr));
                        check(clEnqueueReadBuffer(cl.queue, out, CL_TRUE, 0, actual.size(),
                                                  actual.data(), 0, nullptr, nullptr));
                        int worst = 0;
                        for (size_t i = 0; i < actual.size(); ++i) {
                            const bool padding =
                                i >= bytes ||
                                (i < size_t(ys) * h ? i % ys >= size_t(w * 2)
                                                    : (i - size_t(ys) * h) % uvs >= size_t(w * 2));
                            const int delta = std::abs(int(actual[i]) - int(expected[i]));
                            worst = std::max(worst, delta);
                            // GPU uses float RGB and BT.601; old CPU rounds RGB8 first.
                            if (delta > (padding ? 0 : 2))
                                throw std::runtime_error(
                                    "preview mismatch byte=" + std::to_string(i) +
                                    " delta=" + std::to_string(delta));
                        }
                        ++cases;
                        std::cout << "PASS " << w << 'x' << h << " padding=" << pad
                                  << " mode=" << mode << " min=" << minimum
                                  << " max_byte_error=" << worst << '\n';
                    }
                for (auto m : cl.buffers)
                    clReleaseMemObject(m);
                cl.buffers.clear();
            }
        }
        std::cout << "PASS cases=" << cases << " guards/padding=unchanged\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
