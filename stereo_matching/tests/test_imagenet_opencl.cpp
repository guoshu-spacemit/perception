#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

void check(cl_int code)
{
    if (code != CL_SUCCESS)
        throw std::runtime_error("OpenCL error " + std::to_string(code));
}

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    std::ifstream file(argv[1]);
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    const auto begin = source.find("    // Preview stays in RGB byte range;");
    const auto end = source.find("    if (nhwc)", begin);
    if (begin == std::string::npos || end == std::string::npos)
        return 2;
    // Compile the exact normalization + preview block from the production kernel.
    const std::string code = "__kernel void probe(__global const float* in, __global float* out, "
                                "__global uint* preview) {"
                                "int i=get_global_id(0)*6; float lr=in[i],lg=in[i+1],lb=in[i+2];"
                                "float rr=in[i+3],rg=in[i+4],rb=in[i+5];\n" +
        source.substr(begin, end - begin) +
        "out[i]=lr;out[i+1]=lg;out[i+2]=lb;out[i+3]=rr;out[i+4]=rg;out[i+5]=rb;"
        "preview[get_global_id(0)*2]=lp;preview[get_global_id(0)*2+1]=rp;}";
    cl_platform_id platform;
    cl_device_id device;
    check(clGetPlatformIDs(1, &platform, nullptr));
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check(err);
    const char* text = code.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &text, nullptr, &err);
    check(err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err)
    {
        char log[8192]{};
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << log;
        check(err);
    }
    cl_kernel kernel = clCreateKernel(program, "probe", &err);
    check(err);
    constexpr size_t count = 256;
    std::vector<float> input(count * 6), output(count * 6);
    std::vector<cl_uint> preview(count * 2);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = float((i * 73) % 256);
    auto buffer = [&](cl_mem_flags flags, size_t bytes, void* data)
    {
        cl_mem result = clCreateBuffer(context, flags, bytes, data, &err);
        check(err);
        return result;
    };
    cl_mem in = buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * 4, input.data());
    cl_mem out = buffer(CL_MEM_WRITE_ONLY, output.size() * 4, nullptr);
    cl_mem rgb = buffer(CL_MEM_WRITE_ONLY, preview.size() * 4, nullptr);
    check(clSetKernelArg(kernel, 0, sizeof(in), &in));
    check(clSetKernelArg(kernel, 1, sizeof(out), &out));
    check(clSetKernelArg(kernel, 2, sizeof(rgb), &rgb));
    const double mean[3] = {0.485, 0.456, 0.406};
    const double stddev[3] = {0.229, 0.224, 0.225};
    for (int enabled : {1})
    {
        check(clEnqueueNDRangeKernel(
            queue, kernel, 1, nullptr, &count, nullptr, 0, nullptr, nullptr));
        check(clEnqueueReadBuffer(
            queue, out, CL_TRUE, 0, output.size() * 4, output.data(), 0, nullptr, nullptr));
        check(clEnqueueReadBuffer(
            queue, rgb, CL_TRUE, 0, preview.size() * 4, preview.data(), 0, nullptr, nullptr));
        double max_error = 0;
        for (size_t i = 0; i < input.size(); ++i)
        {
            const double expected =
                enabled ? (input[i] / 255.0 - mean[i % 3]) / stddev[i % 3] : input[i];
            const double error = std::abs(output[i] - expected);
            if (!std::isfinite(output[i]) || error > 1e-5)
                return 1;
            max_error = std::max(max_error, error);
        }
        for (size_t i = 0; i < preview.size(); ++i)
        {
            const cl_uint expected = 0xff000000u | (cl_uint(input[i * 3]) << 16) |
                (cl_uint(input[i * 3 + 1]) << 8) | cl_uint(input[i * 3 + 2]);
            if (preview[i] != expected)
                return 1;
        }
        std::cout << "PASS enabled=" << enabled << " values=" << input.size()
                    << " max_abs_error=" << max_error << " preview=unchanged\n";
    }
    clReleaseMemObject(in);
    clReleaseMemObject(out);
    clReleaseMemObject(rgb);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
