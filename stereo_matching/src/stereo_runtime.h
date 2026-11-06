#ifndef STEREO_RUNTIME_H
#define STEREO_RUNTIME_H

// Private runtime primitives shared by the stereo pipeline and preprocessing.
// Kept in this header to preserve the existing single-translation-unit build.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <CL/cl_ext.h>
#include <sys/mman.h>
#include <unistd.h>

#include "BufferAllocatorWrapper.h"
#include "opencl_profiling.h"

#ifndef LAS2_WITH_STEREO_MATCHING
#include <onnxruntime_cxx_api.h>
#ifdef __riscv
#include "spacemit_ort_env.h"
#endif
#else
#include "tensor_engine_adapter.h"
#endif

namespace
{

std::string errno_msg(const std::string& what) { return what + ": " + std::strerror(errno); }

void check(bool bad, const std::string& msg)
{
    if (bad)
    {
        throw std::runtime_error(msg);
    }
}

size_t align_up(size_t value, size_t align) { return ((value + align - 1) / align) * align; }

class DmaBuffer
{
public:
    DmaBuffer(BufferAllocator* allocator, size_t size)
        : allocator_(allocator), size_(size), map_size_(align_up(size, 4096))
    {
        try
        {
            fd_ = DmabufHeapAllocSystem(allocator_, true, map_size_, 0, 0);
            check(fd_ < 0, "DmabufHeapAllocSystem failed");
            map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            check(map_ == MAP_FAILED, errno_msg("mmap dmabuf"));
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~DmaBuffer() noexcept { cleanup(); }

    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    int fd() const { return fd_; }
    void* data() const { return map_; }
    size_t size() const { return size_; }
    void sync_start(SyncType type)
    {
        DmabufHeapCpuSyncStart(allocator_, static_cast<unsigned int>(fd_), type);
    }
    void sync_end(SyncType type)
    {
        DmabufHeapCpuSyncEnd(allocator_, static_cast<unsigned int>(fd_), type);
    }

private:
    void cleanup() noexcept
    {
        if (map_ != MAP_FAILED)
        {
            munmap(map_, map_size_);
            map_ = MAP_FAILED;
        }
        if (fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }

    BufferAllocator* allocator_ = nullptr;
    int fd_ = -1;
    size_t size_ = 0;
    size_t map_size_ = 0;
    void* map_ = MAP_FAILED;
};

using PFN_clImportMemoryARM = cl_mem(CL_API_CALL*)(
    cl_context, cl_mem_flags, const cl_import_properties_arm*, void*, size_t, cl_int*);
using PFN_clEnqueueAcquireExternalMemObjectsKHR = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*);
using PFN_clEnqueueReleaseExternalMemObjectsKHR = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*);

#ifndef LAS2_WITH_STEREO_MATCHING
class Las2Engine
{
public:
    Las2Engine(const std::string& model_path, const std::string& affinity, int threads = 8)
        : env_(ORT_LOGGING_LEVEL_WARNING, "stereo_runtime")
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef __riscv
        std::unordered_map<std::string, std::string> provider_options;
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(threads);
        provider_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
        Ort::SessionOptionsSpaceMITEnvInit(options, provider_options);
#endif
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
        check(session_->GetInputCount() != 2 || session_->GetOutputCount() < 1,
            "LAS2 requires two image inputs and a disparity output");
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i)
        {
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
        {
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
        }
        // TensorTypeAndShapeInfo is a borrowed view; retain its TypeInfo owner.
        const auto input_type_info = session_->GetInputTypeInfo(0);
        const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        input_type_ = input_tensor_info.GetElementType();
        const auto right_type_info = session_->GetInputTypeInfo(1);
        const auto right_tensor_info = right_type_info.GetTensorTypeAndShapeInfo();
        check(right_tensor_info.GetShape() != input_shape_ ||
                right_tensor_info.GetElementType() != input_type_,
            "LAS2 left/right input shapes and types must match");
        check(input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            "LAS2 inputs must be float32 or float16");
        check(input_shape_.size() != 4 || input_shape_[0] != 1,
            "LAS2 input must be static rank-4 with batch size one");
        if (input_shape_[1] == 3)
        {
            input_layout_ = TensorLayout::Nchw;
            input_h_ = static_cast<int>(input_shape_[2]);
            input_w_ = static_cast<int>(input_shape_[3]);
        }
        else if (input_shape_[3] == 3)
        {
            input_layout_ = TensorLayout::Nhwc;
            input_h_ = static_cast<int>(input_shape_[1]);
            input_w_ = static_cast<int>(input_shape_[2]);
        }
        else
        {
            throw std::runtime_error("LAS2 input shape is neither NCHW nor NHWC");
        }
        check(input_h_ <= 0 || input_w_ <= 0, "LAS2 input shape must be static");

        const auto output_type_info = session_->GetOutputTypeInfo(0);
        const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_type_ = output_tensor_info.GetElementType();
        check(output_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                output_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            "LAS2 disparity output must be float32 or float16");
        output_shape_ = output_tensor_info.GetShape();
        output_elems_ = output_tensor_info.GetElementCount();
        if (output_shape_.empty() || output_elems_ == 0)
        {
            output_shape_ = {1, 1, input_h_, input_w_};
            output_elems_ = static_cast<size_t>(input_h_) * input_w_;
        }
        if (output_shape_.size() >= 2)
        {
            output_h_ = static_cast<int>(output_shape_[output_shape_.size() - 2]);
            output_w_ = static_cast<int>(output_shape_[output_shape_.size() - 1]);
        }
        if (output_h_ <= 0 || output_w_ <= 0)
        {
            output_h_ = input_h_;
            output_w_ = input_w_;
        }

        std::cout << "[LAS2] model=" << model_path << " affinity=" << affinity
                    << " threads=" << threads << " inputs=" << input_names_[0] << ","
                    << input_names_[1] << " output=" << output_names_[0]
                    << " input_shape=" << shape_to_string(input_shape_)
                    << " output_shape=" << shape_to_string(output_shape_)
                    << " input_type=" << (input_is_fp16() ? "float16" : "float32") << " output_type="
                    << (output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "float16" : "float32")
                    << " input_bytes/eye=" << input_bytes() << "\n";
    }

    void infer(float* left, float* right, std::vector<float>& disp)
    {
        // Legacy CPU callers provide float32. DMA callers use infer_native()
        // directly so GPU-produced float16 is never expanded and repacked.
        if (input_is_fp16())
        {
            const size_t count = input_bytes() / sizeof(Ort::Float16_t);
            left_half_.resize(count);
            right_half_.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                left_half_[i] = Ort::Float16_t(left[i]);
                right_half_[i] = Ort::Float16_t(right[i]);
            }
            infer_native(left_half_.data(), right_half_.data(), disp);
        }
        else
        {
            infer_native(left, right, disp);
        }
    }

    void infer_native(void* left, void* right, std::vector<float>& disp)
    {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor(
            mem, left, input_bytes(), input_shape_.data(), input_shape_.size(), input_type_));
        inputs.push_back(Ort::Value::CreateTensor(
            mem, right, input_bytes(), input_shape_.data(), input_shape_.size(), input_type_));
        std::vector<const char*> in_names = {input_names_[0].c_str(), input_names_[1].c_str()};
        std::vector<const char*> out_names = {output_names_[0].c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
            inputs.size(), out_names.data(), out_names.size());
        const size_t out_elems = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        disp.resize(out_elems);
        if (output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            const auto* ptr = outputs[0].GetTensorData<Ort::Float16_t>();
            for (size_t i = 0; i < out_elems; ++i)
                disp[i] = ptr[i].ToFloat();
        }
        else
        {
            const float* ptr = outputs[0].GetTensorData<float>();
            std::copy(ptr, ptr + out_elems, disp.begin());
        }
    }

    void infer_to(float* left, float* right, float* disp)
    {
        if (input_is_fp16() || output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            infer(left, right, converted_disparity_);
            std::copy(converted_disparity_.begin(), converted_disparity_.end(), disp);
            return;
        }
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const size_t tensor_elems = input_layout_ == TensorLayout::Nchw ?
            static_cast<size_t>(3) * input_h_ * input_w_ :
            static_cast<size_t>(input_h_) * input_w_ * 3;
        std::array<Ort::Value, 2> inputs = {Ort::Value::CreateTensor<float>(mem, left, tensor_elems,
                                                input_shape_.data(), input_shape_.size()),
            Ort::Value::CreateTensor<float>(
                mem, right, tensor_elems, input_shape_.data(), input_shape_.size())};
        Ort::Value output = Ort::Value::CreateTensor<float>(
            mem, disp, output_elems_, output_shape_.data(), output_shape_.size());
        const std::array<const char*, 2> in_names = {
            input_names_[0].c_str(), input_names_[1].c_str()};
        const std::array<const char*, 1> out_names = {output_names_[0].c_str()};
        session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
            out_names.data(), &output, 1);
    }

    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }
    int output_width() const { return output_w_; }
    int output_height() const { return output_h_; }
    bool input_is_nhwc() const { return input_layout_ == TensorLayout::Nhwc; }
    bool input_is_fp16() const { return input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }
    size_t input_bytes() const
    {
        return static_cast<size_t>(input_w_) * input_h_ * 3 *
            (input_is_fp16() ? sizeof(Ort::Float16_t) : sizeof(float));
    }

private:
    enum class TensorLayout
    {
        Nchw,
        Nhwc,
    };

    static std::string shape_to_string(const std::vector<int64_t>& shape)
    {
        std::string out = "[";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i)
            {
                out += ",";
            }
            out += std::to_string(shape[i]);
        }
        out += "]";
        return out;
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    size_t output_elems_ = 0;
    ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType output_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<Ort::Float16_t> left_half_, right_half_;
    std::vector<float> converted_disparity_;
    TensorLayout input_layout_ = TensorLayout::Nchw;
    int input_w_ = 0;
    int input_h_ = 0;
    int output_w_ = 0;
    int output_h_ = 0;
};

#endif

std::vector<std::string> split_csv(const std::string& value)
{
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size())
    {
        const size_t comma = value.find(',', start);
        std::string item =
            value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!item.empty())
        {
            out.push_back(item);
        }
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    return out;
}

} // namespace

#endif // STEREO_RUNTIME_H
