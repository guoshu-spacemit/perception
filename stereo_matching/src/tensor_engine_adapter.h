#ifndef TENSOR_ENGINE_ADAPTER_H
#define TENSOR_ENGINE_ADAPTER_H

#include <dlfcn.h>
#include <sched.h>
#include <stereo_matching.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// Keep one facade/session per existing LAS2 engine. The enclosing pipeline
// retains ownership of scheduling, input DMA leases and output buffers.
class Las2Engine
{
public:
    Las2Engine(const std::string& model, const std::string& affinity, int threads = 8)
    {
        std::string list = affinity;
        std::replace(list.begin(), list.end(), ';', ',');
        std::istringstream stream(list);
        std::string token;
        std::vector<int> cores;
        while (std::getline(stream, token, ','))
        {
            std::size_t consumed = 0;
            const int core = std::stoi(token, &consumed);
            if (consumed != token.size() || core < 0 || core >= CPU_SETSIZE ||
                (!cores.empty() && core != cores.back() + 1))
            {
                throw std::runtime_error(
                    "stereo_matching requires consecutive ordered affinity CPUs");
            }
            cores.push_back(core);
        }
        if (cores.empty() || threads <= 0 || static_cast<int>(cores.size()) != threads ||
            (!list.empty() && list.back() == ','))
        {
            throw std::runtime_error("stereo_matching affinity must match threads_per_instance");
        }
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        {
            throw std::runtime_error("cannot read facade worker affinity");
        }
        int facade_cpu = -1;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (CPU_ISSET(cpu, &allowed) &&
                std::find(cores.begin(), cores.end(), cpu) == cores.end())
            {
                facade_cpu = cpu;
                break;
            }
        }
        if (facade_cpu < 0)
        {
            throw std::runtime_error("facade worker needs an allowed CPU outside the EP group");
        }
        validate_runtime();
        spacemit::stereo::Config config;
        config.model_path = model.c_str();
        config.a100_core_count = threads;
        config.instance_count = 1;
        config.first_a100_cpu = cores.front();
        config.first_x100_cpu = facade_cpu;
        // A per-engine initializer must never clear TCM owned by other engines.
        // Operators may clear stale ownership once, before starting the pipeline.
        config.clear_tcm = false;
        config.switch_workers_to_a100 = false;
        if (!matcher_.initialize(config))
        {
            throw std::runtime_error(matcher_.last_error());
        }
        info_ = matcher_.model_info();
        if (info_.input_width <= 0 || info_.input_height <= 0 || info_.output_width <= 0 ||
            info_.output_height <= 0 ||
            info_.input_elements !=
                static_cast<std::size_t>(info_.input_width) * info_.input_height * 3 ||
            info_.output_elements !=
                static_cast<std::size_t>(info_.output_width) * info_.output_height ||
            (info_.input_layout != spacemit::stereo::TensorLayout::nchw &&
                info_.input_layout != spacemit::stereo::TensorLayout::nhwc))
        {
            throw std::runtime_error("stereo_matching returned unsupported tensor geometry");
        }
        std::cout << "[LAS2] backend=stereo_matching model=" << model << " affinity=" << affinity
                    << " threads=" << threads << " facade_cpu=" << facade_cpu
                    << " input=" << input_width() << 'x' << input_height() << '\n';
    }

    static void validate_runtime()
    {
        // This binary SDK initializes its embedded EP allocator through a
        // pthread_create interposer. A transitive dependency can otherwise lose
        // symbol precedence to libc and crash on the first inference request.
        void* library = dlopen("libstereo_matching.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!library)
        {
            throw std::runtime_error("cannot inspect loaded stereo_matching runtime");
        }
        void* required_create = dlsym(library, "pthread_create");
        void* active_create = dlsym(RTLD_DEFAULT, "pthread_create");
        dlclose(library);
        if (!required_create || active_create != required_create)
        {
            throw std::runtime_error(
                "stereo_matching requires startup LD_PRELOAD=/path/to/"
                "stereo_matching/lib/libstereo_matching.so for its EP thread initializer");
        }
    }

    void infer(float* left, float* right, std::vector<float>& disparity)
    {
        disparity.resize(info_.output_elements);
        infer_to(left, right, disparity.data());
    }
    void infer_native(void* left, void* right, std::vector<float>& disparity)
    {
        infer(static_cast<float*>(left), static_cast<float*>(right), disparity);
    }
    bool input_is_fp16() const { return false; }
    size_t input_bytes() const { return info_.input_elements * sizeof(float); }
    void infer_to(float* left, float* right, float* disparity)
    {
        if (!matcher_.infer(left, info_.input_elements, right, info_.input_elements, disparity,
                info_.output_elements))
        {
            throw std::runtime_error(matcher_.last_error());
        }
    }
    int input_width() const { return info_.input_width; }
    int input_height() const { return info_.input_height; }
    int output_width() const { return info_.output_width; }
    int output_height() const { return info_.output_height; }
    bool input_is_nhwc() const
    {
        return info_.input_layout == spacemit::stereo::TensorLayout::nhwc;
    }

private:
    spacemit::stereo::StereoMatcher matcher_;
    spacemit::stereo::ModelInfo info_{};
};

} // namespace

#endif // TENSOR_ENGINE_ADAPTER_H
