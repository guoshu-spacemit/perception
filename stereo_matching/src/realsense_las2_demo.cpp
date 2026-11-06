#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <condition_variable>
#include <unordered_map>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>
#include <vector>

#include <CL/cl_ext.h>
#include <drm.h>
#include <drm_fourcc.h>
#include <linux/videodev2.h>
#ifndef LAS2_WITH_STEREO_MATCHING
#include <onnxruntime_cxx_api.h>
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "BufferAllocatorWrapper.h"
#include "opencl_profiling.hpp"

#if defined(__riscv) && !defined(LAS2_WITH_STEREO_MATCHING)
#include "spacemit_ort_env.h"
#endif

#ifdef LAS2_WITH_STEREO_MATCHING
#include "stereo_matching_engine.hpp"
#endif

namespace {

volatile sig_atomic_t g_running = 1;
constexpr int kModelW = 320;
constexpr int kModelH = 256;

void on_signal(int) {
    g_running = 0;
}

std::string errno_msg(const std::string& what) {
    return what + ": " + std::strerror(errno);
}

void check(bool bad, const std::string& msg) {
    if (bad) {
        throw std::runtime_error(msg);
    }
}

uint32_t fourcc(char a, char b, char c, char d) {
    return v4l2_fourcc(a, b, c, d);
}

struct Rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

struct KittiPair {
    std::string left;
    std::string right;
};

std::vector<std::filesystem::path> list_pngs(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::is_directory(dir)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".png") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<KittiPair> collect_kitti_pairs(const std::string& root) {
    const std::filesystem::path base(root);
    const auto left = list_pngs(base / "image_02" / "data");
    const auto right = list_pngs(base / "image_03" / "data");
    const size_t n = std::min(left.size(), right.size());
    std::vector<KittiPair> pairs;
    pairs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (left[i].filename() != right[i].filename()) {
            throw std::runtime_error("KITTI left/right filename mismatch: " +
                                     left[i].filename().string() + " vs " + right[i].filename().string());
        }
        pairs.push_back({left[i].string(), right[i].string()});
    }
    return pairs;
}

Rect compute_fit_rect(int cell_x, int cell_y, int cell_w, int cell_h, int src_w, int src_h) {
    if (cell_w <= 0 || cell_h <= 0 || src_w <= 0 || src_h <= 0) {
        return {};
    }
    const double scale = std::min(
        static_cast<double>(cell_w) / static_cast<double>(src_w),
        static_cast<double>(cell_h) / static_cast<double>(src_h));
    const int out_w = std::max(1, static_cast<int>(src_w * scale));
    const int out_h = std::max(1, static_cast<int>(src_h * scale));
    return {
        cell_x + (cell_w - out_w) / 2,
        cell_y + (cell_h - out_h) / 2,
        out_w,
        out_h,
    };
}

int run_layout_self_test() {
    Rect r = compute_fit_rect(0, 0, 640, 360, 848, 480);
    if (r.w != 636 || r.h != 360 || r.x != 2 || r.y != 0) {
        std::fprintf(stderr, "layout test failed: got %d,%d %dx%d\n", r.x, r.y, r.w, r.h);
        return 1;
    }
    r = compute_fit_rect(640, 0, 640, 360, 320, 320);
    if (r.w != 360 || r.h != 360 || r.x != 780 || r.y != 0) {
        std::fprintf(stderr, "square layout test failed: got %d,%d %dx%d\n", r.x, r.y, r.w, r.h);
        return 1;
    }
    std::puts("[SelfTest] layout behavior verified");
    return 0;
}

int run_kitti_self_test() {
    const std::filesystem::path root = "../assets/kitti_self_test";
    std::filesystem::create_directories(root / "image_02" / "data");
    std::filesystem::create_directories(root / "image_03" / "data");
    for (const char* name : {"0000000000.png", "0000000001.png"}) {
        std::ofstream(root / "image_02" / "data" / name).put('\0');
        std::ofstream(root / "image_03" / "data" / name).put('\0');
    }
    const auto pairs = collect_kitti_pairs(root.string());
    if (pairs.size() != 2 ||
        pairs[0].left.find("image_02/data/0000000000.png") == std::string::npos ||
        pairs[0].right.find("image_03/data/0000000000.png") == std::string::npos ||
        pairs[1].left.find("image_02/data/0000000001.png") == std::string::npos ||
        pairs[1].right.find("image_03/data/0000000001.png") == std::string::npos) {
        std::fprintf(stderr, "kitti pair discovery test failed\n");
        return 1;
    }
    std::puts("[SelfTest] KITTI pair discovery verified");
    return 0;
}

int parse_int(const char* value, const char* name, int min_value, int max_value) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < min_value || parsed > max_value) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

const char* need_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

struct Options {
    std::string video = "/dev/video11";
    std::string card = "/dev/dri/card1";
    int width = 848;
    int height = 480;
    int fps = 30;
    int frames = 0;
    uint32_t pixel_format = 0;
    bool bench_preprocess = false;
    int loops = 100;
    bool run_las2 = false;
    bool verbose = false;
    int save_warmup_frames = 30;
    double playback_fps = 10.0;
    int pipeline_instances = 1;
    int pipeline_threads = 1;
    int display_streams = 1;
    double input_scale = 0.0;
    double disp_vis_alpha = 0.05;
    double disp_vis_min = std::numeric_limits<double>::quiet_NaN();
    double disp_vis_max = std::numeric_limits<double>::quiet_NaN();
    std::string kitti_seq;
    std::string save_depth = "../assets/realsense_depth_ep.png";
    std::string save_view = "../assets/realsense_view_ep.png";
    std::string model = "/root/c_envs/spacemit_ort/yyx_ws/LAS2_S_256x320.fp16.onnx";
    std::string affinity = "8;9;10;11;12;13;14;15";
    std::string pipeline_affinities = "8,9,10,11,12,13,14,15";
    bool offline_decode = false;
};

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dev") {
            opt.video = need_value(argc, argv, i);
        } else if (arg == "--card") {
            opt.card = need_value(argc, argv, i);
        } else if (arg == "--width") {
            opt.width = parse_int(need_value(argc, argv, i), "width", 2, 4096);
        } else if (arg == "--height") {
            opt.height = parse_int(need_value(argc, argv, i), "height", 2, 4096);
        } else if (arg == "--fps") {
            opt.fps = parse_int(need_value(argc, argv, i), "fps", 1, 300);
        } else if (arg == "--frames") {
            opt.frames = parse_int(need_value(argc, argv, i), "frames", 0, 1000000);
        } else if (arg == "--bench-preprocess") {
            opt.bench_preprocess = true;
        } else if (arg == "--run-las2") {
            opt.run_las2 = true;
        } else if (arg == "--verbose") {
            opt.verbose = true;
        } else if (arg == "--kitti-seq") {
            opt.kitti_seq = need_value(argc, argv, i);
        } else if (arg == "--playback-fps") {
            opt.playback_fps = std::stod(need_value(argc, argv, i));
            check(opt.playback_fps <= 0.0 || opt.playback_fps > 120.0, "playback-fps out of range");
        } else if (arg == "--offline-decode") {
            opt.offline_decode = true;
        } else if (arg == "--pipeline-instances") {
            opt.pipeline_instances = parse_int(need_value(argc, argv, i), "pipeline-instances", 1, 32);
        } else if (arg == "--pipeline-threads") {
            opt.pipeline_threads = parse_int(need_value(argc, argv, i), "pipeline-threads", 1, 16);
        } else if (arg == "--pipeline-affinities") {
            opt.pipeline_affinities = need_value(argc, argv, i);
        } else if (arg == "--display-streams") {
            opt.display_streams = parse_int(need_value(argc, argv, i), "display-streams", 1, 8);
        } else if (arg == "--input-scale") {
            opt.input_scale = std::stod(need_value(argc, argv, i));
        } else if (arg == "--disp-vis-alpha") {
            opt.disp_vis_alpha = std::clamp(std::stod(need_value(argc, argv, i)), 0.0, 1.0);
        } else if (arg == "--disp-vis-min") {
            opt.disp_vis_min = std::stod(need_value(argc, argv, i));
        } else if (arg == "--disp-vis-max") {
            opt.disp_vis_max = std::stod(need_value(argc, argv, i));
        } else if (arg == "--save-depth") {
            opt.save_depth = need_value(argc, argv, i);
        } else if (arg == "--save-view") {
            opt.save_view = need_value(argc, argv, i);
        } else if (arg == "--save-warmup-frames") {
            opt.save_warmup_frames = parse_int(need_value(argc, argv, i), "save-warmup-frames", 0, 1000000);
        } else if (arg == "--model") {
            opt.model = need_value(argc, argv, i);
        } else if (arg == "--affinity") {
            opt.affinity = need_value(argc, argv, i);
        } else if (arg == "--loops") {
            opt.loops = parse_int(need_value(argc, argv, i), "loops", 1, 100000);
        } else if (arg == "--format") {
            const std::string fmt = need_value(argc, argv, i);
            if (fmt == "Y8I") {
                opt.pixel_format = fourcc('Y', '8', 'I', ' ');
            } else if (fmt == "UYVY") {
                opt.pixel_format = V4L2_PIX_FMT_UYVY;
            } else {
                throw std::invalid_argument("unsupported --format: " + fmt);
            }
        } else if (arg == "--self-test-layout") {
            std::exit(run_layout_self_test());
        } else if (arg == "--self-test-kitti") {
            std::exit(run_kitti_self_test());
        } else if (arg == "--help" || arg == "-h") {
            std::printf("usage: %s [--dev /dev/video11] [--card /dev/dri/card1]\n"
                        "          [--width 848] [--height 480] [--fps 30] [--frames 0]\n"
                        "          [--kitti-seq ../datasets/2011_09_26_drive_0009_sync]\n"
                        "          [--input-scale 0] [--playback-fps 10]\n"
                        "          [--offline-decode] [--pipeline-instances 8]\n"
                        "          [--pipeline-affinities 8,9,10,11,12,13,14,15]\n"
                        "          [--run-las2] [--save-depth path.png] [--save-view path.png]\n"
                        "          [--save-warmup-frames 30]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!opt.pixel_format) {
        opt.pixel_format = V4L2_PIX_FMT_UYVY;
    }
    return opt;
}

class V4l2IrCapture {
private:
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
        int dma_fd = -1;
    };

public:
    explicit V4l2IrCapture(const Options& opt) : pixel_format_(opt.pixel_format), width_(opt.width), height_(opt.height) {
        fd_ = open(opt.video.c_str(), O_RDWR | O_CLOEXEC);
        check(fd_ < 0, errno_msg(opt.video));

        v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = static_cast<uint32_t>(opt.width);
        fmt.fmt.pix.height = static_cast<uint32_t>(opt.height);
        fmt.fmt.pix.pixelformat = pixel_format_;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        check(ioctl(fd_, VIDIOC_S_FMT, &fmt) != 0, errno_msg("VIDIOC_S_FMT IR"));
        check(fmt.fmt.pix.pixelformat != pixel_format_, "camera did not accept requested IR format");
        width_ = static_cast<int>(fmt.fmt.pix.width);
        height_ = static_cast<int>(fmt.fmt.pix.height);
        bytes_per_line_ = static_cast<int>(fmt.fmt.pix.bytesperline);
        if (bytes_per_line_ == 0) {
            bytes_per_line_ = width_ * 2;
        }

        v4l2_streamparm parm = {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(opt.fps);
        ioctl(fd_, VIDIOC_S_PARM, &parm);

        v4l2_requestbuffers req = {};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        check(ioctl(fd_, VIDIOC_REQBUFS, &req) != 0, errno_msg("VIDIOC_REQBUFS"));
        check(req.count < 2, "not enough V4L2 buffers");

        buffers_.resize(req.count);
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            check(ioctl(fd_, VIDIOC_QUERYBUF, &buf) != 0, errno_msg("VIDIOC_QUERYBUF"));
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            check(buffers_[i].start == MAP_FAILED, errno_msg("mmap V4L2 buffer"));
            v4l2_exportbuffer exp = {};
            exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            exp.index = i;
            exp.flags = O_CLOEXEC;
            check(ioctl(fd_, VIDIOC_EXPBUF, &exp) != 0, errno_msg("VIDIOC_EXPBUF"));
            buffers_[i].dma_fd = exp.fd;
            check(ioctl(fd_, VIDIOC_QBUF, &buf) != 0, errno_msg("VIDIOC_QBUF"));
        }

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        check(ioctl(fd_, VIDIOC_STREAMON, &type) != 0, errno_msg("VIDIOC_STREAMON"));
        streaming_ = true;

        char f[5] = {
            static_cast<char>(pixel_format_ & 0xff),
            static_cast<char>((pixel_format_ >> 8) & 0xff),
            static_cast<char>((pixel_format_ >> 16) & 0xff),
            static_cast<char>((pixel_format_ >> 24) & 0xff),
            0,
        };
        std::printf("[V4L2] %s %s %dx%d stride=%d\n", opt.video.c_str(), f, width_, height_, bytes_per_line_);
    }

    ~V4l2IrCapture() {
        std::puts("[CLEANUP] V4L2 streamoff");
        if (streaming_) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
        }
        std::puts("[CLEANUP] V4L2 unmap buffers");
        for (auto& b : buffers_) {
            if (b.start && b.start != MAP_FAILED) {
                munmap(b.start, b.length);
            }
            if (b.dma_fd >= 0) {
                close(b.dma_fd);
            }
        }
        std::puts("[CLEANUP] V4L2 close fd");
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool read(std::vector<uint8_t>& left, std::vector<uint8_t>& right) {
        v4l2_buffer buf = {};
        const Buffer* captured = dequeue(buf);
        if (!captured) {
            return false;
        }

        split_frame(static_cast<const uint8_t*>(captured->start), left, right);
        check(ioctl(fd_, VIDIOC_QBUF, &buf) != 0, errno_msg("VIDIOC_QBUF return"));
        return true;
    }

    struct CapturedFrame {
        const uint8_t* data = nullptr;
        int dma_fd = -1;
        int index = -1;
        v4l2_buffer buf = {};
    };

    bool acquire(CapturedFrame& frame) {
        const Buffer* captured = dequeue(frame.buf);
        if (!captured) {
            return false;
        }
        frame.data = static_cast<const uint8_t*>(captured->start);
        frame.dma_fd = captured->dma_fd;
        frame.index = static_cast<int>(frame.buf.index);
        return true;
    }

    void release(const CapturedFrame& frame) {
        v4l2_buffer buf = frame.buf;
        check(ioctl(fd_, VIDIOC_QBUF, &buf) != 0, errno_msg("VIDIOC_QBUF release"));
    }

    void split_frame(const uint8_t* src, std::vector<uint8_t>& left, std::vector<uint8_t>& right) const {
        left.assign(static_cast<size_t>(width_) * height_, 0);
        right.assign(static_cast<size_t>(width_) * height_, 0);
        if (pixel_format_ == fourcc('Y', '8', 'I', ' ')) {
            for (int y = 0; y < height_; ++y) {
                const uint8_t* row = src + static_cast<size_t>(y) * bytes_per_line_;
                for (int x = 0; x < width_; ++x) {
                    left[static_cast<size_t>(y) * width_ + x] = row[x * 2 + 0];
                    right[static_cast<size_t>(y) * width_ + x] = row[x * 2 + 1];
                }
            }
        } else {
            for (int y = 0; y < height_; ++y) {
                const uint8_t* row = src + static_cast<size_t>(y) * bytes_per_line_;
                for (int x = 0; x < width_; x += 2) {
                    const size_t idx0 = static_cast<size_t>(y) * width_ + x;
                    const size_t idx1 = idx0 + 1;
                    left[idx0] = row[x * 2 + 1];
                    right[idx0] = row[x * 2 + 3];
                    if (x + 1 < width_) {
                        left[idx1] = left[idx0];
                        right[idx1] = right[idx0];
                    }
                }
            }
        }
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return bytes_per_line_; }
    size_t buffer_size() const { return buffers_.empty() ? 0 : buffers_[0].length; }

private:
    const Buffer* dequeue(v4l2_buffer& buf) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        timeval tv = {};
        tv.tv_sec = 2;
        int ready = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
        if (ready == 0) {
            std::fprintf(stderr, "[V4L2] frame timeout\n");
            return nullptr;
        }
        if (ready < 0) {
            throw std::runtime_error(errno_msg("select video"));
        }

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) != 0) {
            if (errno == EAGAIN) {
                return nullptr;
            }
            throw std::runtime_error(errno_msg("VIDIOC_DQBUF"));
        }
        return &buffers_[buf.index];
    }

    int fd_ = -1;
    uint32_t pixel_format_ = 0;
    int width_ = 0;
    int height_ = 0;
    int bytes_per_line_ = 0;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
};

size_t align_up(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

class DmaBuffer {
public:
    DmaBuffer(BufferAllocator* allocator, size_t size) : allocator_(allocator), size_(size), map_size_(align_up(size, 4096)) {
        fd_ = DmabufHeapAllocSystem(allocator_, true, map_size_, 0, 0);
        check(fd_ < 0, "DmabufHeapAllocSystem failed");
        map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        check(map_ == MAP_FAILED, errno_msg("mmap dmabuf"));
    }

    ~DmaBuffer() {
        if (map_ && map_ != MAP_FAILED) {
            munmap(map_, map_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    int fd() const { return fd_; }
    void* data() const { return map_; }
    size_t size() const { return size_; }
    void sync_start(SyncType type) { DmabufHeapCpuSyncStart(allocator_, static_cast<unsigned int>(fd_), type); }
    void sync_end(SyncType type) { DmabufHeapCpuSyncEnd(allocator_, static_cast<unsigned int>(fd_), type); }

private:
    BufferAllocator* allocator_ = nullptr;
    int fd_ = -1;
    size_t size_ = 0;
    size_t map_size_ = 0;
    void* map_ = nullptr;
};

const char* kPreprocessKernel = R"CLC(
__kernel void uyvy_to_las2_tensors(
    __global const uchar* uyvy,
    __global float* out,
    int in_w,
    int in_h,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= 320 || y >= 256) return;

    const int resized_h = (in_h * 320) / in_w;
    const int top = (256 - resized_h) / 2;
    int sy;
    if (y < top) {
        sy = 0;
    } else if (y >= top + resized_h) {
        sy = in_h - 1;
    } else {
        sy = ((y - top) * in_h) / resized_h;
    }
    int sx = (x * in_w) / 320;
    sx = sx & ~1;

    __global const uchar* row = uyvy + sy * stride + sx * 2;
    float left = (float)row[1];
    float right = (float)row[3];
    int plane = 320 * 256;
    int idx = y * 320 + x;
    int right_base = plane * 3;
    out[idx] = left;
    out[idx + plane] = left;
    out[idx + plane * 2] = left;
    out[right_base + idx] = right;
    out[right_base + idx + plane] = right;
    out[right_base + idx + plane * 2] = right;
}
)CLC";

using PFN_clImportMemoryARM = cl_mem(CL_API_CALL*)(
    cl_context, cl_mem_flags, const cl_import_properties_arm*, void*, size_t, cl_int*);
using PFN_clEnqueueAcquireExternalMemObjectsKHR = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*);
using PFN_clEnqueueReleaseExternalMemObjectsKHR = cl_int(CL_API_CALL*)(
    cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*);

class OpenClPreprocessor {
public:
    OpenClPreprocessor() {
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
            clGetExtensionFunctionAddressForPlatform(platform_, "clEnqueueAcquireExternalMemObjectsKHR"));
        release_ = reinterpret_cast<PFN_clEnqueueReleaseExternalMemObjectsKHR>(
            clGetExtensionFunctionAddressForPlatform(platform_, "clEnqueueReleaseExternalMemObjectsKHR"));
        check(!import_memory_, "clImportMemoryARM not available");
        check(!acquire_ || !release_, "external memory acquire/release not available");

        const char* src = kPreprocessKernel;
        size_t len = std::strlen(kPreprocessKernel);
        program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
        check(!program_ || err != CL_SUCCESS, "clCreateProgramWithSource failed");
        err = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
            throw std::runtime_error("clBuildProgram failed:\n" + log);
        }
        kernel_ = clCreateKernel(program_, "uyvy_to_las2_tensors", &err);
        check(!kernel_ || err != CL_SUCCESS, "clCreateKernel failed");
    }

    ~OpenClPreprocessor() {
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    cl_mem import_dmabuf(int fd, size_t size) {
        const cl_import_properties_arm props[] = {
            CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM,
            0,
        };
        cl_int err = CL_SUCCESS;
        cl_mem mem = import_memory_(context_, CL_MEM_READ_WRITE, props, &fd, size, &err);
        check(!mem || err != CL_SUCCESS, "clImportMemoryARM failed: " + std::to_string(err));
        return mem;
    }

    double run(cl_mem input, cl_mem output, int in_w, int in_h, int stride) {
        cl_int err = CL_SUCCESS;
        err |= acquire_(queue_, 1, &input, 0, nullptr, nullptr);
        err |= acquire_(queue_, 1, &output, 0, nullptr, nullptr);
        check(err != CL_SUCCESS, "acquire external memory failed");

        int arg = 0;
        err  = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &input);
        err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &output);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &in_w);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &in_h);
        err |= clSetKernelArg(kernel_, arg++, sizeof(int), &stride);
        check(err != CL_SUCCESS, "clSetKernelArg failed");

        size_t global[2] = {kModelW, kModelH};
        cl_event event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, kernel_, 2, nullptr, global, nullptr, 0, nullptr, &event);
        check(err != CL_SUCCESS, "clEnqueueNDRangeKernel failed");

        err = release_(queue_, 1, &input, 0, nullptr, nullptr);
        err |= release_(queue_, 1, &output, 0, nullptr, nullptr);
        check(err != CL_SUCCESS, "release external memory failed");
        clFinish(queue_);

        const double elapsed_ms = las2::detail::opencl_event_ms(event);
        clReleaseEvent(event);
        return elapsed_ms;
    }

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    PFN_clImportMemoryARM import_memory_ = nullptr;
    PFN_clEnqueueAcquireExternalMemObjectsKHR acquire_ = nullptr;
    PFN_clEnqueueReleaseExternalMemObjectsKHR release_ = nullptr;
};

void cpu_preprocess_uyvy(const uint8_t* uyvy, int in_w, int in_h, int stride, std::vector<float>& out) {
    const int plane = kModelW * kModelH;
    out.assign(static_cast<size_t>(plane) * 6, 0.0f);
    const int resized_h = (in_h * kModelW) / in_w;
    const int top = (kModelH - resized_h) / 2;
    const int right_base = plane * 3;
    for (int y = 0; y < kModelH; ++y) {
        int sy = 0;
        if (y < top) {
            sy = 0;
        } else if (y >= top + resized_h) {
            sy = in_h - 1;
        } else {
            sy = ((y - top) * in_h) / resized_h;
        }
        const uint8_t* row = uyvy + static_cast<size_t>(sy) * stride;
        for (int x = 0; x < kModelW; ++x) {
            int sx = (x * in_w) / kModelW;
            sx &= ~1;
            float left = static_cast<float>(row[sx * 2 + 1]);
            float right = static_cast<float>(row[sx * 2 + 3]);
            int idx = y * kModelW + x;
            out[idx] = left;
            out[idx + plane] = left;
            out[idx + plane * 2] = left;
            out[right_base + idx] = right;
            out[right_base + idx + plane] = right;
            out[right_base + idx + plane * 2] = right;
        }
    }
}

#ifndef LAS2_WITH_STEREO_MATCHING
class Las2Engine {
public:
    Las2Engine(const std::string& model_path, const std::string& affinity, int threads = 8)
        : env_(ORT_LOGGING_LEVEL_WARNING, "realsense_las2") {
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
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
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
        if (input_shape_[1] == 3) {
            input_layout_ = TensorLayout::Nchw;
            input_h_ = static_cast<int>(input_shape_[2]);
            input_w_ = static_cast<int>(input_shape_[3]);
        } else if (input_shape_[3] == 3) {
            input_layout_ = TensorLayout::Nhwc;
            input_h_ = static_cast<int>(input_shape_[1]);
            input_w_ = static_cast<int>(input_shape_[2]);
        } else {
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
        if (output_shape_.empty() || output_elems_ == 0) {
            output_shape_ = {1, 1, input_h_, input_w_};
            output_elems_ = static_cast<size_t>(input_h_) * input_w_;
        }
        if (output_shape_.size() >= 2) {
            output_h_ = static_cast<int>(output_shape_[output_shape_.size() - 2]);
            output_w_ = static_cast<int>(output_shape_[output_shape_.size() - 1]);
        }
        if (output_h_ <= 0 || output_w_ <= 0) {
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

    void infer(float* left, float* right, std::vector<float>& disp) {
        // Legacy CPU callers provide float32. DMA callers use infer_native()
        // directly so GPU-produced float16 is never expanded and repacked.
        if (input_is_fp16()) {
            const size_t count = input_bytes() / sizeof(Ort::Float16_t);
            left_half_.resize(count);
            right_half_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                left_half_[i] = Ort::Float16_t(left[i]);
                right_half_[i] = Ort::Float16_t(right[i]);
            }
            infer_native(left_half_.data(), right_half_.data(), disp);
        } else {
            infer_native(left, right, disp);
        }
    }

    void infer_native(void* left, void* right, std::vector<float>& disp) {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor(mem, left, input_bytes(), input_shape_.data(),
                                                  input_shape_.size(), input_type_));
        inputs.push_back(Ort::Value::CreateTensor(mem, right, input_bytes(), input_shape_.data(),
                                                  input_shape_.size(), input_type_));
        std::vector<const char*> in_names = {input_names_[0].c_str(), input_names_[1].c_str()};
        std::vector<const char*> out_names = {output_names_[0].c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
                                     out_names.data(), out_names.size());
        const size_t out_elems = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        disp.resize(out_elems);
        if (output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* ptr = outputs[0].GetTensorData<Ort::Float16_t>();
            for (size_t i = 0; i < out_elems; ++i)
                disp[i] = ptr[i].ToFloat();
        } else {
            const float* ptr = outputs[0].GetTensorData<float>();
            std::copy(ptr, ptr + out_elems, disp.begin());
        }
    }

    void infer_to(float* left, float* right, float* disp) {
        if (input_is_fp16() || output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            infer(left, right, converted_disparity_);
            std::copy(converted_disparity_.begin(), converted_disparity_.end(), disp);
            return;
        }
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const size_t tensor_elems = input_layout_ == TensorLayout::Nchw
            ? static_cast<size_t>(3) * input_h_ * input_w_
            : static_cast<size_t>(input_h_) * input_w_ * 3;
        std::array<Ort::Value, 2> inputs = {
            Ort::Value::CreateTensor<float>(mem, left, tensor_elems,
                                            input_shape_.data(), input_shape_.size()),
            Ort::Value::CreateTensor<float>(mem, right, tensor_elems,
                                            input_shape_.data(), input_shape_.size())};
        Ort::Value output = Ort::Value::CreateTensor<float>(
            mem, disp, output_elems_, output_shape_.data(), output_shape_.size());
        const std::array<const char*, 2> in_names = {input_names_[0].c_str(), input_names_[1].c_str()};
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
    size_t input_bytes() const {
        return static_cast<size_t>(input_w_) * input_h_ * 3 *
               (input_is_fp16() ? sizeof(Ort::Float16_t) : sizeof(float));
    }

private:
    enum class TensorLayout {
        Nchw,
        Nhwc,
    };

    static std::string shape_to_string(const std::vector<int64_t>& shape) {
        std::string out = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) {
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

std::vector<uint32_t> disparity_to_xrgb(const std::vector<float>& disp, int in_w, int in_h, int& out_w, int& out_h) {
    out_w = kModelW;
    out_h = (in_h * kModelW) / in_w;
    int top = (kModelH - out_h) / 2;
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            float v = disp[static_cast<size_t>(y + top) * kModelW + x];
            if (std::isfinite(v)) {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
        }
    }
    std::vector<uint32_t> out(static_cast<size_t>(out_w) * out_h, 0xff000000u);
    if (min_v == std::numeric_limits<float>::max()) {
        return out;
    }
    if (max_v <= min_v) {
        max_v = min_v + 1.0f;
    }
    cv::Mat gray(out_h, out_w, CV_8UC1);
    for (int y = 0; y < out_h; ++y) {
        auto* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < out_w; ++x) {
            float v = disp[static_cast<size_t>(y + top) * kModelW + x];
            float normalized = std::isfinite(v) ? std::clamp((v - min_v) / (max_v - min_v), 0.0f, 1.0f) : 0.0f;
            row[x] = static_cast<uint8_t>(normalized * 255.0f + 0.5f);
        }
    }
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);
    for (int y = 0; y < out_h; ++y) {
        const auto* row = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out_w; ++x) {
            const cv::Vec3b& bgr = row[x];
            out[static_cast<size_t>(y) * out_w + x] =
                0xff000000u |
                (static_cast<uint32_t>(bgr[2]) << 16) |
                (static_cast<uint32_t>(bgr[1]) << 8) |
                static_cast<uint32_t>(bgr[0]);
        }
    }
    return out;
}

void print_timing(const char* name, const std::vector<double>& values) {
    double sum = 0.0;
    double min_v = std::numeric_limits<double>::max();
    double max_v = 0.0;
    for (double v : values) {
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    std::cout << std::fixed << std::setprecision(4)
              << name << " avg=" << (sum / values.size())
              << " min=" << min_v
              << " max=" << max_v << "\n";
}

void save_xrgb_png(const std::string& path, const std::vector<uint32_t>& xrgb, int width, int height) {
    if (path.empty() || xrgb.empty() || width <= 0 || height <= 0) {
        return;
    }
    cv::Mat bgra(height, width, CV_8UC4);
    for (int y = 0; y < height; ++y) {
        auto* row = bgra.ptr<cv::Vec4b>(y);
        for (int x = 0; x < width; ++x) {
            uint32_t p = xrgb[static_cast<size_t>(y) * width + x];
            row[x] = cv::Vec4b(
                static_cast<uint8_t>(p & 0xff),
                static_cast<uint8_t>((p >> 8) & 0xff),
                static_cast<uint8_t>((p >> 16) & 0xff),
                255);
        }
    }
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    if (!cv::imwrite(path, bgr)) {
        throw std::runtime_error("failed to save png: " + path);
    }
    std::cout << "[SAVE] " << path << " " << width << "x" << height << "\n";
}

double resolve_scale(const cv::Mat& image, double requested_scale, int model_w, int model_h) {
    if (requested_scale > 0.0) {
        return requested_scale;
    }
    return std::min(static_cast<double>(model_w) / image.cols,
                    static_cast<double>(model_h) / image.rows);
}

cv::Mat scale_and_pad_bgr(const cv::Mat& image, double scale, int model_w, int model_h) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    check(resized.cols > model_w || resized.rows > model_h,
          "scaled KITTI image is larger than LAS2 input shape; reduce --input-scale");
    const int pad_w = model_w - resized.cols;
    const int pad_h = model_h - resized.rows;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       pad_h / 2, pad_h - pad_h / 2,
                       pad_w / 2, pad_w - pad_w / 2,
                       cv::BORDER_REPLICATE);
    return padded;
}

void bgr_to_las2_tensor(const cv::Mat& bgr, std::vector<float>& tensor, bool nhwc) {
    check(bgr.empty() || bgr.type() != CV_8UC3, "bgr_to_las2_tensor expects CV_8UC3");
    const int plane = bgr.cols * bgr.rows;
    tensor.assign(static_cast<size_t>(plane) * 3, 0.0f);
    for (int y = 0; y < bgr.rows; ++y) {
        const auto* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const int idx = y * bgr.cols + x;
            const cv::Vec3b& p = row[x];
            if (nhwc) {
                tensor[static_cast<size_t>(idx) * 3 + 0] = static_cast<float>(p[2]);
                tensor[static_cast<size_t>(idx) * 3 + 1] = static_cast<float>(p[1]);
                tensor[static_cast<size_t>(idx) * 3 + 2] = static_cast<float>(p[0]);
            } else {
                tensor[idx] = static_cast<float>(p[2]);
                tensor[idx + plane] = static_cast<float>(p[1]);
                tensor[idx + plane * 2] = static_cast<float>(p[0]);
            }
        }
    }
}

std::vector<uint32_t> bgr_to_xrgb(const cv::Mat& bgr) {
    check(bgr.empty() || bgr.type() != CV_8UC3, "bgr_to_xrgb expects CV_8UC3");
    std::vector<uint32_t> out(static_cast<size_t>(bgr.cols) * bgr.rows);
    for (int y = 0; y < bgr.rows; ++y) {
        const auto* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b& p = row[x];
            out[static_cast<size_t>(y) * bgr.cols + x] =
                0xff000000u |
                (static_cast<uint32_t>(p[2]) << 16) |
                (static_cast<uint32_t>(p[1]) << 8) |
                static_cast<uint32_t>(p[0]);
        }
    }
    return out;
}

std::vector<uint32_t> resize_bgr_to_xrgb(const cv::Mat& bgr, int out_w, int out_h) {
    check(bgr.empty() || bgr.type() != CV_8UC3, "resize_bgr_to_xrgb expects CV_8UC3");
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(out_w, out_h), 0.0, 0.0, cv::INTER_AREA);
    return bgr_to_xrgb(resized);
}

std::vector<uint32_t> resize_xrgb_nearest(const std::vector<uint32_t>& src,
                                          int src_w,
                                          int src_h,
                                          int dst_w,
                                          int dst_h) {
    std::vector<uint32_t> out(static_cast<size_t>(dst_w) * dst_h, 0xff000000u);
    for (int y = 0; y < dst_h; ++y) {
        const int sy = std::min(src_h - 1, y * src_h / dst_h);
        const uint32_t* src_row = src.data() + static_cast<size_t>(sy) * src_w;
        uint32_t* dst_row = out.data() + static_cast<size_t>(y) * dst_w;
        for (int x = 0; x < dst_w; ++x) {
            const int sx = std::min(src_w - 1, x * src_w / dst_w);
            dst_row[x] = src_row[sx];
        }
    }
    return out;
}

float percentile(std::vector<float> values, double q) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t idx = std::min(values.size() - 1,
                                static_cast<size_t>(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

struct FloatRange {
    float lo = 0.0f;
    float hi = 1.0f;
    bool valid = false;
};

FloatRange estimate_disp_range(const std::vector<float>& disp,
                               int out_w,
                               int out_h,
                               int top,
                               int left,
                               int disp_w) {
    std::vector<float> valid;
    valid.reserve(static_cast<size_t>(out_w) * out_h);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const float v = disp[static_cast<size_t>(y + top) * disp_w + (x + left)];
            if (std::isfinite(v)) {
                valid.push_back(v);
            }
        }
    }
    if (valid.empty()) {
        return {};
    }
    float lo = percentile(valid, 0.01);
    float hi = percentile(valid, 0.99);
    if (hi <= lo) {
        hi = lo + 1.0f;
    }
    return {lo, hi, true};
}

std::vector<uint32_t> disparity_to_xrgb_range(const std::vector<float>& disp,
                                              int out_w,
                                              int out_h,
                                              int top,
                                              int left,
                                              int disp_w,
                                              float lo,
                                              float hi) {
    std::vector<uint32_t> out(static_cast<size_t>(out_w) * out_h, 0xff000000u);
    if (hi <= lo) {
        hi = lo + 1.0f;
    }
    cv::Mat gray(out_h, out_w, CV_8UC1);
    for (int y = 0; y < out_h; ++y) {
        auto* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < out_w; ++x) {
            const float v = disp[static_cast<size_t>(y + top) * disp_w + (x + left)];
            const float normalized = std::isfinite(v) ? std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
            row[x] = static_cast<uint8_t>(normalized * 255.0f + 0.5f);
        }
    }
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    for (int y = 0; y < out_h; ++y) {
        const auto* row = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out_w; ++x) {
            const cv::Vec3b& bgr = row[x];
            out[static_cast<size_t>(y) * out_w + x] =
                0xff000000u |
                (static_cast<uint32_t>(bgr[2]) << 16) |
                (static_cast<uint32_t>(bgr[1]) << 8) |
                static_cast<uint32_t>(bgr[0]);
        }
    }
    return out;
}

int run_preprocess_bench(const Options& opt) {
    V4l2IrCapture cap(opt);
    V4l2IrCapture::CapturedFrame frame;
    while (!cap.acquire(frame)) {
    }

    BufferAllocator* allocator = CreateDmabufHeapBufferAllocator();
    check(!allocator, "CreateDmabufHeapBufferAllocator failed");

    cl_mem cl_input = nullptr;
    cl_mem cl_output = nullptr;
    try {
        const size_t tensor_bytes = static_cast<size_t>(kModelW) * kModelH * 6 * sizeof(float);
        DmaBuffer tensor(allocator, tensor_bytes);
        OpenClPreprocessor cl;
        cl_input = cl.import_dmabuf(frame.dma_fd, cap.buffer_size());
        cl_output = cl.import_dmabuf(tensor.fd(), tensor.size());

        std::vector<double> cpu_ms;
        std::vector<double> cl_ms;
        cpu_ms.reserve(static_cast<size_t>(opt.loops));
        cl_ms.reserve(static_cast<size_t>(opt.loops));
        std::vector<float> cpu;

        for (int i = 0; i < opt.loops; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            cpu_preprocess_uyvy(frame.data, cap.width(), cap.height(), cap.stride(), cpu);
            auto t1 = std::chrono::steady_clock::now();
            cpu_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        for (int i = 0; i < opt.loops; ++i) {
            cl_ms.push_back(cl.run(cl_input, cl_output, cap.width(), cap.height(), cap.stride()));
        }

        tensor.sync_start(kSyncRead);
        const auto* gpu = static_cast<const float*>(tensor.data());
        double max_abs = 0.0;
        double sum_abs = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            double diff = std::abs(static_cast<double>(gpu[i]) - static_cast<double>(cpu[i]));
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
            if (diff != 0.0) {
                ++bad;
            }
        }
        tensor.sync_end(kSyncRead);

        std::cout << "[BENCH] input_dmabuf_fd=" << frame.dma_fd
                  << " output_dmabuf_fd=" << tensor.fd()
                  << " tensor_bytes=" << tensor_bytes
                  << " elements=" << cpu.size() << "\n";
        print_timing("cpu_ms", cpu_ms);
        print_timing("opencl_kernel_ms", cl_ms);
        std::cout << std::setprecision(8)
                  << "verify max_abs=" << max_abs
                  << " mean_abs=" << (sum_abs / cpu.size())
                  << " bad_ne_0=" << bad << "/" << cpu.size() << "\n";

        if (cl_input) clReleaseMemObject(cl_input);
        if (cl_output) clReleaseMemObject(cl_output);
        cap.release(frame);
        FreeDmabufHeapBufferAllocator(allocator);
        return bad == 0 ? 0 : 2;
    } catch (...) {
        if (cl_input) clReleaseMemObject(cl_input);
        if (cl_output) clReleaseMemObject(cl_output);
        cap.release(frame);
        FreeDmabufHeapBufferAllocator(allocator);
        throw;
    }
}

class DrmFramebuffer {
public:
    explicit DrmFramebuffer(const std::string& card) {
        fd_ = open(card.c_str(), O_RDWR | O_CLOEXEC);
        check(fd_ < 0, errno_msg(card));
        discover();
        buffers_[0] = create_dumb();
        buffers_[1] = create_dumb();
        draw_index_ = 1;
        scan_index_ = 0;
        clear_buffer(buffers_[scan_index_]);
        check(drmModeSetCrtc(fd_, crtc_id_, buffers_[scan_index_].fb_id, 0, 0, &connector_id_, 1, &mode_) != 0,
              errno_msg("drmModeSetCrtc"));
        std::printf("[DRM] card=%s connector=%u crtc=%u mode=%dx%d fb=%u\n",
                    card.c_str(), connector_id_, crtc_id_, width_, height_, buffers_[scan_index_].fb_id);
    }

    ~DrmFramebuffer() {
        std::puts("[CLEANUP] DRM restore crtc");
        if (old_crtc_) {
            drmModeSetCrtc(fd_, old_crtc_->crtc_id, old_crtc_->buffer_id, old_crtc_->x, old_crtc_->y,
                           &connector_id_, 1, &old_crtc_->mode);
            drmModeFreeCrtc(old_crtc_);
        }
        std::puts("[CLEANUP] DRM destroy buffers");
        destroy_dumb(buffers_[0]);
        destroy_dumb(buffers_[1]);
        std::puts("[CLEANUP] DRM close fd");
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void begin_frame(bool clear = true) {
        if (clear) {
            clear_buffer(buffers_[draw_index_]);
        }
    }

    void present() {
        int waiting = 1;
        check(drmModePageFlip(fd_, crtc_id_, buffers_[draw_index_].fb_id, DRM_MODE_PAGE_FLIP_EVENT, &waiting) != 0,
              errno_msg("drmModePageFlip"));
        drmEventContext ev = {};
        ev.version = DRM_EVENT_CONTEXT_VERSION;
        ev.page_flip_handler = [](int, unsigned int, unsigned int, unsigned int, void* data) {
            *static_cast<int*>(data) = 0;
        };
        while (waiting) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);
            int ret = select(fd_ + 1, &fds, nullptr, nullptr, nullptr);
            if (ret < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(errno_msg("select drm"));
            }
            drmHandleEvent(fd_, &ev);
        }
        std::swap(draw_index_, scan_index_);
    }

    std::vector<uint32_t> snapshot() const {
        std::vector<uint32_t> out(static_cast<size_t>(width_) * height_);
        const DumbBuffer& buffer = buffers_[scan_index_];
        for (int y = 0; y < height_; ++y) {
            const uint32_t* row = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint8_t*>(buffer.map) + static_cast<size_t>(y) * buffer.pitch);
            std::copy(row, row + width_, out.begin() + static_cast<size_t>(y) * width_);
        }
        return out;
    }

    void clear() {
        clear_buffer(buffers_[draw_index_]);
    }

    void draw_gray(const std::vector<uint8_t>& gray, int src_w, int src_h, const Rect& dst) {
        if (gray.empty() || dst.w <= 0 || dst.h <= 0) {
            return;
        }
        for (int y = 0; y < dst.h; ++y) {
            int sy = std::min(src_h - 1, y * src_h / dst.h);
            uint32_t* out = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(buffers_[draw_index_].map) +
                static_cast<size_t>(dst.y + y) * buffers_[draw_index_].pitch) + dst.x;
            for (int x = 0; x < dst.w; ++x) {
                int sx = std::min(src_w - 1, x * src_w / dst.w);
                uint8_t v = gray[static_cast<size_t>(sy) * src_w + sx];
                out[x] = 0xff000000u | (static_cast<uint32_t>(v) << 16) |
                         (static_cast<uint32_t>(v) << 8) | static_cast<uint32_t>(v);
            }
        }
    }

    void draw_xrgb(const std::vector<uint32_t>& pixels, int src_w, int src_h, const Rect& dst) {
        if (pixels.empty() || dst.w <= 0 || dst.h <= 0) {
            return;
        }
        for (int y = 0; y < dst.h; ++y) {
            int sy = std::min(src_h - 1, y * src_h / dst.h);
            uint32_t* out = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(buffers_[draw_index_].map) +
                static_cast<size_t>(dst.y + y) * buffers_[draw_index_].pitch) + dst.x;
            for (int x = 0; x < dst.w; ++x) {
                int sx = std::min(src_w - 1, x * src_w / dst.w);
                out[x] = pixels[static_cast<size_t>(sy) * src_w + sx];
            }
        }
    }

    void blit_xrgb(const std::vector<uint32_t>& pixels, int src_w, int src_h, const Rect& dst) {
        if (pixels.empty() || dst.w <= 0 || dst.h <= 0 || src_w != dst.w || src_h != dst.h) {
            draw_xrgb(pixels, src_w, src_h, dst);
            return;
        }
        for (int y = 0; y < dst.h; ++y) {
            uint32_t* out = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(buffers_[draw_index_].map) +
                static_cast<size_t>(dst.y + y) * buffers_[draw_index_].pitch) + dst.x;
            const uint32_t* in = pixels.data() + static_cast<size_t>(y) * src_w;
            std::copy(in, in + src_w, out);
        }
    }

    void draw_text_overlay(const std::vector<std::string>& lines) {
        draw_text_overlay_at(lines, 8, 8);
    }

    void draw_text_overlay_at(const std::vector<std::string>& lines, int x, int y) {
        if (lines.empty()) {
            return;
        }
        DumbBuffer& buffer = buffers_[draw_index_];
        cv::Mat bgra(height_, width_, CV_8UC4, buffer.map, buffer.pitch);
        const int line_h = 24;
        const int box_w = 360;
        const int box_h = 12 + static_cast<int>(lines.size()) * line_h;
        x = std::clamp(x, 0, std::max(0, width_ - box_w));
        y = std::clamp(y, 0, std::max(0, height_ - box_h));
        cv::rectangle(bgra, cv::Rect(x, y, box_w, box_h), cv::Scalar(0, 0, 0, 210), cv::FILLED);
        for (size_t i = 0; i < lines.size(); ++i) {
            cv::putText(bgra, lines[i], cv::Point(x + 10, y + 24 + static_cast<int>(i) * line_h),
                        cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255, 255), 1, cv::LINE_AA);
        }
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct DumbBuffer {
        uint32_t handle = 0;
        uint32_t pitch = 0;
        uint64_t size = 0;
        uint32_t fb_id = 0;
        uint32_t* map = nullptr;
    };

    void discover() {
        drmModeRes* res = drmModeGetResources(fd_);
        check(!res, errno_msg("drmModeGetResources"));
        for (int i = 0; i < res->count_connectors; ++i) {
            drmModeConnector* conn = drmModeGetConnector(fd_, res->connectors[i]);
            if (conn && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
                connector_id_ = conn->connector_id;
                mode_ = conn->modes[0];
                width_ = mode_.hdisplay;
                height_ = mode_.vdisplay;
                drmModeEncoder* enc = conn->encoder_id ? drmModeGetEncoder(fd_, conn->encoder_id) : nullptr;
                if (!enc) {
                    for (int j = 0; j < conn->count_encoders && !enc; ++j) {
                        enc = drmModeGetEncoder(fd_, conn->encoders[j]);
                    }
                }
                if (enc) {
                    crtc_id_ = enc->crtc_id;
                    if (!crtc_id_) {
                        for (int c = 0; c < res->count_crtcs; ++c) {
                            if (enc->possible_crtcs & (1 << c)) {
                                crtc_id_ = res->crtcs[c];
                                break;
                            }
                        }
                    }
                    drmModeFreeEncoder(enc);
                }
                drmModeFreeConnector(conn);
                break;
            }
            if (conn) {
                drmModeFreeConnector(conn);
            }
        }
        drmModeFreeResources(res);
        check(!connector_id_ || !crtc_id_, "no connected DRM output");
        old_crtc_ = drmModeGetCrtc(fd_, crtc_id_);
    }

    DumbBuffer create_dumb() {
        DumbBuffer out;
        drm_mode_create_dumb create = {};
        create.width = width_;
        create.height = height_;
        create.bpp = 32;
        check(drmIoctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0, errno_msg("CREATE_DUMB"));
        out.handle = create.handle;
        out.pitch = create.pitch;
        out.size = create.size;

        uint32_t handles[4] = {out.handle, 0, 0, 0};
        uint32_t pitches[4] = {out.pitch, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        check(drmModeAddFB2(fd_, width_, height_, DRM_FORMAT_XRGB8888, handles, pitches, offsets,
                            &out.fb_id, 0) != 0,
              errno_msg("drmModeAddFB2"));

        drm_mode_map_dumb map = {};
        map.handle = out.handle;
        check(drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0, errno_msg("MAP_DUMB"));
        out.map = static_cast<uint32_t*>(mmap(nullptr, out.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, map.offset));
        check(out.map == MAP_FAILED, errno_msg("mmap dumb"));
        return out;
    }

    void clear_buffer(const DumbBuffer& buffer) {
        std::fill(buffer.map,
                  buffer.map + static_cast<size_t>(buffer.pitch) * height_ / sizeof(uint32_t),
                  0xff000000u);
    }

    void destroy_dumb(DumbBuffer& buffer) {
        if (buffer.map) {
            munmap(buffer.map, buffer.size);
            buffer.map = nullptr;
        }
        if (buffer.fb_id) {
            drmModeRmFB(fd_, buffer.fb_id);
            buffer.fb_id = 0;
        }
        if (buffer.handle) {
            drm_mode_destroy_dumb destroy = {};
            destroy.handle = buffer.handle;
            drmIoctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
            buffer.handle = 0;
        }
    }

    int fd_ = -1;
    uint32_t connector_id_ = 0;
    uint32_t crtc_id_ = 0;
    drmModeCrtc* old_crtc_ = nullptr;
    drmModeModeInfo mode_ = {};
    int width_ = 0;
    int height_ = 0;
    DumbBuffer buffers_[2];
    int draw_index_ = 0;
    int scan_index_ = 0;
};

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        std::string item = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!item.empty()) {
            out.push_back(item);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

struct CachedKittiFrame {
    std::vector<float> left_tensor;
    std::vector<float> right_tensor;
    std::vector<uint32_t> left_display_xrgb;
    std::vector<uint32_t> right_display_xrgb;
    Rect left_rect;
    Rect right_rect;
    Rect depth_rect;
    int image_w = 0;
    int image_h = 0;
    int depth_w = 0;
    int depth_h = 0;
    int pad_left = 0;
    int pad_top = 0;
    double decode_ms = 0.0;
    double resize_ms = 0.0;
    double tensor_ms = 0.0;
};

struct PipelineSlot {
    std::unique_ptr<Las2Engine> engine;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv_job;
    std::condition_variable cv_done;
    const CachedKittiFrame* frame = nullptr;
    int frame_id = -1;
    bool has_job = false;
    bool finished = true;
    bool quit = false;
    std::vector<float> disp_buffers[2];
    int write_disp = 0;
    int ready_disp = 0;
    double infer_ms = 0.0;
};

void pipeline_worker(PipelineSlot* slot) {
    while (true) {
        const CachedKittiFrame* frame = nullptr;
        {
            std::unique_lock<std::mutex> lock(slot->mutex);
            slot->cv_job.wait(lock, [&] { return slot->has_job || slot->quit; });
            if (slot->quit) {
                break;
            }
            frame = slot->frame;
        }

        const auto t0 = std::chrono::steady_clock::now();
        slot->engine->infer(const_cast<float*>(frame->left_tensor.data()),
                            const_cast<float*>(frame->right_tensor.data()),
                            slot->disp_buffers[slot->write_disp]);
        const auto t1 = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            slot->ready_disp = slot->write_disp;
            slot->write_disp = 1 - slot->write_disp;
            slot->has_job = false;
            slot->finished = true;
        }
        slot->cv_done.notify_one();
    }
}

void dispatch_pipeline_job(PipelineSlot& slot, const CachedKittiFrame& frame, int frame_id) {
    {
        std::lock_guard<std::mutex> lock(slot.mutex);
        slot.frame = &frame;
        slot.frame_id = frame_id;
        slot.finished = false;
        slot.has_job = true;
    }
    slot.cv_job.notify_one();
}

struct CompletedFrame {
    const CachedKittiFrame* frame = nullptr;
    std::vector<float> disp;
    int infer_frame = 0;
    double infer_ms = 0.0;
};

struct RealtimeResultPool {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<CompletedFrame> frames;
    size_t max_frames = 16;
    int active_workers = 0;
    bool done = false;
    uint64_t produced = 0;
    uint64_t dropped = 0;
};

struct RealtimeWorkerState {
    std::unique_ptr<Las2Engine> engine;
    std::thread worker;
};

void realtime_worker_loop(RealtimeWorkerState* worker,
                          const std::vector<CachedKittiFrame>* cached,
                          RealtimeResultPool* pool,
                          std::atomic<int>* next_infer_frame,
                          int target_frames) {
    while (g_running) {
        const int infer_frame = next_infer_frame->fetch_add(1);
        if (target_frames > 0 && infer_frame >= target_frames) {
            break;
        }
        const CachedKittiFrame& frame = (*cached)[static_cast<size_t>(infer_frame) % cached->size()];
        CompletedFrame completed;
        completed.frame = &frame;
        completed.infer_frame = infer_frame;
        const auto t0 = std::chrono::steady_clock::now();
        worker->engine->infer(const_cast<float*>(frame.left_tensor.data()),
                              const_cast<float*>(frame.right_tensor.data()),
                              completed.disp);
        const auto t1 = std::chrono::steady_clock::now();
        completed.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        {
            std::lock_guard<std::mutex> lock(pool->mutex);
            if (pool->frames.size() >= pool->max_frames) {
                pool->frames.pop_front();
                ++pool->dropped;
            }
            pool->frames.push_back(std::move(completed));
            ++pool->produced;
        }
        pool->cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        if (pool->active_workers > 0) {
            --pool->active_workers;
        }
        if (pool->active_workers == 0) {
            pool->done = true;
        }
    }
    pool->cv.notify_all();
}

int run_kitti_offline_pipeline(const Options& opt, const std::vector<KittiPair>& pairs) {
    DrmFramebuffer drm(opt.card);
    const auto affinities = split_csv(opt.pipeline_affinities);
    check(static_cast<int>(affinities.size()) < opt.pipeline_instances,
          "--pipeline-affinities must provide at least --pipeline-instances entries");

    std::vector<std::unique_ptr<PipelineSlot>> slots;
    slots.reserve(static_cast<size_t>(opt.pipeline_instances));
    for (int i = 0; i < opt.pipeline_instances; ++i) {
        auto slot = std::make_unique<PipelineSlot>();
        slot->engine = std::make_unique<Las2Engine>(opt.model, affinities[i], opt.pipeline_threads);
        slots.push_back(std::move(slot));
    }

    const int model_w = slots[0]->engine->input_width();
    const int model_h = slots[0]->engine->input_height();
    const int disp_w = slots[0]->engine->output_width();
    const int disp_h = slots[0]->engine->output_height();
    const bool input_is_nhwc = slots[0]->engine->input_is_nhwc();

    std::vector<CachedKittiFrame> cached;
    cached.reserve(pairs.size());
    double total_decode_ms = 0.0;
    double total_resize_ms = 0.0;
    double total_tensor_ms = 0.0;
    std::printf("[OFFLINE] decode/preprocess pairs=%zu model=%dx%d\n", pairs.size(), model_w, model_h);
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat left_bgr = cv::imread(pairs[i].left, cv::IMREAD_COLOR);
        cv::Mat right_bgr = cv::imread(pairs[i].right, cv::IMREAD_COLOR);
        const auto t1 = std::chrono::steady_clock::now();
        check(left_bgr.empty(), "failed to read KITTI left image: " + pairs[i].left);
        check(right_bgr.empty(), "failed to read KITTI right image: " + pairs[i].right);
        check(left_bgr.size() != right_bgr.size(), "KITTI left/right image size mismatch");

        const double scale = resolve_scale(left_bgr, opt.input_scale, model_w, model_h);
        cv::Mat left_padded = scale_and_pad_bgr(left_bgr, scale, model_w, model_h);
        cv::Mat right_padded = scale_and_pad_bgr(right_bgr, scale, model_w, model_h);
        const auto t2 = std::chrono::steady_clock::now();

        CachedKittiFrame frame;
        frame.image_w = left_bgr.cols;
        frame.image_h = left_bgr.rows;
        const int cell_w = drm.width() / 3;
        const int cell_h = drm.height() / opt.display_streams;
        frame.left_rect = compute_fit_rect(0, 0, cell_w, cell_h, frame.image_w, frame.image_h);
        frame.right_rect = compute_fit_rect(cell_w, 0, cell_w, cell_h, frame.image_w, frame.image_h);
        frame.left_display_xrgb = resize_bgr_to_xrgb(left_bgr, frame.left_rect.w, frame.left_rect.h);
        frame.right_display_xrgb = resize_bgr_to_xrgb(right_bgr, frame.right_rect.w, frame.right_rect.h);
        bgr_to_las2_tensor(left_padded, frame.left_tensor, input_is_nhwc);
        bgr_to_las2_tensor(right_padded, frame.right_tensor, input_is_nhwc);
        const int scaled_w = std::max(1, static_cast<int>(std::round(left_bgr.cols * scale)));
        const int scaled_h = std::max(1, static_cast<int>(std::round(left_bgr.rows * scale)));
        frame.depth_w = scaled_w;
        frame.depth_h = scaled_h;
        frame.pad_left = (disp_w - scaled_w) / 2;
        frame.pad_top = (disp_h - scaled_h) / 2;
        frame.depth_rect = compute_fit_rect(cell_w * 2, 0, cell_w, cell_h, frame.depth_w, frame.depth_h);
        const auto t3 = std::chrono::steady_clock::now();

        frame.decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame.resize_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        frame.tensor_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        total_decode_ms += frame.decode_ms;
        total_resize_ms += frame.resize_ms;
        total_tensor_ms += frame.tensor_ms;
        cached.push_back(std::move(frame));
        if ((i + 1) % 50 == 0 || i + 1 == pairs.size()) {
            std::printf("[OFFLINE] cached=%zu/%zu\n", i + 1, pairs.size());
        }
    }
    std::printf("[OFFLINE] avg_decode=%.3f avg_resize=%.3f avg_tensor_xrgb=%.3f\n",
                total_decode_ms / cached.size(),
                total_resize_ms / cached.size(),
                total_tensor_ms / cached.size());

    std::puts("[PIPELINE] warmup sequential per instance");
    for (int i = 0; i < opt.pipeline_instances; ++i) {
        std::vector<float> warmup_disp;
        const auto& frame = cached[static_cast<size_t>(i) % cached.size()];
        const auto t0 = std::chrono::steady_clock::now();
        slots[i]->engine->infer(const_cast<float*>(frame.left_tensor.data()),
                                const_cast<float*>(frame.right_tensor.data()),
                                warmup_disp);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("[WARMUP] instance=%d infer_ms=%.3f\n", i,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    int target_frames = opt.frames;
    if (target_frames > 0 && target_frames < opt.save_warmup_frames) {
        target_frames = opt.save_warmup_frames;
    }

    if (opt.display_streams > 1) {
        const int streams = opt.display_streams;
        check(opt.pipeline_instances < streams,
              "--pipeline-instances must be >= --display-streams");
        check((opt.pipeline_instances % streams) != 0,
              "--pipeline-instances must be divisible by --display-streams");
        const int instances_per_stream = opt.pipeline_instances / streams;
        const int cell_h = drm.height() / streams;
        std::printf("[PIPELINE] ordered multi-stream streams=%d instances_per_stream=%d display_fps=%.3f\n",
                    streams, instances_per_stream, opt.playback_fps);

        for (int i = 0; i < opt.pipeline_instances; ++i) {
            slots[i]->worker = std::thread(pipeline_worker, slots[i].get());
        }

        std::vector<int> stream_offsets(streams, 0);
        std::vector<int> stream_next(streams, 0);
        std::vector<int> stream_read_index(streams, 0);
        for (int s = 0; s < streams; ++s) {
            stream_offsets[s] = static_cast<int>((static_cast<size_t>(s) * cached.size()) / streams);
            stream_next[s] = 0;
            for (int k = 0; k < instances_per_stream; ++k) {
                const int slot_index = s * instances_per_stream + k;
                const int source = stream_offsets[s] + stream_next[s];
                dispatch_pipeline_job(*slots[static_cast<size_t>(slot_index)],
                                      cached[static_cast<size_t>(source) % cached.size()],
                                      source);
                ++stream_next[s];
            }
        }

        std::vector<FloatRange> vis_ranges(streams);
        const bool fixed_vis_range = std::isfinite(opt.disp_vis_min) && std::isfinite(opt.disp_vis_max);
        if (fixed_vis_range) {
            for (auto& range : vis_ranges) {
                range = {
                    static_cast<float>(opt.disp_vis_min),
                    static_cast<float>(opt.disp_vis_max),
                    true,
                };
            }
        }

        std::vector<uint32_t> saved_depth_xrgb;
        std::vector<uint32_t> saved_view_xrgb;
        int saved_depth_w = 0;
        int saved_depth_h = 0;
        int saved_frame = 0;
        int frame_count = 0;
        const auto frame_period = std::chrono::nanoseconds(
            static_cast<int64_t>(1000000000.0 / opt.playback_fps));
        const auto pipeline_start = std::chrono::steady_clock::now();

        while (g_running && (target_frames == 0 || frame_count < target_frames)) {
            const auto wait_start = std::chrono::steady_clock::now();
            std::vector<double> infer_ms(static_cast<size_t>(streams), 0.0);
            std::vector<int> infer_frames(static_cast<size_t>(streams), 0);
            std::vector<const CachedKittiFrame*> stream_frames(static_cast<size_t>(streams), nullptr);
            std::vector<std::vector<float>> stream_disps(static_cast<size_t>(streams));
            std::vector<std::vector<uint32_t>> stream_depths(static_cast<size_t>(streams));

            for (int s = 0; s < streams; ++s) {
                const int slot_index = s * instances_per_stream + stream_read_index[s];
                PipelineSlot& slot = *slots[static_cast<size_t>(slot_index)];
                {
                    std::unique_lock<std::mutex> lock(slot.mutex);
                    slot.cv_done.wait(lock, [&] { return slot.finished || !g_running; });
                }
                if (!g_running) {
                    break;
                }
                infer_ms[static_cast<size_t>(s)] = slot.infer_ms;
                infer_frames[static_cast<size_t>(s)] = slot.frame_id;
                const CachedKittiFrame& frame = *slot.frame;
                stream_frames[static_cast<size_t>(s)] = &frame;
                stream_disps[static_cast<size_t>(s)] = slot.disp_buffers[slot.ready_disp];

                if (!fixed_vis_range) {
                    const FloatRange frame_range = estimate_disp_range(stream_disps[static_cast<size_t>(s)],
                                                                       frame.depth_w, frame.depth_h,
                                                                       frame.pad_top, frame.pad_left, disp_w);
                    if (frame_range.valid) {
                        FloatRange& range = vis_ranges[static_cast<size_t>(s)];
                        if (!range.valid) {
                            range = frame_range;
                        } else {
                            const float alpha = static_cast<float>(opt.disp_vis_alpha);
                            range.lo = range.lo * (1.0f - alpha) + frame_range.lo * alpha;
                            range.hi = range.hi * (1.0f - alpha) + frame_range.hi * alpha;
                        }
                    }
                }
                const FloatRange& range = vis_ranges[static_cast<size_t>(s)];
                if (range.valid) {
                    stream_depths[static_cast<size_t>(s)] = disparity_to_xrgb_range(stream_disps[static_cast<size_t>(s)],
                        frame.depth_w, frame.depth_h, frame.pad_top, frame.pad_left, disp_w,
                        range.lo, range.hi);
                } else {
                    stream_depths[static_cast<size_t>(s)].assign(
                        static_cast<size_t>(frame.depth_w) * frame.depth_h, 0xff000000u);
                }
            }
            if (!g_running) {
                break;
            }
            const auto ready_time = std::chrono::steady_clock::now();
            const double wait_ms = std::chrono::duration<double, std::milli>(ready_time - wait_start).count();

            const auto display_start = wait_start;
            drm.begin_frame();
            for (int s = 0; s < streams; ++s) {
                const CachedKittiFrame& frame = *stream_frames[static_cast<size_t>(s)];
                const int row_y = s * cell_h;
                Rect left_rect = frame.left_rect;
                Rect right_rect = frame.right_rect;
                Rect depth_rect = frame.depth_rect;
                left_rect.y += row_y;
                right_rect.y += row_y;
                depth_rect.y += row_y;
                const std::vector<uint32_t> depth_display_xrgb = resize_xrgb_nearest(
                    stream_depths[static_cast<size_t>(s)],
                    frame.depth_w, frame.depth_h, depth_rect.w, depth_rect.h);
                drm.blit_xrgb(frame.left_display_xrgb, frame.left_rect.w, frame.left_rect.h, left_rect);
                drm.blit_xrgb(frame.right_display_xrgb, frame.right_rect.w, frame.right_rect.h, right_rect);
                drm.blit_xrgb(depth_display_xrgb, depth_rect.w, depth_rect.h, depth_rect);
            }
            double total_fps = 0.0;
            std::vector<std::string> overlay_lines;
            char line[160];
            for (int s = 0; s < streams; ++s) {
                double stream_fps = 0.0;
                for (int k = 0; k < instances_per_stream; ++k) {
                    const int slot_index = s * instances_per_stream + k;
                    const double slot_ms = slots[static_cast<size_t>(slot_index)]->infer_ms;
                    if (slot_ms > 0.0) {
                        stream_fps += 1000.0 / slot_ms;
                    }
                }
                total_fps += stream_fps;
                std::snprintf(line, sizeof(line), "S%d infer: %.2f FPS (%d inst)",
                              s, stream_fps, instances_per_stream);
                overlay_lines.emplace_back(line);
            }
            std::snprintf(line, sizeof(line), "Total infer FPS: %.2f (%d inst)",
                          total_fps, opt.pipeline_instances);
            overlay_lines.insert(overlay_lines.begin(), line);
            drm.draw_text_overlay(overlay_lines);
            drm.present();
            const auto present_end = std::chrono::steady_clock::now();
            double throttle_ms = 0.0;
            const auto active_elapsed = present_end - display_start;
            if (active_elapsed < frame_period) {
                const auto sleep_for = frame_period - active_elapsed;
                throttle_ms = std::chrono::duration<double, std::milli>(sleep_for).count();
                std::this_thread::sleep_for(sleep_for);
            }
            const auto display_end = std::chrono::steady_clock::now();

            for (int s = 0; s < streams; ++s) {
                if (target_frames == 0 || frame_count + 1 < target_frames) {
                    const int slot_index = s * instances_per_stream + stream_read_index[s];
                    const int source = stream_offsets[s] + stream_next[s];
                    dispatch_pipeline_job(*slots[static_cast<size_t>(slot_index)],
                                          cached[static_cast<size_t>(source) % cached.size()],
                                          source);
                    ++stream_next[s];
                    stream_read_index[s] = (stream_read_index[s] + 1) % instances_per_stream;
                }
            }

            ++frame_count;
            if (frame_count >= opt.save_warmup_frames) {
                saved_view_xrgb.clear();
                saved_depth_xrgb = stream_depths[0];
                saved_depth_w = stream_frames[0]->depth_w;
                saved_depth_h = stream_frames[0]->depth_h;
                saved_frame = frame_count;
            }
            if (frame_count % 30 == 0) {
                const double elapsed = std::chrono::duration<double>(display_end - pipeline_start).count();
                const double fps = elapsed > 0.0 ? frame_count / elapsed : 0.0;
                const double avg_infer_ms = std::accumulate(infer_ms.begin(), infer_ms.end(), 0.0) / streams;
                const double display_ms = std::chrono::duration<double, std::milli>(present_end - ready_time).count();
                std::printf("[MULTI4_DISPLAY] frame=%d fps=%.2f total_infer_fps=%.2f avg_displayed_infer_ms=%.3f wait_ms=%.3f display_ms=%.3f throttle_ms=%.3f sources=%d,%d,%d,%d\n",
                            frame_count, fps, total_fps, avg_infer_ms, wait_ms, display_ms, throttle_ms,
                            infer_frames.size() > 0 ? infer_frames[0] : -1,
                            infer_frames.size() > 1 ? infer_frames[1] : -1,
                            infer_frames.size() > 2 ? infer_frames[2] : -1,
                            infer_frames.size() > 3 ? infer_frames[3] : -1);
            }
        }

        for (int s = 0; s < opt.pipeline_instances; ++s) {
            {
                std::lock_guard<std::mutex> lock(slots[s]->mutex);
                slots[s]->quit = true;
            }
            slots[s]->cv_job.notify_one();
        }
        for (int s = 0; s < opt.pipeline_instances; ++s) {
            if (slots[s]->worker.joinable()) {
                slots[s]->worker.join();
            }
        }

        if (frame_count > 0) {
            if (saved_view_xrgb.empty()) {
                saved_view_xrgb = drm.snapshot();
                saved_frame = frame_count;
            }
            save_xrgb_png(opt.save_depth, saved_depth_xrgb, saved_depth_w, saved_depth_h);
            save_xrgb_png(opt.save_view, saved_view_xrgb, drm.width(), drm.height());
            std::printf("[SAVE] saved frame=%d\n", saved_frame);
        }
        std::printf("[PIPELINE] multi-stream exit frames=%d\n", frame_count);
        return 0;
    }

    for (auto& slot : slots) {
        slot->worker = std::thread(pipeline_worker, slot.get());
    }
    std::printf("[PIPELINE] ordered slots=%d display_fps=%.3f\n",
                opt.pipeline_instances, opt.playback_fps);

    int next_dispatch = 0;
    const int prefill = target_frames > 0
        ? std::min(opt.pipeline_instances, target_frames)
        : opt.pipeline_instances;
    for (int i = 0; i < prefill; ++i) {
        dispatch_pipeline_job(*slots[i], cached[static_cast<size_t>(i) % cached.size()], i);
        ++next_dispatch;
    }

    FloatRange vis_range;
    const bool fixed_vis_range = std::isfinite(opt.disp_vis_min) && std::isfinite(opt.disp_vis_max);
    if (fixed_vis_range) {
        vis_range = {
            static_cast<float>(opt.disp_vis_min),
            static_cast<float>(opt.disp_vis_max),
            true,
        };
    }
    std::vector<uint32_t> saved_depth_xrgb;
    std::vector<uint32_t> saved_view_xrgb;
    int saved_depth_w = 0;
    int saved_depth_h = 0;
    int saved_frame = 0;
    int frame_count = 0;
    const auto frame_period = std::chrono::nanoseconds(
        static_cast<int64_t>(1000000000.0 / opt.playback_fps));
    const auto pipeline_start = std::chrono::steady_clock::now();

    while (g_running && (target_frames == 0 || frame_count < target_frames)) {
        PipelineSlot& slot = *slots[static_cast<size_t>(frame_count) % slots.size()];
        const auto wait_start = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock(slot.mutex);
            slot.cv_done.wait(lock, [&] { return slot.finished || !g_running; });
        }
        if (!g_running) {
            break;
        }
        const auto ready_time = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(ready_time - wait_start).count();
        const double infer_ms = slot.infer_ms;
        const int infer_frame = slot.frame_id;
        const CachedKittiFrame& frame = *slot.frame;

        if (!fixed_vis_range) {
            const std::vector<float>& ready_disp = slot.disp_buffers[slot.ready_disp];
            const FloatRange frame_range = estimate_disp_range(ready_disp, frame.depth_w, frame.depth_h,
                                                               frame.pad_top, frame.pad_left, disp_w);
            if (frame_range.valid) {
                if (!vis_range.valid) {
                    vis_range = frame_range;
                } else {
                    const float alpha = static_cast<float>(opt.disp_vis_alpha);
                    vis_range.lo = vis_range.lo * (1.0f - alpha) + frame_range.lo * alpha;
                    vis_range.hi = vis_range.hi * (1.0f - alpha) + frame_range.hi * alpha;
                }
            }
        }
        std::vector<uint32_t> depth_xrgb;
        const std::vector<float>& ready_disp = slot.disp_buffers[slot.ready_disp];
        if (vis_range.valid) {
            depth_xrgb = disparity_to_xrgb_range(ready_disp, frame.depth_w, frame.depth_h,
                                                 frame.pad_top, frame.pad_left, disp_w,
                                                 vis_range.lo, vis_range.hi);
        } else {
            depth_xrgb.assign(static_cast<size_t>(frame.depth_w) * frame.depth_h, 0xff000000u);
        }
        const auto colorize_end = std::chrono::steady_clock::now();

        if (target_frames == 0 || next_dispatch < target_frames) {
            dispatch_pipeline_job(slot, cached[static_cast<size_t>(next_dispatch) % cached.size()], next_dispatch);
            ++next_dispatch;
        }

        const auto display_start = ready_time;
        drm.begin_frame();
        const std::vector<uint32_t> depth_display_xrgb = resize_xrgb_nearest(
            depth_xrgb, frame.depth_w, frame.depth_h, frame.depth_rect.w, frame.depth_rect.h);
        drm.blit_xrgb(frame.left_display_xrgb, frame.left_rect.w, frame.left_rect.h, frame.left_rect);
        drm.blit_xrgb(frame.right_display_xrgb, frame.right_rect.w, frame.right_rect.h, frame.right_rect);
        drm.blit_xrgb(depth_display_xrgb, frame.depth_rect.w, frame.depth_rect.h, frame.depth_rect);
        drm.present();
        const auto present_end = std::chrono::steady_clock::now();
        double throttle_ms = 0.0;
        const auto active_elapsed = present_end - display_start;
        if (active_elapsed < frame_period) {
            const auto sleep_for = frame_period - active_elapsed;
            throttle_ms = std::chrono::duration<double, std::milli>(sleep_for).count();
            std::this_thread::sleep_for(sleep_for);
        }
        const auto display_end = std::chrono::steady_clock::now();

        ++frame_count;
        if (frame_count >= opt.save_warmup_frames) {
            saved_depth_xrgb = depth_xrgb;
            saved_depth_w = frame.depth_w;
            saved_depth_h = frame.depth_h;
            saved_frame = frame_count;
        }
        if (frame_count % 30 == 0) {
            const double elapsed = std::chrono::duration<double>(display_end - pipeline_start).count();
            const double fps = elapsed > 0.0 ? frame_count / elapsed : 0.0;
            const double colorize_ms = std::chrono::duration<double, std::milli>(colorize_end - ready_time).count();
            const double display_ms = std::chrono::duration<double, std::milli>(present_end - colorize_end).count();
            std::printf("[PIPELINE_DISPLAY] frame=%d fps=%.2f infer_frame=%d infer_ms=%.3f wait_ms=%.3f colorize_ms=%.3f display_ms=%.3f throttle_ms=%.3f source=%zu/%zu\n",
                        frame_count, fps, infer_frame, infer_ms, wait_ms,
                        colorize_ms, display_ms, throttle_ms,
                        static_cast<size_t>(infer_frame) % cached.size(), cached.size());
            if (vis_range.valid) {
                std::printf("[VIS_RANGE] lo=%.4f hi=%.4f alpha=%.3f fixed=%d\n",
                            vis_range.lo, vis_range.hi, opt.disp_vis_alpha, fixed_vis_range ? 1 : 0);
            }
        }
    }

    for (auto& slot : slots) {
        {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->quit = true;
        }
        slot->cv_job.notify_one();
    }
    for (auto& slot : slots) {
        if (slot->worker.joinable()) {
            slot->worker.join();
        }
    }

    if (frame_count > 0) {
        if (saved_view_xrgb.empty()) {
            saved_view_xrgb = drm.snapshot();
            saved_frame = frame_count;
        }
        save_xrgb_png(opt.save_depth, saved_depth_xrgb, saved_depth_w, saved_depth_h);
        save_xrgb_png(opt.save_view, saved_view_xrgb, drm.width(), drm.height());
        std::printf("[SAVE] saved frame=%d\n", saved_frame);
    }
    std::printf("[PIPELINE] exit frames=%d\n", frame_count);
    return 0;
}

int run_kitti_demo(const Options& opt) {
    const auto pairs = collect_kitti_pairs(opt.kitti_seq);
    check(pairs.empty(), "no KITTI image_02/image_03 PNG pairs found in " + opt.kitti_seq);
    std::printf("[KITTI] seq=%s pairs=%zu playback_fps=%.3f\n",
                opt.kitti_seq.c_str(), pairs.size(), opt.playback_fps);
    if (opt.offline_decode || opt.pipeline_instances > 1) {
        return run_kitti_offline_pipeline(opt, pairs);
    }

    DrmFramebuffer drm(opt.card);
    Las2Engine las2(opt.model, opt.affinity);
    std::vector<float> left_tensor;
    std::vector<float> right_tensor;
    std::vector<float> disp;
    std::vector<uint32_t> depth_xrgb;
    std::vector<uint32_t> saved_depth_xrgb;
    std::vector<uint32_t> saved_view_xrgb;
    int depth_w = 0;
    int depth_h = 0;
    int saved_depth_w = 0;
    int saved_depth_h = 0;
    int saved_frame = 0;
    int frame_count = 0;
    FloatRange vis_range;
    const bool fixed_vis_range = std::isfinite(opt.disp_vis_min) && std::isfinite(opt.disp_vis_max);
    if (fixed_vis_range) {
        vis_range = {
            static_cast<float>(opt.disp_vis_min),
            static_cast<float>(opt.disp_vis_max),
            true,
        };
    }
    int target_frames = opt.frames;
    if (target_frames > 0 && target_frames < opt.save_warmup_frames) {
        target_frames = opt.save_warmup_frames;
    }
    const auto frame_period = std::chrono::nanoseconds(
        static_cast<int64_t>(1000000000.0 / opt.playback_fps));

    while (g_running && (target_frames == 0 || frame_count < target_frames)) {
        const auto frame_start = std::chrono::steady_clock::now();
        const KittiPair& pair = pairs[static_cast<size_t>(frame_count) % pairs.size()];
        cv::Mat left_bgr = cv::imread(pair.left, cv::IMREAD_COLOR);
        cv::Mat right_bgr = cv::imread(pair.right, cv::IMREAD_COLOR);
        const auto decode_end = std::chrono::steady_clock::now();
        check(left_bgr.empty(), "failed to read KITTI left image: " + pair.left);
        check(right_bgr.empty(), "failed to read KITTI right image: " + pair.right);
        check(left_bgr.size() != right_bgr.size(), "KITTI left/right image size mismatch");

        const int model_w = las2.input_width();
        const int model_h = las2.input_height();
        const int disp_w = las2.output_width();
        const int disp_h = las2.output_height();
        const double scale = resolve_scale(left_bgr, opt.input_scale, model_w, model_h);
        cv::Mat left_padded = scale_and_pad_bgr(left_bgr, scale, model_w, model_h);
        cv::Mat right_padded = scale_and_pad_bgr(right_bgr, scale, model_w, model_h);
        const auto resize_end = std::chrono::steady_clock::now();
        bgr_to_las2_tensor(left_padded, left_tensor, las2.input_is_nhwc());
        bgr_to_las2_tensor(right_padded, right_tensor, las2.input_is_nhwc());
        const auto tensor_end = std::chrono::steady_clock::now();
        const auto infer_start = std::chrono::steady_clock::now();
        las2.infer(left_tensor.data(), right_tensor.data(), disp);
        const auto infer_end = std::chrono::steady_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        const double infer_fps = infer_ms > 0.0 ? 1000.0 / infer_ms : 0.0;
        std::printf("[INFER] frame=%d infer_ms=%.3f infer_fps=%.2f\n",
                    frame_count + 1, infer_ms, infer_fps);

        const int scaled_w = std::max(1, static_cast<int>(std::round(left_bgr.cols * scale)));
        const int scaled_h = std::max(1, static_cast<int>(std::round(left_bgr.rows * scale)));
        const int pad_left = (disp_w - scaled_w) / 2;
        const int pad_top = (disp_h - scaled_h) / 2;
        depth_w = scaled_w;
        depth_h = scaled_h;
        if (!fixed_vis_range) {
            const FloatRange frame_range = estimate_disp_range(disp, depth_w, depth_h, pad_top, pad_left, disp_w);
            if (frame_range.valid) {
                if (!vis_range.valid) {
                    vis_range = frame_range;
                } else {
                    const float alpha = static_cast<float>(opt.disp_vis_alpha);
                    vis_range.lo = vis_range.lo * (1.0f - alpha) + frame_range.lo * alpha;
                    vis_range.hi = vis_range.hi * (1.0f - alpha) + frame_range.hi * alpha;
                }
            }
        }
        if (vis_range.valid) {
            depth_xrgb = disparity_to_xrgb_range(disp, depth_w, depth_h, pad_top, pad_left,
                                                 disp_w, vis_range.lo, vis_range.hi);
        } else {
            depth_xrgb.assign(static_cast<size_t>(depth_w) * depth_h, 0xff000000u);
        }
        const auto colorize_end = std::chrono::steady_clock::now();

        drm.begin_frame();
        const int cell_w = drm.width() / 3;
        const int cell_h = drm.height();
        const Rect left_rect = compute_fit_rect(0, 0, cell_w, cell_h, left_bgr.cols, left_bgr.rows);
        const Rect right_rect = compute_fit_rect(cell_w, 0, cell_w, cell_h, right_bgr.cols, right_bgr.rows);
        const Rect depth_rect = compute_fit_rect(cell_w * 2, 0, cell_w, cell_h, depth_w, depth_h);
        const std::vector<uint32_t> left_xrgb = bgr_to_xrgb(left_bgr);
        const std::vector<uint32_t> right_xrgb = bgr_to_xrgb(right_bgr);
        drm.draw_xrgb(left_xrgb, left_bgr.cols, left_bgr.rows, left_rect);
        drm.draw_xrgb(right_xrgb, right_bgr.cols, right_bgr.rows, right_rect);
        drm.draw_xrgb(depth_xrgb, depth_w, depth_h, depth_rect);
        drm.present();
        const auto frame_end = std::chrono::steady_clock::now();
        const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - frame_start).count();
        const double resize_ms = std::chrono::duration<double, std::milli>(resize_end - decode_end).count();
        const double tensor_ms = std::chrono::duration<double, std::milli>(tensor_end - resize_end).count();
        const double colorize_ms = std::chrono::duration<double, std::milli>(colorize_end - infer_end).count();
        const double display_ms = std::chrono::duration<double, std::milli>(frame_end - colorize_end).count();
        const double e2e_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        const double e2e_fps = e2e_ms > 0.0 ? 1000.0 / e2e_ms : 0.0;

        ++frame_count;
        std::printf("[STAGE_MS] frame=%d decode=%.3f resize=%.3f tensor=%.3f infer=%.3f colorize=%.3f display=%.3f\n",
                    frame_count, decode_ms, resize_ms, tensor_ms, infer_ms, colorize_ms, display_ms);
        std::printf("[E2E] frame=%d e2e_ms=%.3f e2e_fps=%.2f\n",
                    frame_count, e2e_ms, e2e_fps);
        if (frame_count >= opt.save_warmup_frames) {
            saved_view_xrgb = drm.snapshot();
            saved_depth_xrgb = depth_xrgb;
            saved_depth_w = depth_w;
            saved_depth_h = depth_h;
            saved_frame = frame_count;
        }
        if (frame_count % 30 == 0) {
            std::printf("[KITTI] displayed frames=%d source=%zu/%zu\n",
                        frame_count, static_cast<size_t>(frame_count) % pairs.size(), pairs.size());
            if (vis_range.valid) {
                std::printf("[VIS_RANGE] lo=%.4f hi=%.4f alpha=%.3f fixed=%d\n",
                            vis_range.lo, vis_range.hi, opt.disp_vis_alpha, fixed_vis_range ? 1 : 0);
            }
        }

        const auto elapsed = std::chrono::steady_clock::now() - frame_start;
        if (elapsed < frame_period) {
            std::this_thread::sleep_for(frame_period - elapsed);
        }
    }

    if (frame_count > 0) {
        if (saved_view_xrgb.empty()) {
            saved_view_xrgb = drm.snapshot();
            saved_depth_xrgb = depth_xrgb;
            saved_depth_w = depth_w;
            saved_depth_h = depth_h;
            saved_frame = frame_count;
        }
        save_xrgb_png(opt.save_depth, saved_depth_xrgb, saved_depth_w, saved_depth_h);
        save_xrgb_png(opt.save_view, saved_view_xrgb, drm.width(), drm.height());
        std::printf("[SAVE] saved frame=%d\n", saved_frame);
    }
    std::printf("[KITTI] exit frames=%d\n", frame_count);
    return 0;
}

}  // namespace

#ifndef LAS2_REALSENSE_NO_MAIN
int main(int argc, char** argv) {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        Options opt = parse_options(argc, argv);
        if (opt.bench_preprocess) {
            return run_preprocess_bench(opt);
        }
        if (!opt.kitti_seq.empty()) {
            return run_kitti_demo(opt);
        }
        std::puts("[STAGE] create V4L2 capture");
        V4l2IrCapture cap(opt);
        std::puts("[STAGE] create DRM framebuffer");
        DrmFramebuffer drm(opt.card);
        BufferAllocator* allocator = nullptr;
        std::unique_ptr<DmaBuffer> tensor;
        std::unique_ptr<OpenClPreprocessor> cl_preprocess;
        std::unique_ptr<Las2Engine> las2;
        cl_mem cl_output = nullptr;
        const size_t tensor_elems = static_cast<size_t>(3) * kModelH * kModelW;
        if (opt.run_las2) {
            std::puts("[STAGE] create dmabuf allocator");
            allocator = CreateDmabufHeapBufferAllocator();
            check(!allocator, "CreateDmabufHeapBufferAllocator failed");
            std::puts("[STAGE] allocate output tensor dmabuf");
            tensor = std::make_unique<DmaBuffer>(allocator, tensor_elems * 2 * sizeof(float));
            std::puts("[STAGE] create OpenCL preprocessor");
            cl_preprocess = std::make_unique<OpenClPreprocessor>();
            std::puts("[STAGE] import output tensor dmabuf to OpenCL");
            cl_output = cl_preprocess->import_dmabuf(tensor->fd(), tensor->size());
            std::puts("[STAGE] create LAS2 SpaceMIT EP session");
            las2 = std::make_unique<Las2Engine>(opt.model, opt.affinity);
            std::puts("[STAGE] LAS2 session ready");
        }

        int frame_count = 0;
        std::vector<uint8_t> left;
        std::vector<uint8_t> right;
        std::vector<float> disp;
        std::vector<uint32_t> depth_xrgb;
        int depth_w = 0;
        int depth_h = 0;
        std::vector<uint32_t> saved_depth_xrgb;
        std::vector<uint32_t> saved_view_xrgb;
        int saved_depth_w = 0;
        int saved_depth_h = 0;
        int saved_frame = 0;
        int target_frames = opt.frames;
        if (target_frames > 0 && target_frames < opt.save_warmup_frames) {
            target_frames = opt.save_warmup_frames;
        }
        while (g_running && (target_frames == 0 || frame_count < target_frames)) {
            if (opt.run_las2) {
                V4l2IrCapture::CapturedFrame frame;
                if (opt.verbose) std::printf("[FRAME %d] acquire\n", frame_count + 1);
                if (!cap.acquire(frame)) {
                    continue;
                }
                cap.split_frame(frame.data, left, right);
                cl_mem cl_input = nullptr;
                try {
                    if (opt.verbose) std::printf("[FRAME %d] import input dmabuf fd=%d\n", frame_count + 1, frame.dma_fd);
                    cl_input = cl_preprocess->import_dmabuf(frame.dma_fd, cap.buffer_size());
                    if (opt.verbose) std::printf("[FRAME %d] opencl preprocess\n", frame_count + 1);
                    cl_preprocess->run(cl_input, cl_output, cap.width(), cap.height(), cap.stride());
                    if (cl_input) {
                        clReleaseMemObject(cl_input);
                        cl_input = nullptr;
                    }
                    if (opt.verbose) std::printf("[FRAME %d] ort infer\n", frame_count + 1);
                    tensor->sync_start(kSyncRead);
                    float* base = static_cast<float*>(tensor->data());
                    las2->infer(base, base + tensor_elems, disp);
                    tensor->sync_end(kSyncRead);
                    if (opt.verbose) std::printf("[FRAME %d] colorize\n", frame_count + 1);
                    depth_xrgb = disparity_to_xrgb(disp, cap.width(), cap.height(), depth_w, depth_h);
                } catch (...) {
                    if (cl_input) {
                        clReleaseMemObject(cl_input);
                    }
                    cap.release(frame);
                    throw;
                }
                cap.release(frame);
            } else {
                if (!cap.read(left, right)) {
                    continue;
                }
            }
            if (opt.verbose) std::printf("[FRAME %d] display\n", frame_count + 1);
            drm.begin_frame();
            int cell_w = drm.width() / 3;
            int cell_h = drm.height();
            Rect left_rect = compute_fit_rect(0, 0, cell_w, cell_h, cap.width(), cap.height());
            Rect right_rect = compute_fit_rect(cell_w, 0, cell_w, cell_h, cap.width(), cap.height());
            Rect depth_rect = compute_fit_rect(cell_w * 2, 0, cell_w, cell_h,
                                               depth_w > 0 ? depth_w : cap.width(),
                                               depth_h > 0 ? depth_h : cap.height());
            drm.draw_gray(left, cap.width(), cap.height(), left_rect);
            drm.draw_gray(right, cap.width(), cap.height(), right_rect);
            if (!depth_xrgb.empty()) {
                drm.draw_xrgb(depth_xrgb, depth_w, depth_h, depth_rect);
            }
            drm.present();

            ++frame_count;
            if (frame_count >= opt.save_warmup_frames) {
                saved_view_xrgb = drm.snapshot();
                if (!depth_xrgb.empty()) {
                    saved_depth_xrgb = depth_xrgb;
                    saved_depth_w = depth_w;
                    saved_depth_h = depth_h;
                }
                saved_frame = frame_count;
            }
            if (frame_count % 30 == 0) {
                std::printf("[APP] displayed frames=%d\n", frame_count);
            }
        }
        if (frame_count > 0) {
            if (saved_view_xrgb.empty()) {
                std::printf("[SAVE] no frame reached warmup=%d; saving current frame\n", opt.save_warmup_frames);
                saved_view_xrgb = drm.snapshot();
                if (!depth_xrgb.empty()) {
                    saved_depth_xrgb = depth_xrgb;
                    saved_depth_w = depth_w;
                    saved_depth_h = depth_h;
                }
                saved_frame = frame_count;
            }
            if (!saved_depth_xrgb.empty()) {
                save_xrgb_png(opt.save_depth, saved_depth_xrgb, saved_depth_w, saved_depth_h);
            }
            save_xrgb_png(opt.save_view, saved_view_xrgb, drm.width(), drm.height());
            std::printf("[SAVE] saved frame=%d\n", saved_frame);
        }
        if (cl_output) {
            std::puts("[CLEANUP] release OpenCL output mem");
            clReleaseMemObject(cl_output);
            cl_output = nullptr;
        }
        std::puts("[CLEANUP] destroy LAS2 session");
        las2.reset();
        std::puts("[CLEANUP] destroy OpenCL preprocessor");
        cl_preprocess.reset();
        std::puts("[CLEANUP] destroy tensor dmabuf");
        tensor.reset();
        if (allocator) {
            std::puts("[CLEANUP] free dmabuf allocator");
            FreeDmabufHeapBufferAllocator(allocator);
            allocator = nullptr;
        }
        std::printf("[APP] exit frames=%d\n", frame_count);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }
}
#endif
