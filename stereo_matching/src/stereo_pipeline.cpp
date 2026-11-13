#include "stereo_pipeline.h"
#include "stereo_pipeline_internal.h"

#include "stereo_preprocessing.h"

// Depends on RectifyMaps and YAML helpers defined by the implementation above.
#include "omni_rectification.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __riscv
#include <riscv_vector.h>
#endif

namespace
{

std::mutex g_external_gpu_mutex;

class DirectV4l2YuyvCapture
{
public:
    explicit DirectV4l2YuyvCapture(const StereoPipelineOptions& opt)
    {
        try
        {
            initialize(opt);
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~DirectV4l2YuyvCapture() noexcept { cleanup(); }

    DirectV4l2YuyvCapture(const DirectV4l2YuyvCapture&) = delete;
    DirectV4l2YuyvCapture& operator=(const DirectV4l2YuyvCapture&) = delete;

private:
    void initialize(const StereoPipelineOptions& opt)
    {
        fd_ = open(opt.dev.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd_ < 0)
            fail("open " + opt.dev);

        v4l2_capability capability{};
        xioctl(VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
        if (!(capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) ||
            !(capability.device_caps & V4L2_CAP_STREAMING))
        {
            throw std::runtime_error("camera does not support V4L2 streaming capture");
        }

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = static_cast<__u32>(opt.width);
        format.fmt.pix.height = static_cast<__u32>(opt.height);
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        format.fmt.pix.field = V4L2_FIELD_ANY;
        xioctl(VIDIOC_S_FMT, &format, "VIDIOC_S_FMT YUYV");
        if (format.fmt.pix.width != static_cast<__u32>(opt.width) ||
            format.fmt.pix.height != static_cast<__u32>(opt.height) ||
            format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV)
        {
            throw std::runtime_error("camera rejected requested YUYV geometry");
        }
        width_ = static_cast<int>(format.fmt.pix.width);
        height_ = static_cast<int>(format.fmt.pix.height);
        stride_ = std::max(static_cast<int>(format.fmt.pix.bytesperline), width_ * 2);

        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = static_cast<__u32>(opt.fps);
        xioctl(VIDIOC_S_PARM, &parm, "VIDIOC_S_PARM");

        v4l2_requestbuffers request{};
        request.count = 6;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        xioctl(VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
        if (request.count < 3)
            throw std::runtime_error("camera returned too few V4L2 buffers");
        buffers_.resize(request.count);
        for (std::uint32_t i = 0; i < request.count; ++i)
        {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            xioctl(VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
            buffers_[i].size = buffer.length;
            buffers_[i].map = mmap(
                nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
            if (buffers_[i].map == MAP_FAILED)
                fail("mmap V4L2 buffer");

            v4l2_exportbuffer export_buffer{};
            export_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            export_buffer.index = i;
            export_buffer.flags = O_CLOEXEC;
            xioctl(VIDIOC_EXPBUF, &export_buffer, "VIDIOC_EXPBUF");
            buffers_[i].dma_fd = export_buffer.fd;
            xioctl(VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
        streaming_ = true;
        std::printf("[V4L2] dev=%s size=%dx%d fps=%d format=YUYV buffers=%zu dma-buf\n",
            opt.dev.c_str(), width_, height_, opt.fps, buffers_.size());
    }

    void cleanup() noexcept
    {
        if (streaming_)
        {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (auto& buffer : buffers_)
        {
            if (buffer.dma_fd >= 0)
            {
                close(buffer.dma_fd);
                buffer.dma_fd = -1;
            }
            if (buffer.map != MAP_FAILED)
            {
                munmap(buffer.map, buffer.size);
                buffer.map = MAP_FAILED;
            }
        }
        buffers_.clear();
        if (fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }

public:
    bool get_frame(VideoFrameInfo& frame, int& dma_fd, double& capture_ms)
    {
        const auto start = std::chrono::steady_clock::now();
        pollfd descriptor{fd_, POLLIN, 0};
        int result;
        do
        {
            result = poll(&descriptor, 1, 3000);
        } while (result < 0 && errno == EINTR);
        if (result <= 0)
            return false;

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &buffer) != 0)
            return false;
        if (buffer.index >= buffers_.size())
            throw std::runtime_error("invalid V4L2 buffer index");

        std::memset(&frame, 0, sizeof(frame));
        frame.u32Idx = buffer.index;
        frame.stCommFrameInfo.u32Width = static_cast<U32>(width_);
        frame.stCommFrameInfo.u32Height = static_cast<U32>(height_);
        frame.stVFrame.u32PlaneStride[0] = static_cast<U32>(stride_);
        frame.stVFrame.u32TotalSize = static_cast<U32>(buffers_[buffer.index].size);
        frame.stVFrame.u32Fd[0] = buffers_[buffer.index].dma_fd;
        dma_fd = buffers_[buffer.index].dma_fd;
        capture_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return true;
    }

    void release_frame(const VideoFrameInfo& frame)
    {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = frame.u32Idx;
        xioctl(VIDIOC_QBUF, &buffer, "VIDIOC_QBUF release");
    }

private:
    struct Buffer
    {
        void* map = MAP_FAILED;
        size_t size = 0;
        int dma_fd = -1;
    };

    [[noreturn]] void fail(const std::string& operation) const
    {
        throw std::runtime_error(operation + ": " + std::strerror(errno));
    }

    void xioctl(unsigned long request, void* argument, const char* operation)
    {
        int result;
        do
        {
            result = ioctl(fd_, request, argument);
        } while (result < 0 && errno == EINTR);
        if (result < 0)
            fail(operation);
    }

    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    int stride_ = 0;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
};

} // namespace

struct Las2Handle
{
    Las2Config cfg{};
    StereoPipelineOptions opt;
    std::shared_ptr<const RectifyMaps> maps;
    las2::DepthCameraCalibration depth_camera;
    std::unique_ptr<DirectV4l2YuyvCapture> decoder;
    std::unique_ptr<StereoPreprocessorOpenCl> cl;
    BufferAllocator* allocator = nullptr;
    std::unique_ptr<DmaBuffer> external_input_dma;
    std::vector<std::unique_ptr<StereoPipelineSlot>> slots;
    std::vector<std::unique_ptr<Las2Engine>> engines;
    std::vector<double> engine_infer_ms;
    std::thread prepare_thread;
    std::mutex prepare_mutex;
    std::condition_variable prepare_cv;
    std::mutex get_frame_mutex;
    std::vector<int> slot_states;
    std::vector<int> slot_frame_ids;
    bool stop_prepare = false;
    std::exception_ptr prepare_error;
    int model_w = 0;
    int model_h = 0;
    int disp_w = 0;
    int disp_h = 0;
    bool nhwc = false;
    int next_frame = 0;
    int next_engine = 0;
    bool started = false;
    std::vector<float> last_disp;
    std::vector<uint8_t> last_rgb;
    std::vector<float> last_depth;
    double focal_baseline_m_px = 0.0;
    int threads_per_instance = 1;
    bool produce_rgb888 = true;
    las2::ExternalStereoInputConfig external_config;
    std::mutex lease_mutex;
    std::vector<bool> leased_slots;
    std::string last_error;
};

void update_depth_camera_calibration(Las2Handle* h)
{
    if (!h || !h->maps || h->model_w <= 0 || h->model_h <= 0 || h->maps->width <= 0 ||
        h->maps->height <= 0)
    {
        throw std::runtime_error("invalid LAS2 depth camera geometry");
    }
    const double scale_x = static_cast<double>(h->model_w) / h->maps->width;
    const double scale_y = static_cast<double>(h->model_h) / h->maps->height;

    auto& calibration = h->depth_camera;
    calibration.width = h->model_w;
    calibration.height = h->model_h;
    calibration.fx = h->maps->fx * scale_x;
    calibration.fy = h->maps->fy * scale_y;
    calibration.cx = (h->maps->cx + 0.5) * scale_x - 0.5;
    calibration.cy = (h->maps->cy + 0.5) * scale_y - 0.5;
    calibration.left_from_depth = {h->maps->rectified_from_left[0], h->maps->rectified_from_left[3],
        h->maps->rectified_from_left[6], 0.0, h->maps->rectified_from_left[1],
        h->maps->rectified_from_left[4], h->maps->rectified_from_left[7], 0.0,
        h->maps->rectified_from_left[2], h->maps->rectified_from_left[5],
        h->maps->rectified_from_left[8], 0.0, 0.0, 0.0, 0.0, 1.0};
    for (int i = 0; i < 3; ++i)
        calibration.left_from_depth[i * 4 + 3] = h->maps->depth_origin_in_left[i];
}

namespace
{

std::string affinities_to_csv(const Las2Config& cfg)
{
    if (cfg.core_affinity && cfg.core_affinity[0] != '\0')
    {
        return cfg.core_affinity;
    }
    std::ostringstream os;
    const int n = std::min(cfg.affinity_count, LAS2_MAX_INSTANCES);
    for (int i = 0; i < n; ++i)
    {
        if (i)
            os << ',';
        os << cfg.affinities[i];
    }
    return os.str();
}

// A group is the CPU affinity assigned to one inference instance.  The
// legacy API uses a comma-separated list with one CPU per instance.  The
// external-input API also accepts slash-separated groups.  The EP expects
// semicolons between threads in one group; commas are normalized for callers
// that prefer CSV notation, for example: 8;9/10;11/12;13/14;15.
std::vector<std::string> split_affinity_groups(const std::string& value)
{
    if (value.find('/') == std::string::npos)
    {
        return split_csv(value);
    }

    std::vector<std::string> groups;
    size_t start = 0;
    while (start <= value.size())
    {
        const size_t slash = value.find('/', start);
        std::string group =
            value.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        std::replace(group.begin(), group.end(), ',', ';');
        if (!group.empty())
            groups.push_back(std::move(group));
        if (slash == std::string::npos)
            break;
        start = slash + 1;
    }
    return groups;
}

StereoPipelineOptions MakePipelineOptions(const Las2Config& cfg)
{
    StereoPipelineOptions opt;
    if (cfg.video_device)
        opt.dev = cfg.video_device;
    if (cfg.model_path)
        opt.model = cfg.model_path;
    if (cfg.calib_path)
        opt.calib = cfg.calib_path;
    opt.uvc_format = cfg.pixel_format == LAS2_PIXEL_FORMAT_MJPEG ? "mjpeg" : "yuyv";
    opt.width = cfg.camera_width > 0 ? cfg.camera_width : 4000;
    opt.height = cfg.camera_height > 0 ? cfg.camera_height : 1200;
    opt.fps = cfg.camera_fps > 0 ? cfg.camera_fps : 30;
    opt.crop_x = std::max(0, cfg.crop_x);
    opt.rect_w = cfg.rect_width > 0 ? cfg.rect_width : 1920;
    opt.rect_h = cfg.rect_height > 0 ? cfg.rect_height : 1200;
    opt.left_x = opt.crop_x;
    opt.right_x = opt.left_x + opt.rect_w;
    opt.swap_lr = cfg.swap_lr != 0;
    opt.use_remap = cfg.use_remap != 0;
    opt.roi_mask_x = std::max(0, cfg.roi_mask_x);
    opt.instances = std::clamp(cfg.core_count, 1, LAS2_MAX_INSTANCES);
    opt.affinities = affinities_to_csv(cfg);
    opt.rectify_alpha = std::clamp(cfg.rectify_alpha, -1.0, 1.0);
    opt.nv12_image_sampler = cfg.nv12_image_sampler != 0;
    return opt;
}

void set_error(Las2Handle* h, const std::string& error)
{
    if (h)
        h->last_error = error;
}

void prepare_loop(Las2Handle* h)
{
    try
    {
        while (true)
        {
            size_t index = 0;
            {
                std::unique_lock<std::mutex> lock(h->prepare_mutex);
                h->prepare_cv.wait(lock,
                    [&]
                    {
                        if (h->stop_prepare)
                            return true;
                        return std::any_of(h->slot_states.begin(), h->slot_states.end(),
                            [](int state) { return state == 0 || state == 2; });
                    });
                if (h->stop_prepare)
                    break;

                auto free_slot = std::find(h->slot_states.begin(), h->slot_states.end(), 0);
                if (free_slot != h->slot_states.end())
                {
                    index = static_cast<size_t>(free_slot - h->slot_states.begin());
                }
                else
                {
                    int oldest_id = INT_MAX;
                    for (size_t i = 0; i < h->slot_states.size(); ++i)
                    {
                        if (h->slot_states[i] == 2 && h->slot_frame_ids[i] < oldest_id)
                        {
                            oldest_id = h->slot_frame_ids[i];
                            index = i;
                        }
                    }
                }
                h->slot_states[index] = 1;
            }

            StereoPipelineSlot& slot = *h->slots[index];
            PreprocessedStereoFrame frame = decode_preprocess_usb(*h->decoder, *h->cl, slot, h->opt,
                h->model_w, h->model_h, h->disp_w, h->disp_h, h->nhwc);
            {
                std::lock_guard<std::mutex> lock(h->prepare_mutex);
                slot.frame = std::move(frame);
                slot.frame_id = h->next_frame++;
                h->slot_frame_ids[index] = slot.frame_id;
                h->slot_states[index] = 2;
            }
            h->prepare_cv.notify_all();
        }
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(h->prepare_mutex);
        h->prepare_error = std::current_exception();
        h->prepare_cv.notify_all();
    }
}

void xrgb_to_rgb888(const std::vector<uint32_t>& xrgb, std::vector<uint8_t>& rgb)
{
    rgb.resize(xrgb.size() * 3);
    for (size_t i = 0; i < xrgb.size(); ++i)
    {
        const uint32_t pixel = xrgb[i];
        rgb[i * 3 + 0] = static_cast<uint8_t>((pixel >> 16) & 0xffu);
        rgb[i * 3 + 1] = static_cast<uint8_t>((pixel >> 8) & 0xffu);
        rgb[i * 3 + 2] = static_cast<uint8_t>(pixel & 0xffu);
    }
}

void disparity_to_depth(const float* disparity, float* depth, int width, int height, int pad_left,
    int pad_top, int valid_width, int valid_height, float focal_baseline_m_px)
{
    const size_t count = static_cast<size_t>(width) * height;
    std::fill(depth, depth + count, 0.0f);
    const int x0 = std::clamp(pad_left, 0, width);
    const int y0 = std::clamp(pad_top, 0, height);
    const int x1 = std::clamp(pad_left + valid_width, 0, width);
    const int y1 = std::clamp(pad_top + valid_height, 0, height);
    constexpr float kMinDisparity = 1.0e-6f;

    for (int y = y0; y < y1; ++y)
    {
        const float* src = disparity + static_cast<size_t>(y) * width + x0;
        float* dst = depth + static_cast<size_t>(y) * width + x0;
        size_t remaining = static_cast<size_t>(x1 - x0);
#ifdef __riscv
        while (remaining > 0)
        {
            const size_t vl = __riscv_vsetvl_e32m4(remaining);
            const vfloat32m4_t disp = __riscv_vle32_v_f32m4(src, vl);
            const vbool8_t valid = __riscv_vmfgt_vf_f32m4_b8(disp, kMinDisparity, vl);
            const vfloat32m4_t metric = __riscv_vfrdiv_vf_f32m4(disp, focal_baseline_m_px, vl);
            __riscv_vse32_v_f32m4_m(valid, dst, metric, vl);
            src += vl;
            dst += vl;
            remaining -= vl;
        }
#else
        for (size_t x = 0; x < remaining; ++x)
        {
            const float disp = src[x];
            if (disp > kMinDisparity)
                dst[x] = focal_baseline_m_px / disp;
        }
#endif
    }
}

void stop_internal(Las2Handle* h)
{
    if (!h)
        return;
    {
        std::lock_guard<std::mutex> lock(h->prepare_mutex);
        h->stop_prepare = true;
    }
    h->prepare_cv.notify_all();
    if (h->prepare_thread.joinable())
        h->prepare_thread.join();
    for (auto& slot : h->slots)
    {
        release_slot_cl(*slot);
    }
    h->slots.clear();
    h->engines.clear();
    h->engine_infer_ms.clear();
    h->cl.reset();
    h->maps.reset();
    h->external_input_dma.reset();
    h->decoder.reset();
    if (h->allocator)
    {
        FreeDmabufHeapBufferAllocator(h->allocator);
        h->allocator = nullptr;
    }
    h->started = false;
}

void start_external(Las2Handle* h)
{
    if (!h)
    {
        throw std::runtime_error("null LAS2 handle");
    }
    h->maps = h->external_config.vertical_omni ?
        std::make_shared<const RectifyMaps>(
            load_omni_rectify_maps(h->opt.calib, h->external_config)) :
        load_shared_rectify_maps(h->opt.calib, h->opt.rect_w, h->opt.rect_h, h->opt.rectify_alpha);
    const auto affinities = split_affinity_groups(h->opt.affinities);
    check(static_cast<int>(affinities.size()) != h->opt.instances,
        "core_affinity count must equal core_count");

#ifdef LAS2_WITH_STEREO_MATCHING
    Las2Engine::validate_runtime();
#endif
    h->allocator = CreateDmabufHeapBufferAllocator();
    check(!h->allocator, "CreateDmabufHeapBufferAllocator failed");

    h->engines.reserve(static_cast<size_t>(h->opt.instances));
    for (int i = 0; i < h->opt.instances; ++i)
    {
        h->engines.push_back(std::make_unique<Las2Engine>(
            h->opt.model, affinities[static_cast<size_t>(i)], h->threads_per_instance));
    }
    h->engine_infer_ms.assign(h->engines.size(), 0.0);
    h->model_w = h->engines[0]->input_width();
    h->model_h = h->engines[0]->input_height();
    h->disp_w = h->engines[0]->output_width();
    h->disp_h = h->engines[0]->output_height();
    h->nhwc = h->engines[0]->input_is_nhwc();
    h->cl = std::make_unique<StereoPreprocessorOpenCl>(*h->maps, h->engines[0]->input_is_fp16());
    check(h->disp_w != h->model_w || h->disp_h != h->model_h,
        "model output size must match input size for aligned RGB/depth output");

    const double input_scale = static_cast<double>(h->model_w) / h->opt.rect_w;
    h->focal_baseline_m_px = h->maps->focal_px * input_scale * h->maps->baseline_m;
    update_depth_camera_calibration(h);

    const size_t tensor_bytes = h->engines[0]->input_bytes();
    const size_t preview_bytes = static_cast<size_t>(h->model_w) * h->model_h * sizeof(uint32_t);
    for (int slot_index = 0; slot_index < h->external_config.pipeline_slots; ++slot_index)
    {
        auto slot = std::make_unique<StereoPipelineSlot>();
        slot->left_tensor_dma = std::make_unique<DmaBuffer>(h->allocator, tensor_bytes);
        slot->right_tensor_dma = std::make_unique<DmaBuffer>(h->allocator, tensor_bytes);
        slot->left_preview_dma = std::make_unique<DmaBuffer>(h->allocator, preview_bytes);
        slot->right_preview_dma = std::make_unique<DmaBuffer>(h->allocator, preview_bytes);
        slot->depth_dma = std::make_unique<DmaBuffer>(
            h->allocator, static_cast<size_t>(h->disp_w) * h->disp_h * sizeof(float));
        slot->cl_left_tensor =
            h->cl->import_dmabuf(slot->left_tensor_dma->fd(), slot->left_tensor_dma->size());
        slot->cl_right_tensor =
            h->cl->import_dmabuf(slot->right_tensor_dma->fd(), slot->right_tensor_dma->size());
        slot->cl_left_preview =
            h->cl->import_dmabuf(slot->left_preview_dma->fd(), slot->left_preview_dma->size());
        slot->cl_right_preview =
            h->cl->import_dmabuf(slot->right_preview_dma->fd(), slot->right_preview_dma->size());
        slot->cl_depth = h->cl->import_dmabuf(slot->depth_dma->fd(), slot->depth_dma->size());
        h->slots.push_back(std::move(slot));
    }
    h->leased_slots.assign(h->slots.size(), false);
    h->next_engine = 0;
    h->next_frame = 0;
    h->started = true;
}

// CPU accesses to borrowed DMA buffers must finish before hardware reuses them.
class ScopedDmaAccess
{
public:
    ScopedDmaAccess(int fd, std::uint64_t access) : fd_(fd), access_(access)
    {
        dma_buf_sync sync{};
        sync.flags = DMA_BUF_SYNC_START | access_;
        check(ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync) != 0, "DMA CPU sync start failed");
    }
    ~ScopedDmaAccess()
    {
        dma_buf_sync sync{};
        sync.flags = DMA_BUF_SYNC_END | access_;
        ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync);
    }
    ScopedDmaAccess(const ScopedDmaAccess&) = delete;
    ScopedDmaAccess& operator=(const ScopedDmaAccess&) = delete;

private:
    int fd_;
    std::uint64_t access_;
};
class ScopedDmaMapping
{
public:
    ScopedDmaMapping(int fd, size_t bytes) : bytes_(bytes)
    {
        data_ = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        check(data_ == MAP_FAILED, "input DMA mmap failed");
    }
    ~ScopedDmaMapping() { munmap(data_, bytes_); }
    const void* data() const { return data_; }
    ScopedDmaMapping(const ScopedDmaMapping&) = delete;
    ScopedDmaMapping& operator=(const ScopedDmaMapping&) = delete;

private:
    void* data_;
    size_t bytes_;
};

PreprocessedStereoFrame preprocess_external_nv12(
    Las2Handle* h, const las2::Nv12DmaFrame& input, StereoPipelineSlot& slot)
{
    check(input.dma_fd < 0, "external NV12 frame has no DMA-BUF fd");
    check(input.width != h->opt.width || input.height != h->opt.height,
        "external NV12 geometry does not match LAS2 configuration");
    check(input.y_stride < input.width || input.uv_stride < input.width,
        "external NV12 stride is smaller than image width");
    const size_t required_size = static_cast<size_t>(input.y_stride) * input.height +
        static_cast<size_t>(input.uv_stride) * (input.height / 2);
    check(
        input.total_size < required_size, "external NV12 DMA-BUF is smaller than its plane layout");

    VideoFrameInfo mpp_frame{};
    mpp_frame.stVdecFrameInfo.stCommFrameInfo.u32Width = static_cast<U32>(input.width);
    mpp_frame.stVdecFrameInfo.stCommFrameInfo.u32Height = static_cast<U32>(input.height);
    mpp_frame.stVFrame.u32PlaneNum = 2;
    mpp_frame.stVFrame.u32PlaneStride[0] = static_cast<U32>(input.y_stride);
    mpp_frame.stVFrame.u32PlaneStride[1] = static_cast<U32>(input.uv_stride);
    mpp_frame.stVFrame.u32TotalSize = static_cast<U32>(input.total_size);

    const int scaled_w = h->model_w;
    const int scaled_h = h->model_h;
    const int pad_left = 0;
    const int pad_top = 0;
    double opencl_ms = 0.0;
    {
        std::lock_guard<std::mutex> gpu_lock(g_external_gpu_mutex);
        // K3 PowerVR retains allocations after transient ARM imports are
        // acquired/released. Import one private buffer for this pipeline and
        // reuse it; the borrowed source can be released as soon as prepare ends.
        if (!h->external_input_dma)
            h->external_input_dma = std::make_unique<DmaBuffer>(h->allocator, required_size);
        check(h->external_input_dma->size() != required_size,
            "external NV12 layout changed; reinitialize the pipeline");
        {
            ScopedDmaMapping source(input.dma_fd, required_size);
            ScopedDmaAccess source_read(input.dma_fd, DMA_BUF_SYNC_READ);
            ScopedDmaAccess target_write(h->external_input_dma->fd(), DMA_BUF_SYNC_WRITE);
            std::memcpy(h->external_input_dma->data(), source.data(), required_size);
        }
        const cl_mem cl_input =
            h->cl->import_input_dmabuf_cached(h->external_input_dma->fd(), required_size);
        opencl_ms = h->cl->run(cl_input, slot.cl_left_tensor, slot.cl_right_tensor,
            slot.cl_left_preview, slot.cl_right_preview, mpp_frame, h->opt, h->model_w, h->model_h,
            h->nhwc, scaled_w, scaled_h, pad_left, pad_top);
    }

    PreprocessedStereoFrame output;
    if (h->produce_rgb888)
    {
        output.left_xrgb = read_preview(*slot.left_preview_dma, h->model_w, h->model_h);
    }
    output.image_w = h->model_w;
    output.image_h = h->model_h;
    output.depth_w = scaled_w;
    output.depth_h = scaled_h;
    output.pad_left = pad_left;
    output.pad_top = pad_top;
    output.dma_fd = input.dma_fd;
    output.opencl_ms = opencl_ms;
    return output;
}

void infer_external_frame(Las2Handle* h, StereoPipelineSlot& slot, PreprocessedStereoFrame&& frame,
    int frame_id, las2::Frame* output)
{
    const size_t engine_index = static_cast<size_t>(h->next_engine++ % h->engines.size());
    const auto infer_started = std::chrono::steady_clock::now();
    slot.left_tensor_dma->sync_start(kSyncRead);
    slot.right_tensor_dma->sync_start(kSyncRead);
    h->engines[engine_index]->infer_native(
        slot.left_tensor_dma->data(), slot.right_tensor_dma->data(), h->last_disp);
    slot.left_tensor_dma->sync_end(kSyncRead);
    slot.right_tensor_dma->sync_end(kSyncRead);
    h->engine_infer_ms[engine_index] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - infer_started)
            .count();

    if (h->produce_rgb888)
    {
        xrgb_to_rgb888(frame.left_xrgb, h->last_rgb);
    }
    else
    {
        h->last_rgb.clear();
    }
    slot.depth_dma->sync_start(kSyncWrite);
    disparity_to_depth(h->last_disp.data(), static_cast<float*>(slot.depth_dma->data()), h->disp_w,
        h->disp_h, frame.pad_left, frame.pad_top, frame.depth_w, frame.depth_h,
        static_cast<float>(h->focal_baseline_m_px));
    slot.depth_dma->sync_end(kSyncWrite);
    if (!h->last_rgb.empty())
    {
        output->rgb = {h->last_rgb.data(), h->disp_w, h->disp_h, h->disp_w * 3};
    }
    output->disparity = {
        h->last_disp.data(), h->disp_w, h->disp_h, h->disp_w * static_cast<int>(sizeof(float))};
    output->depth = {static_cast<const float*>(slot.depth_dma->data()), h->disp_w, h->disp_h,
        h->disp_w * static_cast<int>(sizeof(float))};
    output->frame_id = frame_id;
}

} // namespace

extern "C"
{

    void las2_default_config(Las2Config* cfg)
    {
        if (!cfg)
            return;
        std::memset(cfg, 0, sizeof(*cfg));
        cfg->video_device = nullptr;
        cfg->model_path = nullptr;
        cfg->calib_path = nullptr;
        cfg->pixel_format = LAS2_PIXEL_FORMAT_YUYV;
        cfg->camera_width = 4000;
        cfg->camera_height = 1200;
        cfg->camera_fps = 30;
        cfg->crop_x = 160;
        cfg->rect_width = 1920;
        cfg->rect_height = 1200;
        cfg->swap_lr = 1;
        cfg->use_remap = 1;
        cfg->roi_mask_x = 0;
        cfg->core_count = 8;
        cfg->affinity_count = 8;
        cfg->queue_depth = 0;
        cfg->core_affinity = nullptr;
        cfg->rectify_alpha = 0.0;
        cfg->nv12_image_sampler = 0;
        for (int i = 0; i < cfg->affinity_count; ++i)
        {
            cfg->affinities[i] = 8 + i;
        }
    }

    Las2Handle* las2_create(const Las2Config* cfg)
    {
        if (!cfg)
            return nullptr;
        auto* h = new Las2Handle();
        h->cfg = *cfg;
        if (h->cfg.core_count <= 0 || h->cfg.core_count > LAS2_MAX_INSTANCES)
        {
            h->last_error = "core_count must be in [1, 16]";
            return h;
        }
        const std::string affinity = affinities_to_csv(h->cfg);
        const auto affinity_groups = split_affinity_groups(affinity);
        if (static_cast<int>(affinity_groups.size()) != h->cfg.core_count)
        {
            h->last_error = "core_affinity must contain exactly one group per core_count";
            return h;
        }
        h->opt = MakePipelineOptions(*cfg);
        return h;
    }

    int las2_start(Las2Handle* h)
    {
        if (!h)
            return -1;
        if (!h->last_error.empty())
            return -1;
        if (h->started)
            return 0;
        try
        {
            h->maps = load_shared_rectify_maps(
                h->opt.calib, h->opt.rect_w, h->opt.rect_h, h->opt.rectify_alpha);
            const auto affinities = split_affinity_groups(h->opt.affinities);
            check(static_cast<int>(affinities.size()) != h->opt.instances,
                "core_affinity count must equal core_count");
            check(h->opt.left_x < 0 || h->opt.right_x < 0 ||
                    h->opt.left_x + h->opt.rect_w > h->opt.width ||
                    h->opt.right_x + h->opt.rect_w > h->opt.width,
                "invalid source layout: left/right rect exceeds frame width");

#ifdef LAS2_WITH_STEREO_MATCHING
            Las2Engine::validate_runtime();
#endif
            h->allocator = CreateDmabufHeapBufferAllocator();
            check(!h->allocator, "CreateDmabufHeapBufferAllocator failed");

            h->engines.reserve(static_cast<size_t>(h->opt.instances));
            for (int i = 0; i < h->opt.instances; ++i)
            {
                h->engines.push_back(std::make_unique<Las2Engine>(
                    h->opt.model, affinities[static_cast<size_t>(i)], 1));
            }
            h->engine_infer_ms.assign(h->engines.size(), 0.0);

            h->model_w = h->engines[0]->input_width();
            h->model_h = h->engines[0]->input_height();
            h->disp_w = h->engines[0]->output_width();
            h->disp_h = h->engines[0]->output_height();
            h->nhwc = h->engines[0]->input_is_nhwc();
            h->cl = std::make_unique<StereoPreprocessorOpenCl>(
                *h->maps, h->engines[0]->input_is_fp16());
            check(h->disp_w != h->model_w || h->disp_h != h->model_h,
                "model output size must match input size for aligned RGB/depth output");
            const double input_scale = static_cast<double>(h->model_w) / h->opt.rect_w;
            h->focal_baseline_m_px = h->maps->focal_px * input_scale * h->maps->baseline_m;
            update_depth_camera_calibration(h);
            const size_t tensor_bytes = h->engines[0]->input_bytes();
            const size_t preview_bytes =
                static_cast<size_t>(h->model_w) * h->model_h * sizeof(uint32_t);

            const int prepare_slots = std::max(2, h->cfg.queue_depth > 0 ? h->cfg.queue_depth : 3);
            h->slots.reserve(static_cast<size_t>(prepare_slots));
            for (int i = 0; i < prepare_slots; ++i)
            {
                auto slot = std::make_unique<StereoPipelineSlot>();
                slot->left_tensor_dma = std::make_unique<DmaBuffer>(h->allocator, tensor_bytes);
                slot->right_tensor_dma = std::make_unique<DmaBuffer>(h->allocator, tensor_bytes);
                slot->left_preview_dma = std::make_unique<DmaBuffer>(h->allocator, preview_bytes);
                slot->right_preview_dma = std::make_unique<DmaBuffer>(h->allocator, preview_bytes);
                slot->cl_left_tensor = h->cl->import_dmabuf(
                    slot->left_tensor_dma->fd(), slot->left_tensor_dma->size());
                slot->cl_right_tensor = h->cl->import_dmabuf(
                    slot->right_tensor_dma->fd(), slot->right_tensor_dma->size());
                slot->cl_left_preview = h->cl->import_dmabuf(
                    slot->left_preview_dma->fd(), slot->left_preview_dma->size());
                slot->cl_right_preview = h->cl->import_dmabuf(
                    slot->right_preview_dma->fd(), slot->right_preview_dma->size());
                h->slots.push_back(std::move(slot));
            }

            // Start direct V4L2 capture only after all inference sessions are ready.
            h->decoder = std::make_unique<DirectV4l2YuyvCapture>(h->opt);

            h->next_frame = 0;
            h->next_engine = 0;
            h->slot_states.assign(h->slots.size(), 0);
            h->slot_frame_ids.assign(h->slots.size(), -1);
            h->stop_prepare = false;
            h->prepare_error = nullptr;
            h->started = true;
            h->prepare_thread = std::thread(prepare_loop, h);
            return 0;
        }
        catch (const std::exception& e)
        {
            set_error(h, e.what());
            stop_internal(h);
            return -1;
        }
    }

    int las2_get_frame(Las2Handle* h, Las2InternalFrame* out, int timeout_ms)
    {
        if (!h || !out)
            return -1;
        if (!h->started)
        {
            set_error(h, "las2_get_frame called before las2_start");
            return -1;
        }
        std::lock_guard<std::mutex> call_lock(h->get_frame_mutex);
        try
        {
            size_t slot_index = 0;
            {
                std::unique_lock<std::mutex> lock(h->prepare_mutex);
                const auto available = [&]
                {
                    return h->prepare_error ||
                        std::any_of(h->slot_states.begin(), h->slot_states.end(),
                            [](int state) { return state == 2; });
                };
                bool ready = false;
                if (timeout_ms < 0)
                {
                    h->prepare_cv.wait(lock, available);
                    ready = true;
                }
                else
                {
                    ready = h->prepare_cv.wait_for(
                        lock, std::chrono::milliseconds(timeout_ms), available);
                }
                if (!ready)
                    return 1;
                if (h->prepare_error)
                    std::rethrow_exception(h->prepare_error);
                int newest_id = -1;
                for (size_t i = 0; i < h->slot_states.size(); ++i)
                {
                    if (h->slot_states[i] == 2 && h->slot_frame_ids[i] > newest_id)
                    {
                        newest_id = h->slot_frame_ids[i];
                        slot_index = i;
                    }
                }
                h->slot_states[slot_index] = 3;
            }
            h->prepare_cv.notify_all();

            StereoPipelineSlot& slot = *h->slots[slot_index];
            const size_t engine_index = static_cast<size_t>(h->next_engine++ % h->engines.size());
            const auto infer_t0 = std::chrono::steady_clock::now();
            slot.left_tensor_dma->sync_start(kSyncRead);
            slot.right_tensor_dma->sync_start(kSyncRead);
            h->engines[engine_index]->infer_native(
                slot.left_tensor_dma->data(), slot.right_tensor_dma->data(), h->last_disp);
            slot.left_tensor_dma->sync_end(kSyncRead);
            slot.right_tensor_dma->sync_end(kSyncRead);
            const double infer_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - infer_t0)
                                        .count();
            h->engine_infer_ms[engine_index] = infer_ms;

            xrgb_to_rgb888(slot.frame.left_xrgb, h->last_rgb);
            h->last_depth.resize(h->last_disp.size());
            const auto depth_t0 = std::chrono::steady_clock::now();
            disparity_to_depth(h->last_disp.data(), h->last_depth.data(), h->disp_w, h->disp_h,
                slot.frame.pad_left, slot.frame.pad_top, slot.frame.depth_w, slot.frame.depth_h,
                static_cast<float>(h->focal_baseline_m_px));
            const auto depth_t1 = std::chrono::steady_clock::now();
            const double depth_ms =
                std::chrono::duration<double, std::milli>(depth_t1 - depth_t0).count();
            double total_infer_fps = 0.0;
            for (const double slot_ms : h->engine_infer_ms)
            {
                if (slot_ms > 0.0)
                    total_infer_fps += 1000.0 / slot_ms;
            }

            std::memset(out, 0, sizeof(*out));
            out->disparity = h->last_disp.data();
            out->width = h->disp_w;
            out->height = h->disp_h;
            out->frame_id = slot.frame_id;
            out->decode_ms = slot.frame.decode_ms;
            out->opencl_ms = slot.frame.opencl_ms;
            out->infer_ms = infer_ms;
            out->total_infer_fps = total_infer_fps;
            out->left_rgb = h->last_rgb.data();
            out->depth_m = h->last_depth.data();
            out->rgb_stride_bytes = h->disp_w * 3;
            out->depth_stride_bytes = h->disp_w * static_cast<int>(sizeof(float));
            out->depth_convert_ms = depth_ms;
            out->focal_baseline_m_px = h->focal_baseline_m_px;

            {
                std::lock_guard<std::mutex> lock(h->prepare_mutex);
                h->slot_states[slot_index] = 0;
            }
            h->prepare_cv.notify_all();
            return 0;
        }
        catch (const std::exception& e)
        {
            {
                std::lock_guard<std::mutex> lock(h->prepare_mutex);
                for (int& state : h->slot_states)
                {
                    if (state == 3)
                        state = 0;
                }
            }
            h->prepare_cv.notify_all();
            set_error(h, e.what());
            return -1;
        }
    }

    void las2_release_frame(Las2Handle*, Las2InternalFrame* frame)
    {
        if (frame)
            std::memset(frame, 0, sizeof(*frame));
    }

    void las2_stop(Las2Handle* h) { stop_internal(h); }

    void las2_destroy(Las2Handle* h)
    {
        if (!h)
            return;
        stop_internal(h);
        delete h;
    }

    const char* las2_last_error(Las2Handle* h)
    {
        if (!h)
            return "null Las2Handle";
        return h->last_error.c_str();
    }

} // extern "C"

struct las2::Camera::Impl
{
    Las2Handle* handle = nullptr;
    std::string model_path;
    std::string calib_path;
    std::string core_affinity;
    std::string error;
};

las2::Camera::Camera() : impl_(new Impl()) {}

las2::Camera::~Camera()
{
    shutdown();
    delete impl_;
}

bool las2::Camera::initialize(const char* usb_device, const char* calibration_yaml, int core_count,
    const char* core_affinity, const char* model_path)
{
    shutdown();
    if (!usb_device || usb_device[0] == '\0')
    {
        impl_->error = "usb_device is empty";
        return false;
    }
    if (!calibration_yaml || calibration_yaml[0] == '\0')
    {
        impl_->error = "calibration_yaml is empty";
        return false;
    }
    if (!model_path || model_path[0] == '\0')
    {
        impl_->error = "model_path is empty";
        return false;
    }
    if (core_count < 1 || core_count > LAS2_MAX_INSTANCES)
    {
        impl_->error = "core_count must be in [1, 16]";
        return false;
    }
    if (!core_affinity || core_affinity[0] == '\0')
    {
        impl_->error = "core_affinity is empty";
        return false;
    }

    impl_->model_path = model_path;
    impl_->calib_path = calibration_yaml;
    impl_->core_affinity = core_affinity;

    Las2Config cfg;
    las2_default_config(&cfg);
    cfg.video_device = usb_device;
    cfg.model_path = impl_->model_path.c_str();
    cfg.calib_path = impl_->calib_path.c_str();
    cfg.core_count = core_count;
    cfg.core_affinity = impl_->core_affinity.c_str();

    impl_->handle = las2_create(&cfg);
    if (!impl_->handle || las2_start(impl_->handle) != 0)
    {
        impl_->error =
            impl_->handle ? las2_last_error(impl_->handle) : "failed to create LAS2 pipeline";
        shutdown();
        return false;
    }
    impl_->error.clear();
    return true;
}

bool las2::Camera::initialize(
    const char* usb_device, const char* calibration_yaml, int core_count, const char* model_path)
{
    std::ostringstream affinity;
    for (int i = 0; i < core_count; ++i)
    {
        if (i)
            affinity << ',';
        affinity << 8 + i;
    }
    return initialize(usb_device, calibration_yaml, core_count, affinity.str().c_str(), model_path);
}

bool las2::Camera::initialize(
    const char* usb_device, const char* calibration_yaml, const char* model_path)
{
    return initialize(usb_device, calibration_yaml, 8, "8,9,10,11,12,13,14,15", model_path);
}

bool las2::Camera::get_frame(Frame& frame, int timeout_ms)
{
    frame = {};
    if (!impl_->handle)
    {
        impl_->error = "camera is not initialized";
        return false;
    }
    Las2InternalFrame internal{};
    const int rc = las2_get_frame(impl_->handle, &internal, timeout_ms);
    if (rc != 0)
    {
        impl_->error = rc == 1 ? "get_frame timed out" : las2_last_error(impl_->handle);
        return false;
    }
    frame.rgb = {internal.left_rgb, internal.width, internal.height, internal.rgb_stride_bytes};
    frame.depth = {internal.depth_m, internal.width, internal.height, internal.depth_stride_bytes};
    frame.frame_id = internal.frame_id;
    impl_->error.clear();
    return true;
}

void las2::Camera::shutdown()
{
    if (impl_ && impl_->handle)
    {
        las2_destroy(impl_->handle);
        impl_->handle = nullptr;
    }
}

const char* las2::Camera::last_error() const
{
    return impl_ ? impl_->error.c_str() : "invalid camera object";
}

struct las2::ExternalStereo::Impl
{
    double last_opencl_ms = 0.0;
    std::shared_ptr<Las2Handle> handle;
    std::string model_path;
    std::string calib_path;
    std::string core_affinity;
    std::string error;
};

las2::ExternalStereo::ExternalStereo() : impl_(new Impl()) {}

las2::ExternalStereo::~ExternalStereo()
{
    shutdown();
    delete impl_;
}

bool las2::ExternalStereo::initialize(
    const char* calibration_yaml, int core_count, const char* core_affinity, const char* model_path)
{
    return initialize(calibration_yaml, core_count, 1, core_affinity, model_path);
}

bool las2::ExternalStereo::initialize(const char* calibration_yaml, int core_count,
    int threads_per_instance, const char* core_affinity, const char* model_path)
{
    return initialize(calibration_yaml, core_count, threads_per_instance, core_affinity, model_path,
        ExternalStereoInputConfig{});
}

bool las2::ExternalStereo::initialize(const char* calibration_yaml, int core_count,
    int threads_per_instance, const char* core_affinity, const char* model_path,
    const ExternalStereoInputConfig& input_config)
{
    shutdown();
    if (!calibration_yaml || calibration_yaml[0] == '\0')
    {
        impl_->error = "calibration_yaml is empty";
        return false;
    }
    if (!model_path || model_path[0] == '\0')
    {
        impl_->error = "model_path is empty";
        return false;
    }
    if (core_count < 1 || core_count > LAS2_MAX_INSTANCES)
    {
        impl_->error = "core_count must be in [1, 16]";
        return false;
    }
    if (threads_per_instance < 1 || threads_per_instance > 16)
    {
        impl_->error = "threads_per_instance must be in [1, 16]";
        return false;
    }
    if (!core_affinity || core_affinity[0] == '\0')
    {
        impl_->error = "core_affinity is empty";
        return false;
    }
    if (input_config.frame_width <= 0 || input_config.frame_height <= 0 ||
        input_config.eye_width <= 0 || input_config.eye_height <= 0 ||
        (input_config.frame_width & 1) != 0 || (input_config.frame_height & 1) != 0 ||
        (input_config.eye_width & 1) != 0 || (input_config.eye_height & 1) != 0 ||
        input_config.crop_x < 0 || input_config.roi_mask_x < 0 ||
        (!input_config.vertical_omni &&
            input_config.crop_x + 2 * input_config.eye_width > input_config.frame_width) ||
        input_config.eye_height > input_config.frame_height)
    {
        impl_->error = "invalid side-by-side NV12 input geometry";
        return false;
    }
    if (!std::isfinite(input_config.crop_bottom_ratio) || input_config.crop_bottom_ratio < 0.0 ||
        input_config.crop_bottom_ratio >= 1.0 ||
        (!input_config.vertical_omni && input_config.crop_bottom_ratio != 0.0))
    {
        impl_->error = "crop_bottom_ratio must be finite, in [0, 1), and requires vertical_omni";
        return false;
    }
    if (input_config.pipeline_slots < 2 || input_config.pipeline_slots > 4 ||
        (input_config.vertical_omni &&
            (input_config.left_camera < 0 || input_config.right_camera < 0 ||
                input_config.left_camera == input_config.right_camera ||
                input_config.eye_width != input_config.frame_width ||
                input_config.frame_height % input_config.eye_height != 0 ||
                input_config.left_camera >= input_config.frame_height / input_config.eye_height ||
                input_config.right_camera >= input_config.frame_height / input_config.eye_height ||
                input_config.rectified_width <= 0 || input_config.rectified_height <= 0 ||
                input_config.rectified_width > 4096 || input_config.rectified_height > 4096 ||
                !std::isfinite(input_config.horizontal_fov_degrees) ||
                input_config.horizontal_fov_degrees <= 0 ||
                input_config.horizontal_fov_degrees >= 180 || input_config.swap_lr ||
                !input_config.use_remap)))
    {
        impl_->error = "invalid pipeline slots or stacked omni geometry";
        return false;
    }
    if (input_config.rectify_alpha < -1.0 || input_config.rectify_alpha > 1.0)
    {
        impl_->error = "rectify_alpha must be in [-1, 1]";
        return false;
    }

    impl_->calib_path = calibration_yaml;
    impl_->model_path = model_path;
    impl_->core_affinity = core_affinity;
    Las2Config cfg{};
    las2_default_config(&cfg);
    cfg.model_path = impl_->model_path.c_str();
    cfg.calib_path = impl_->calib_path.c_str();
    cfg.core_count = core_count;
    cfg.core_affinity = impl_->core_affinity.c_str();
    cfg.pixel_format = LAS2_PIXEL_FORMAT_MJPEG;
    cfg.camera_width = input_config.frame_width;
    cfg.camera_height = input_config.frame_height;
    cfg.crop_x = input_config.crop_x;
    cfg.rect_width = input_config.eye_width;
    cfg.rect_height = input_config.eye_height;
    cfg.swap_lr = input_config.swap_lr ? 1 : 0;
    cfg.use_remap = input_config.use_remap ? 1 : 0;
    cfg.nv12_image_sampler = input_config.nv12_image_sampler ? 1 : 0;
    cfg.roi_mask_x = input_config.roi_mask_x;
    cfg.rectify_alpha = input_config.rectify_alpha;
    impl_->handle.reset(las2_create(&cfg), las2_destroy);
    if (!impl_->handle || !impl_->handle->last_error.empty())
    {
        impl_->error = impl_->handle ? impl_->handle->last_error : "failed to create LAS2 pipeline";
        shutdown();
        return false;
    }
    impl_->handle->threads_per_instance = threads_per_instance;
    impl_->handle->produce_rgb888 = input_config.produce_rgb888;
    impl_->handle->external_config = input_config;
    if (input_config.vertical_omni)
    {
        auto& opt = impl_->handle->opt;
        opt.left_x = opt.right_x = 0;
        opt.left_y = input_config.left_camera * input_config.eye_height;
        opt.right_y = input_config.right_camera * input_config.eye_height;
        opt.source_eye_width = input_config.eye_width;
        opt.source_eye_height = input_config.eye_height;
        opt.rect_w = input_config.rectified_width;
        opt.rect_h = input_config.rectified_height;
    }

    try
    {
        // The producer supplies decoded NV12, so LAS2 must not select the
        // direct YUYV capture path when constructing its OpenCL kernel state.
        impl_->handle->opt.uvc_format = "mjpeg";
        start_external(impl_->handle.get());
        impl_->error.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        impl_->error = error.what();
        shutdown();
        return false;
    }
}

bool las2::ExternalStereo::process(const Nv12DmaFrame& input, Frame& output)
{
    output = {};
    if (!impl_->handle || !impl_->handle->started)
    {
        impl_->error = "external stereo is not initialized";
        return false;
    }
    try
    {
        StereoPipelineSlot& slot = *impl_->handle->slots.front();
        PreprocessedStereoFrame preprocessed =
            preprocess_external_nv12(impl_->handle.get(), input, slot);
        const double opencl_ms = preprocessed.opencl_ms;
        infer_external_frame(
            impl_->handle.get(), slot, std::move(preprocessed), input.frame_id, &output);
        impl_->last_opencl_ms = opencl_ms;
        impl_->error.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        impl_->error = error.what();
        return false;
    }
}

double las2::ExternalStereo::last_opencl_ms() const { return impl_ ? impl_->last_opencl_ms : 0.0; }

bool las2::ExternalStereo::warmup(int iterations)
{
    if (!impl_->handle || !impl_->handle->started)
    {
        impl_->error = "external stereo is not initialized";
        return false;
    }
    if (iterations < 0 || iterations > 100)
    {
        impl_->error = "warmup iterations must be in [0, 100]";
        return false;
    }
    if (iterations == 0)
    {
        impl_->error.clear();
        return true;
    }

    Las2Handle* handle = impl_->handle.get();
    const int width = handle->opt.width;
    const int height = handle->opt.height;
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    const std::size_t total_size = y_size + y_size / 2U;
    try
    {
        DmaBuffer input(handle->allocator, total_size);
        // Synthetic frames have compact strides, which need not match the
        // producer's aligned layout. Keep their private copy/import separate
        // from the real stream and restore an existing live buffer unchanged.
        struct ScopedWarmupInput
        {
            explicit ScopedWarmupInput(Las2Handle* pipeline) : handle(pipeline)
            {
                std::lock_guard<std::mutex> gpu_lock(g_external_gpu_mutex);
                previous = std::move(handle->external_input_dma);
            }
            ~ScopedWarmupInput()
            {
                std::lock_guard<std::mutex> gpu_lock(g_external_gpu_mutex);
                if (handle->external_input_dma)
                {
                    // The cache owns an import of the private copy, never of
                    // input.fd(). Release it before closing/reusing its fd.
                    handle->cl->release_input_dmabuf_cached(handle->external_input_dma->fd());
                    handle->external_input_dma.reset();
                }
                handle->external_input_dma = std::move(previous);
            }
            Las2Handle* handle;
            std::unique_ptr<DmaBuffer> previous;
        } warmup_input(handle);
        input.sync_start(kSyncWrite);
        std::memset(input.data(), 128, total_size);
        input.sync_end(kSyncWrite);

        Nv12DmaFrame frame;
        frame.dma_fd = input.fd();
        frame.width = width;
        frame.height = height;
        frame.y_stride = width;
        frame.uv_stride = width;
        frame.total_size = total_size;
        frame.frame_id = -1;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            Frame output;
            if (!process(frame, output))
            {
                return false;
            }
        }
        impl_->error.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        impl_->error = error.what();
        return false;
    }
}

bool las2::ExternalStereo::render_nv12_preview(const Nv12PreviewTarget& target)
{
    return render_nv12_preview(target, PreviewMode::Depth, target.depth_min_m, target.depth_max_m);
}

bool las2::ExternalStereo::render_nv12_preview(
    const Nv12PreviewTarget& target, PreviewMode mode, float minimum, float maximum)
{
    if (!impl_->handle || !impl_->handle->started || impl_->handle->slots.empty())
    {
        impl_->error = "external stereo is not initialized";
        return false;
    }
    if (target.dma_fd < 0 || target.width != impl_->handle->disp_w * 2 ||
        target.height != impl_->handle->disp_h || target.y_stride_bytes < target.width ||
        target.uv_stride_bytes < target.width || (impl_->handle->disp_w & 1) != 0 ||
        (target.height & 1) != 0 ||
        target.total_size < static_cast<std::uint64_t>(target.y_stride_bytes) * target.height +
                static_cast<std::uint64_t>(target.uv_stride_bytes) * (target.height / 2) ||
        (mode != PreviewMode::Depth && mode != PreviewMode::Disparity) || !std::isfinite(minimum) ||
        !std::isfinite(maximum) || !(maximum > minimum))
    {
        impl_->error = "invalid NV12 preview target";
        return false;
    }
    try
    {
        StereoPipelineSlot& slot = *impl_->handle->slots.front();
        cl_mem output = impl_->handle->cl->import_input_dmabuf_cached(
            target.dma_fd, static_cast<size_t>(target.total_size));
        std::lock_guard<std::mutex> gpu_lock(g_external_gpu_mutex);
        const bool disparity = mode == PreviewMode::Disparity;
        if (disparity &&
            impl_->handle->last_disp.size() !=
                static_cast<size_t>(impl_->handle->disp_w) * impl_->handle->disp_h)
        {
            impl_->error = "no disparity available; call process() before preview";
            return false;
        }
        impl_->handle->cl->compose_nv12_depth(slot.cl_left_preview, slot.cl_depth, output,
            impl_->handle->disp_w, impl_->handle->disp_h, target.y_stride_bytes,
            target.uv_stride_bytes, minimum, maximum,
            disparity ? impl_->handle->last_disp.data() : nullptr);
        impl_->error.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        impl_->error = error.what();
        return false;
    }
}

bool las2::ExternalStereo::depth_camera_calibration(DepthCameraCalibration& output) const
{
    output = {};
    if (!impl_ || !impl_->handle || !impl_->handle->started)
        return false;
    output = impl_->handle->depth_camera;
    return true;
}

void las2::ExternalStereo::shutdown()
{
    if (impl_ && impl_->handle)
    {
        impl_->handle.reset();
    }
}

const char* las2::ExternalStereo::last_error() const
{
    return impl_ ? impl_->error.c_str() : "invalid external stereo object";
}

struct las2::PreparedStereoFrame::Impl
{
    std::shared_ptr<Las2Handle> owner;
    std::size_t slot = 0;
    PreprocessedStereoFrame frame;
    int frame_id = -1;
    double preprocess_ms = 0;
    bool consumed = false;
    ~Impl()
    {
        if (owner)
        {
            std::lock_guard<std::mutex> lock(owner->lease_mutex);
            owner->leased_slots[slot] = false;
        }
    }
};
las2::PreparedStereoFrame::PreparedStereoFrame() = default;
las2::PreparedStereoFrame::~PreparedStereoFrame() = default;
las2::PreparedStereoFrame::PreparedStereoFrame(PreparedStereoFrame&&) noexcept = default;
las2::PreparedStereoFrame& las2::PreparedStereoFrame::operator=(
    PreparedStereoFrame&&) noexcept = default;
las2::PreparedStereoFrame::operator bool() const noexcept { return bool(impl_); }

las2::PreparedStereoFrame las2::ExternalStereo::prepare(const Nv12DmaFrame& input)
{
    if (!impl_->handle || !impl_->handle->started)
        throw std::runtime_error("LAS2 not initialized");
    PreparedStereoFrame result;
    auto h = impl_->handle;
    auto state = std::make_unique<PreparedStereoFrame::Impl>();
    {
        std::lock_guard<std::mutex> lock(h->lease_mutex);
        const auto slot = std::find(h->leased_slots.begin(), h->leased_slots.end(), false);
        if (slot == h->leased_slots.end())
            return result;
        state->slot = std::distance(h->leased_slots.begin(), slot);
        *slot = true;
        state->owner = h;
    }
    const auto start = std::chrono::steady_clock::now();
    state->frame = preprocess_external_nv12(h.get(), input, *h->slots[state->slot]);
    state->frame_id = input.frame_id;
    state->preprocess_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    result.impl_ = std::move(state);
    return result;
}

las2::Frame las2::ExternalStereo::infer_prepared(
    PreparedStereoFrame& prepared, const StereoOutputTarget& target)
{
    if (!prepared.impl_ || prepared.impl_->owner != impl_->handle || prepared.impl_->consumed)
        throw std::runtime_error("invalid or consumed prepared frame");
    auto& state = *prepared.impl_;
    auto* h = state.owner.get();
    if (target.depth_fd < 0 || target.rgb_fd < 0 || !target.depth_data || !target.rgb_data ||
        target.depth_stride < h->disp_w * int(sizeof(float)) ||
        target.depth_stride % sizeof(float) || target.rgb_stride < h->model_w * 3 ||
        target.depth_size < static_cast<std::uint64_t>(target.depth_stride) * h->disp_h ||
        target.rgb_size < static_cast<std::uint64_t>(target.rgb_stride) * h->model_h)
        throw std::runtime_error("invalid caller-owned LAS2 output buffers");
    state.consumed = true;
    auto& slot = *h->slots[state.slot];
    Frame output;
    output.frame_id = state.frame_id;
    output.preprocess_ms = state.preprocess_ms;
    const auto started = std::chrono::steady_clock::now();
    slot.left_tensor_dma->sync_start(kSyncRead);
    slot.right_tensor_dma->sync_start(kSyncRead);
    try
    {
        h->engines[static_cast<std::size_t>(h->next_engine++ % h->engines.size())]->infer_native(
            slot.left_tensor_dma->data(), slot.right_tensor_dma->data(), h->last_disp);
    }
    catch (...)
    {
        slot.left_tensor_dma->sync_end(kSyncRead);
        slot.right_tensor_dma->sync_end(kSyncRead);
        throw;
    }
    slot.left_tensor_dma->sync_end(kSyncRead);
    slot.right_tensor_dma->sync_end(kSyncRead);
    const auto inferred = std::chrono::steady_clock::now();
    output.inference_ms = std::chrono::duration<double, std::milli>(inferred - started).count();
    dma_buf_sync sync{};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    check(ioctl(target.depth_fd, DMA_BUF_IOCTL_SYNC, &sync) != 0, "depth target sync start failed");
    for (int y = 0; y < h->disp_h; ++y)
    {
        auto* row = reinterpret_cast<float*>(
            reinterpret_cast<char*>(target.depth_data) + y * target.depth_stride);
        disparity_to_depth(h->last_disp.data() + y * h->disp_w, row, h->disp_w, 1, 0, 0, h->disp_w,
            1, static_cast<float>(h->focal_baseline_m_px));
    }
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    check(ioctl(target.depth_fd, DMA_BUF_IOCTL_SYNC, &sync) != 0, "depth target sync end failed");
    {
        // The rectified XRGB result is already available in the prepared slot.
        // Pack on the CPU to avoid another transient OpenCL import per output.
        ScopedDmaAccess source_read(slot.left_preview_dma->fd(), DMA_BUF_SYNC_READ);
        ScopedDmaAccess target_write(target.rgb_fd, DMA_BUF_SYNC_WRITE);
        const auto* pixels = static_cast<const std::uint32_t*>(slot.left_preview_dma->data());
        for (int y = 0; y < h->model_h; ++y)
        {
            auto* row = target.rgb_data + y * target.rgb_stride;
            for (int x = 0; x < h->model_w; ++x)
            {
                const auto pixel = pixels[y * h->model_w + x];
                row[3 * x] = static_cast<std::uint8_t>(pixel >> 16);
                row[3 * x + 1] = static_cast<std::uint8_t>(pixel >> 8);
                row[3 * x + 2] = static_cast<std::uint8_t>(pixel);
            }
        }
    }
    output.postprocess_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inferred)
            .count();
    output.depth = {target.depth_data, h->disp_w, h->disp_h, target.depth_stride};
    output.rgb = {target.rgb_data, h->model_w, h->model_h, target.rgb_stride};
    output.disparity = {h->last_disp.data(), h->disp_w, h->disp_h, h->disp_w * int(sizeof(float))};
    return output;
}
