#ifndef STEREO_PIPELINE_H
#define STEREO_PIPELINE_H

#include <array>
#include <cstdint>
#include <memory>

namespace las2
{

struct RgbImage
{
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

struct DepthImage
{
    const float* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

struct DisparityImage
{
    const float* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

struct Frame
{
    /* Engine-owned views expire at the next inference or shutdown.
        infer_prepared() depth/RGB views instead belong to its caller. */
    RgbImage rgb;
    DisparityImage disparity;
    DepthImage depth;
    int frame_id = -1;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

// Calibration of Frame::depth.  The image geometry is the LAS2 model output
// geometry, including letterbox padding.  left_from_depth transforms points
// from the rectified depth camera into the original cam0 (VIO rig) frame.
struct DepthCameraCalibration
{
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::array<double, 16> left_from_depth{
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
};

// Caller-owned NV12 DMA-BUF. The descriptor remains valid only for the
// duration of process()/prepare(). The external path copies into one reusable
// DMA buffer to avoid transient-import allocation retention on K3 PowerVR.
struct Nv12DmaFrame
{
    int dma_fd = -1;
    int width = 0;
    int height = 0;
    int y_stride = 0;
    int uv_stride = 0;
    std::uint64_t total_size = 0;
    std::uint64_t timestamp_ns = 0;
    int frame_id = -1;
};

// Geometry of a side-by-side NV12 frame supplied to ExternalStereo. Defaults
// preserve the USB stereo camera layout used by existing callers. Setting
// use_remap=false is appropriate for an already-rectified stereo dataset.
struct ExternalStereoInputConfig
{
    int frame_width = 4000;
    int frame_height = 1200;
    int crop_x = 160;
    int eye_width = 1920;
    int eye_height = 1200;
    bool swap_lr = true;
    bool use_remap = true;
    // Persistent R8/RG8 DMA image views; false retains the buffer reference path.
    // Warm up before advertising readiness to absorb first-use import latency.
    bool nv12_image_sampler = true;
    int roi_mask_x = 0;
    double rectify_alpha = 0.0;
    bool produce_rgb888 = true;
    // Optional stacked omni+radtan input. Legacy side-by-side remains default.
    bool vertical_omni = false;
    int left_camera = 0;
    int right_camera = 1;
    int rectified_width = 384;
    int rectified_height = 288;
    double horizontal_fov_degrees = 100.0;
    int pipeline_slots = 2;
    // Bottom fraction removed from the virtual rectified pair before resizing.
    // Only supported for vertical_omni; zero preserves the existing geometry.
    double crop_bottom_ratio = 0.0;
};

// Caller-owned NV12 DMA-BUF used for the optional display preview. The output
// is the rectified left image followed by colorized depth, side by side.
struct Nv12PreviewTarget
{
    int dma_fd = -1;
    int width = 0;
    int height = 0;
    int y_stride_bytes = 0;
    int uv_stride_bytes = 0;
    std::uint64_t total_size = 0;
    float depth_min_m = 0.0F;
    float depth_max_m = 3.0F;
};

enum class PreviewMode
{
    Depth,
    Disparity
};

class Camera
{
public:
    Camera();
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool initialize(const char* usb_device, const char* calibration_yaml, int core_count,
        const char* core_affinity, const char* model_path);
    /* Uses consecutive cores starting at CPU 8. */
    bool initialize(const char* usb_device, const char* calibration_yaml, int core_count,
        const char* model_path);
    bool initialize(const char* usb_device, const char* calibration_yaml, const char* model_path);
    /* Runs one inference on the latest frame prepared by the capture pipeline. */
    bool get_frame(Frame& frame, int timeout_ms = -1);
    void shutdown();
    const char* last_error() const;

private:
    struct Impl;
    Impl* impl_;
};

// ExternalStereo is the shared-camera counterpart to Camera. It owns LAS2
// preprocessing and inference resources but never opens a V4L2 device. This
// lets an MPP producer retain ownership of capture and pass its decoded NV12
// DMA-BUF directly to LAS2.
// A bounded prepared-slot lease. Releasing this object recycles its slot;
// it retains its owning pipeline even if ExternalStereo is shut down.
class PreparedStereoFrame
{
public:
    PreparedStereoFrame();
    ~PreparedStereoFrame();
    PreparedStereoFrame(PreparedStereoFrame&&) noexcept;
    PreparedStereoFrame& operator=(PreparedStereoFrame&&) noexcept;
    explicit operator bool() const noexcept;

private:
    friend class ExternalStereo;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Borrowed, caller-owned output buffers. Capacity/stride must cover every row.
// Depth is written directly on the CPU with DMA cache synchronization; RGB
// is packed on the CPU from the GPU-rectified XRGB slot. Buffers remain caller-owned.
struct StereoOutputTarget
{
    int depth_fd = -1, rgb_fd = -1;
    float* depth_data = nullptr;
    std::uint8_t* rgb_data = nullptr;
    std::uint64_t depth_size = 0, rgb_size = 0;
    int depth_stride = 0, rgb_stride = 0;
};

class ExternalStereo
{
public:
    ExternalStereo();
    ~ExternalStereo();

    ExternalStereo(const ExternalStereo&) = delete;
    ExternalStereo& operator=(const ExternalStereo&) = delete;

    bool initialize(const char* calibration_yaml, int core_count, const char* core_affinity,
        const char* model_path);
    bool initialize(const char* calibration_yaml, int core_count, int threads_per_instance,
        const char* core_affinity, const char* model_path);
    bool initialize(const char* calibration_yaml, int core_count, int threads_per_instance,
        const char* core_affinity, const char* model_path,
        const ExternalStereoInputConfig& input_config);
    // Exercise preprocessing, inference, and output conversion using an
    // internally allocated neutral NV12 DMA-BUF. This removes first-use
    // latency before a caller advertises pipeline readiness.
    bool warmup(int iterations = 2);
    bool process(const Nv12DmaFrame& input, Frame& output);
    // GPU event duration of the last successful process(): fused stereo
    // remap/normalization/tensor conversion only, excluding imports and waits.
    // Returns NaN if profiling is unavailable/invalid, or zero before process().
    // Ignore non-finite samples when aggregating timings; output remains valid.
    // Read on the process() thread. Does not change Frame's ABI.
    double last_opencl_ms() const;
    // One prepare thread may overlap one infer_prepared thread. These APIs
    // throw on invalid input/runtime errors; an empty lease means all slots
    // are busy. Do not mix them with process/warmup/render on this object.
    PreparedStereoFrame prepare(const Nv12DmaFrame& input);
    Frame infer_prepared(PreparedStereoFrame& prepared, const StereoOutputTarget& target);
    bool render_nv12_preview(const Nv12PreviewTarget& target);
    // Same DMA output, selectable color range (meters or model-output pixels).
    // Invalid samples are black. Depth clips outside the range; disparity
    // saturates to the endpoint colors. The original overload remains depth-only.
    bool render_nv12_preview(
        const Nv12PreviewTarget& target, PreviewMode mode, float minimum, float maximum);
    bool depth_camera_calibration(DepthCameraCalibration& output) const;
    void shutdown();
    const char* last_error() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace las2

#endif // STEREO_PIPELINE_H
