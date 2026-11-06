#ifndef LAS2_USB_STEREO_INTERNAL_H
#define LAS2_USB_STEREO_INTERNAL_H

#include <stdint.h>

#define LAS2_MAX_INSTANCES 16

typedef enum Las2PixelFormat {
    LAS2_PIXEL_FORMAT_YUYV = 0,
    LAS2_PIXEL_FORMAT_MJPEG = 1
} Las2PixelFormat;

typedef struct Las2Config {
    const char* video_device;
    const char* model_path;
    const char* calib_path;
    Las2PixelFormat pixel_format;
    int camera_width;
    int camera_height;
    int camera_fps;
    int crop_x;
    int rect_width;
    int rect_height;
    int swap_lr;
    int use_remap;
    int roi_mask_x;
    int core_count;
    int affinities[LAS2_MAX_INSTANCES];
    int affinity_count;
    int queue_depth;
    const char* core_affinity;
    double rectify_alpha;
    int nv12_image_sampler;
} Las2Config;

typedef struct Las2InternalFrame {
    const float* disparity;
    int width;
    int height;
    int frame_id;
    double decode_ms;
    double opencl_ms;
    double infer_ms;
    double total_infer_fps;
    const uint8_t* left_rgb;
    const float* depth_m;
    int rgb_stride_bytes;
    int depth_stride_bytes;
    double depth_convert_ms;
    double focal_baseline_m_px;
} Las2InternalFrame;

typedef struct Las2Handle Las2Handle;

extern "C" {
void las2_default_config(Las2Config* cfg);
Las2Handle* las2_create(const Las2Config* cfg);
int las2_start(Las2Handle* handle);
int las2_get_frame(Las2Handle* handle, Las2InternalFrame* out, int timeout_ms);
void las2_release_frame(Las2Handle* handle, Las2InternalFrame* frame);
void las2_stop(Las2Handle* handle);
void las2_destroy(Las2Handle* handle);
const char* las2_last_error(Las2Handle* handle);
}

#endif
