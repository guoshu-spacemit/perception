# 平面图像拼接 (Planar Image Stitching)

基于双目相机的平面图像拼接模块，用于将左右两路视角重叠的图像拼接为一张宽幅全景图。程序从摄像头（或回退图片）读取两路图像，经特征配准、曝光补偿、拼接缝求解后由 GPU 实时渲染；无显示器时可离屏渲染并将最后一帧保存为图片。

特征检测支持两种模式，可在 `config.json` 中切换：

- `superpoint_lightglue`（默认）：基于 SuperPoint 关键点检测 + LightGlue 匹配，精度更高
- `orb`：基于 ORB 特征检测 + 比值测试互匹配，速度更快

## 功能简介

- 特征配准：支持 SuperPoint+LightGlue 和 ORB 两条路径，RANSAC 求解单应矩阵，支持在线估计与读取标定文件
- 曝光补偿：估计右图相对左图的 RGB 增益与偏置
- 拼接缝求解：Voronoi seam finder 求解重叠区拼接缝
- 多波段融合：拉普拉斯金字塔融合，消除亮度跳变
- 全景输出：支持屏幕显示与离屏渲染

## 目录结构

```text
planar_image_stitching/
├── CMakeLists.txt
├── README.md
├── config.json
├── run_planar.sh
├── lib/
│   └── libplanar_stitcher_core.a
├── include/planar_stitcher/
├── assets/
│   ├── s0_left.jpg             # 回退图片
│   ├── s0_right.jpg
│   └── model/
│       ├── superpoint_360x640.q.onnx
│       └── lightglue_for_superpoint_512_depth9.fp16.onnx
├── output/                     # 运行时生成的图片和标定文件
└── stitcher_test/
    ├── planar_stitcher_main.cpp
    └── calib.xml
```

## 模型文件

SuperPoint + LightGlue 的 ONNX 模型体积较大，不随源码仓库分发，首次使用前需执行一次下载脚本：

```bash
./assets/model/download_model.sh
```

脚本会自动定位自身所在目录，无论从哪个路径调用，模型都会下载到 `assets/model/` 下：

```text
assets/model/
├── download_model.sh
├── superpoint_360x640.q.onnx
└── lightglue_for_superpoint_512_depth9.fp16.onnx
```

模型下载地址：https://archive.spacemit.com/spacemit-ai/model_zoo/vision/

`config.json` 中通过相对路径 `./assets/model/` 引用，该路径相对于**程序运行时的当前工作目录**解析，因此需在本目录下启动（见下方"构建与运行"）。

## MPP 源码

MPP 配置时默认按本目录的相对路径 `../../../multimedia/mpp/` 查找，CMake 会将其解析为绝对路径。如果 MPP 源码位于其他位置，配置时指定路径：

```bash
cmake -S . -B build -DMPP_ROOT=/path/to/mpp
cmake --build build -j8
```

MPP 仓库：https://github.com/spacemit-com/mpp.git

## 环境依赖

**OpenCV**

预编译库链接 opencv-spacemit 4.14，运行时需要兼容版本：

```bash
sudo apt install opencv-spacemit=4.14.0-2bb4
```

安装其他版本可能导致 `symbol lookup error`。

**ONNX Runtime**

预编译库包含 SuperPoint+LightGlue 代码路径，必须链接 ONNX Runtime（即使只使用 ORB 模式）：

```bash
sudo apt install spacemit-onnxruntime
```

安装后确认库文件存在：

```bash
ls /usr/include/onnxruntime_cxx_api.h
ls /usr/lib/libonnxruntime.so /usr/lib/libspacemit_ep.so
```

**其余依赖**

```bash
sudo apt install libx11-dev libegl-dev libgles2
```

## 构建与运行

在本目录构建示例，然后运行：

```bash
cd /path/to/planar_image_stitching  # 替换为实际安装路径
cmake -S . -B build
cmake --build build -j8
./run_planar.sh
```

也可以直接运行可执行文件，配置文件路径可任意指定，但 config.json 中的相对路径（如模型路径 `./assets/model/`）是相对于**程序运行时的当前工作目录**解析的，而不是相对于配置文件本身所在目录。因此推荐在本目录下运行，或使用绝对路径：

```bash
cd /path/to/planar_image_stitching
./build/planar_stitcher                    # 使用默认 config.json
./build/planar_stitcher /other/config.json # 使用其他配置文件（模型路径需为绝对路径）
```

按 Ctrl+C 终止；离屏模式退出时保存最后一帧。

## 配置说明

运行参数统一在本目录的 `config.json` 中配置。

| 配置项 | 类型 | 说明 |
|---|---|---|
| `input.left_image` / `input.right_image` | string | 回退图片路径 |
| `output.image` | string | 拼接结果保存路径 |
| `camera.enable` | bool | `false` 跳过摄像头直接用图片；`true` 时失败自动回退 |
| `camera.width` / `camera.height` | int | 摄像头和输入图片尺寸 |
| `camera.device` | int | VI 设备编号 |
| `camera.timeout_ms` | int | 单帧采集超时（毫秒） |
| `camera.mipi_lanes` / `camera.mipi_mbps` | int | MIPI 参数 |
| `registration.mode` | string | `auto` 在线估计；`file` 读取标定文件 |
| `registration.file` | string | `mode=file` 时读取的标定文件 |
| `registration.save_to` | string | 保存配准结果的路径 |
| `feature_detection.detector` | string | `superpoint_lightglue`（默认）或 `orb` |
| `feature_detection.orb.work_max_width` | int | ORB 检测前的图像缩放宽度 |
| `feature_detection.orb.fast_threshold` | int | ORB FAST 角点阈值 |
| `feature_detection.orb.max_features` | int | ORB 最大特征点数 |
| `feature_detection.superpoint_lightglue.superpoint_model` | string | SP 模型路径 |
| `feature_detection.superpoint_lightglue.lightglue_model` | string | LightGlue 模型路径 |
| `feature_detection.superpoint_lightglue.num_keypoints` | int | 最大关键点数（默认 512） |
| `feature_detection.superpoint_lightglue.nms_radius` | int | NMS 半径 |
| `feature_detection.superpoint_lightglue.remove_borders` | int | 边缘剔除像素数 |
| `feature_detection.superpoint_lightglue.filter_threshold` | float | LightGlue 置信度阈值 |
| `feature_detection.superpoint_lightglue.num_threads` | int | 推理线程数 |
| `feature_detection.superpoint_lightglue.provider` | string | 推理后端（`SpaceMITExecutionProvider` 或 `CPUExecutionProvider`） |
| `ransac.*` | - | RANSAC 参数 |
| `blending.num_bands` | int | 融合波段数，`0` 自动选择 |
| `runtime.frames` | int | 渲染帧数，`0` 持续运行 |
| `runtime.sleep_us` | int | 每帧节流睡眠（微秒） |
| `runtime.force_offscreen` | bool | `true` 强制离屏渲染 |

## 输入与离屏渲染

**输入约束**

- `camera.enable=false` 时读取 `input.left_image` 和 `input.right_image`，图片尺寸须与 `camera.width` × `camera.height` 一致
- `camera.enable=true` 时打开摄像头设备（`camera.device_left`/`device_right`），失败自动回退到图片输入
- 所有输入帧经 MPP JPEG 解码或 VI 捕获后统一转为 NV12 格式送入拼接流水线

**渲染目标**

输出画布固定为 **1920×1080**（横向平面拼接），输入尺寸通过 letterbox 或 crop 适配：

- 输入 ≤ 1920×1080：保持宽高比，居中 letterbox，黑边填充
- 输入 > 1920×1080：保持宽高比，crop 到 1920×1080

**离屏渲染**

- 无显示器或 `DISPLAY` 环境变量为空时，自动切换到 EGL Pbuffer 离屏渲染
- 手动强制：`runtime.force_offscreen=true`
- 离屏模式下仍可通过 `output.image` 保存结果图片

**无桌面登录时的解码回退**

`/dev/video*` 的访问权限通常只授予当前占用 seat 的图形会话（例如 `sddm`）。通过 SSH 或在无桌面登录的情况下运行时，进程没有这些权限，MPP 的 VDEC 硬件解码会失败。

此时程序自动回退到 CPU 解码（OpenCV 解码 JPEG 后上传纹理），渲染仍然走 GPU：

```text
[decoder] VDEC_EnableChn failed (-400); falling back to CPU JPEG decode
```

需要注意：

- 回退只影响 `camera.enable=false` 的图片输入路径
- 帧率会低于硬件解码路径（实测约 22 FPS，硬件路径约 31 FPS）
- 如需硬件解码，请在有桌面登录的会话中运行，或将用户加入 `video` 组并调整设备 ACL

## 切换检测器

检测器通过 `config.json` 切换，修改后直接运行即可（无需重新编译）：

使用 SuperPoint+LightGlue（默认）：

```json
"feature_detection": {
  "detector": "superpoint_lightglue",
  "superpoint_lightglue": {
    "superpoint_model": "./assets/model/superpoint_360x640.q.onnx",
    "lightglue_model": "./assets/model/lightglue_for_superpoint_512_depth9.fp16.onnx"
  }
}
```

使用 ORB：

```json
"feature_detection": {
  "detector": "orb",
  "orb": {
    "work_max_width": 960,
    "fast_threshold": 20,
    "max_features": 2000
  }
}
```

## 典型场景

无摄像头，用图片调试：

```json
"camera": { "enable": false },
"input": { "left_image": "./assets/s0_left.jpg", "right_image": "./assets/s0_right.jpg" }
```

无显示器，离屏保存：

```json
"runtime": { "force_offscreen": true },
"output": { "image": "./output/planar.jpg" }
```

固定帧数测试：

```json
"camera": { "enable": false },
"registration": { "mode": "file", "file": "./stitcher_test/calib.xml" },
"runtime": { "frames": 200, "force_offscreen": true }
```

## 效果图

输入图像（`config.json` 中 `input.left_image` / `input.right_image`）：

<table><tr>
<td><img src="assets/s0_left.jpg" width="480" alt="左路输入"></td>
<td><img src="assets/s0_right.jpg" width="480" alt="右路输入"></td>
</tr></table>

渲染输出（`output.image`）：

<table><tr>
<td><img src="assets/images/planar_output.jpg" width="960" alt="拼接输出"></td>
</tr></table>

## 性能指标

1080P 输出下的实测帧率：

| 指标 | 帧率 (FPS) |
|---|---|
| 屏幕渲染 | 29 |
| 离屏渲染 | 31 |

以上为既有测量值，实际性能取决于输入尺寸、配准模式、融合波段数和目标硬件。
