# 鱼眼图像拼接 (Fisheye Image Stitching)

基于多路鱼眼相机的图像拼接模块，用于生成车辆/机器人周视全景图像（Surround View）。程序从摄像头（或回退图片）读取四路图像，经碗面拼接后实时渲染 3D 全景与 2D 展开视图；无显示器时可离屏渲染并将最后一帧保存为图片。

## 功能简介

- 鱼眼相机标定：内参、畸变系数求解
- 去畸变与投影变换：将鱼眼图像映射到统一的 2D 鸟瞰图以及 3D 图
- 车模文件渲染：加载车模 3D 文件，采用 GPU 虚拟视点渲染
- 拼接融合：重叠区域的拼接缝平滑过渡处理
- 全景输出：支持屏幕显示与离屏渲染

## 目录结构

```text
fisheye_image_stitching/
├── CMakeLists.txt
├── README.md
├── config.json             # 运行配置入口
├── run_live_vi.sh
├── lib/                    # 图像拼接库
│   ├── libsvrender.so
│   ├── libsvmcalibrate.so
│   └── libsvmparam.so
├── include/                # 图像拼接头文件及 glm
└── sv_avm_test/
    ├── _aParam.xml
    ├── sv_avm_render_main_test.cpp
    └── res/                # 车模与四路回退图片
```

## MPP 源码

MPP 以源码方式编译，配置时默认按本目录的相对路径 `../../../multimedia/mpp/` 查找，CMake 会将其解析为绝对路径。如果 MPP 源码位于其他位置，配置编译时指定路径：

```bash
cmake -S . -B build -DMPP_ROOT=/path/to/mpp
cmake --build build -j8
```

MPP 仓库：https://github.com/spacemit-com/mpp.git

## 环境依赖

**OpenCV（spacemit 定制版）**

图像拼接库链接 opencv-spacemit 4.14，运行时需要兼容版本：

```bash
sudo apt install opencv-spacemit=4.14.0-2bb4
```

版本不兼容时可能出现 `symbol lookup error`。

**其余依赖**

```bash
sudo apt install libgoogle-glog-dev libassimp-dev \
                 libx11-dev libegl-dev libgles2
```

## 构建与运行

在本目录构建示例，然后使用脚本运行：

```bash
cmake -S . -B build
cmake --build build -j8
./run_live_vi.sh
```

脚本会自动进入 `build/` 运行可执行文件。可通过命令行参数临时覆盖 `config.json` 中的配置：

```bash
./run_live_vi.sh --frames 100
```

按 Ctrl+C 终止；若开启离屏渲染，退出时自动保存最后一帧。

## 配置说明

所有运行参数统一在本目录的 `config.json` 中配置，`build/` 下不需要配置文件副本。

| 配置项 | 类型 | 说明 |
|---|---|---|
| `live_vi` | bool | 启用摄像头输入；不可用时自动回退为图片 |
| `frames` | int | 渲染帧数，`0` 表示持续运行直到 Ctrl+C |
| `sleep_us` | int | 每帧节流睡眠（微秒），`0` 表示不限速 |
| `grid_subdiv` | int | 碗面网格细分数，可选 `90 / 180 / 360 / 720` |
| `live_vi_dev` | int | VI 设备编号 |
| `live_vi_width` / `live_vi_height` | int | 摄像头分辨率（像素） |
| `live_vi_timeout_ms` | int | 单帧采集超时（毫秒） |
| `live_vi_mipi_lanes` | int | MIPI 通道数 |
| `live_vi_mbps` | int | MIPI 带宽（Mbps） |
| `use_fallback_image` | bool | `true` 跳过摄像头，强制使用图片 |
| `fallback_image_dir` | string | 回退图片目录，默认 `sv_avm_test/res` |
| `calibration_image_save_dir` | string | 保存四路标定图片的目录 |
| `force_offscreen` | bool | `true` 强制离屏渲染 |
| `offscreen_output_path` | string | 最后一帧的输出路径 |
| `offscreen_width` / `offscreen_height` | int | 离屏输出尺寸 |

## 输入源与回退逻辑

- `live_vi: true`：先尝试摄像头，失败时读取 `fallback_image_dir` 中的图片。
- `live_vi: false` 或 `use_fallback_image: true`：跳过摄像头，直接读取图片。

图片目录应包含 `imagech0.jpg` 至 `imagech3.jpg`，分别对应左、右、前、后四路摄像头。

## 显示器与离屏渲染

- 有显示器时渲染到屏幕；`offscreen_output_path` 被忽略。
- 无显示器时自动切换为 EGL Pbuffer 离屏渲染，退出时保存最后一帧；输出目录不存在时自动创建。

关闭显示器电源不等于无显示器。X Server 仍在运行时，程序会先尝试创建窗口表面；需要主动离屏时将 `force_offscreen` 设为 `true`。

## 典型场景

摄像头与显示器：保持默认配置，运行 `./run_live_vi.sh`。

无摄像头，用图片调试：

```json
"use_fallback_image": true,
"fallback_image_dir": "sv_avm_test/res"
```

无显示器，离屏保存：

```json
"force_offscreen": true,
"offscreen_output_path": "output/render.jpg"
```

固定帧数测试：

```json
"frames": 200,
"use_fallback_image": true,
"force_offscreen": true,
"offscreen_output_path": "output/test_frame.jpg"
```

## 参数文件与注意事项

- `_aParam.xml`：相机内外参及畸变系数。
- `concept_BUS cycles.dae`：车模文件。
- `imagech0.jpg` 至 `imagech3.jpg`：四路 1280x720 鱼眼图片。
- 每个 Apriltag 应完整处于相机重叠区域；每路相机需拍摄到两个 Apriltag，水平 FOV 约 180 度。

## 效果图

输入图像（`fallback_image_dir` 目录下的四路鱼眼图，即 `sv_avm_test/res/imagech0.jpg` 至 `imagech3.jpg`）：

<table><tr>
<td><img src="sv_avm_test/res/imagech0.jpg" width="320" alt="ch0"></td>
<td><img src="sv_avm_test/res/imagech1.jpg" width="320" alt="ch1"></td>
</tr><tr>
<td><img src="sv_avm_test/res/imagech2.jpg" width="320" alt="ch2"></td>
<td><img src="sv_avm_test/res/imagech3.jpg" width="320" alt="ch3"></td>
</tr></table>

渲染输出（`offscreen_output_path`）：

<table><tr>
<td><img src="assets/images/avm_output.jpg" width="960" alt="AVM 渲染输出"></td>
</tr></table>

## 性能指标

既有测试条件：4 路 1280x720 鱼眼输入，Release 构建，1920x1080 输出。

| 指标 | 2D 鸟瞰 + 3D 渲染 |
|---|---|
| 离屏帧率 (FPS) | 110 |
| 屏幕帧率 (FPS) | 60 |

以上是既有测量值，不代表当前环境的实测性能。
