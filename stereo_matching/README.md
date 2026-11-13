# Stereo Matching

面向 SpacemiT K3 的 LAS2 双目匹配库。从左右目图像推理视差，可用于生成视差图，
也可结合相机标定计算米制深度，供 VSLAM 建图使用。

## 目录结构

```text
stereo_matching/
├── include/
│   ├── stereo_matching.h           # 图片 demo 使用的 tensor API
│   └── stereo_pipeline.h           # 相机和 ExternalStereo 接口
├── lib/
│   ├── libstereo_matching.so       # 视差推理库
│   └── libstereo_matching_runtime.so
├── demo/demo.cpp                   # 读取左右图并保存视差图
├── src/                            # 双目校正及深度处理
├── cmake/                          # 构建及安装配置
├── thirdparty/                     # 第三方编译依赖
├── model/                         # 运行 demo 前在本地创建
├── CMakeLists.txt
├── package.xml
├── NOTICE
├── README.md
└── run.sh
```

## 运行图片 demo

在 K3 上准备 C++17 编译器、CMake 和 OpenCV 开发包。demo 读取一对左右图，
按模型输入尺寸 `384×288` 进行推理，输出彩色视差图。

从本目录执行以下命令，将原 demo 的配套模型下载到本地 `model/`。

```bash
model_path=model/LAS2_M_288x384.fp16.onnx
mkdir -p model
curl -fL -o "$model_path" \
  https://raw.githubusercontent.com/spacemit-robotics/perception/5cfa1ace513a922cdec83bb426194b32926542f7/stereo_matching/model/LAS2_M_288x384.fp16.onnx &&
  printf '%s  %s\n' daef9f5c3e2181116bde4d045dcb8338fe519fc46ec5e82fc41961d590c3ba9a "$model_path" |
    sha256sum -c -
```

下面用 Middlebury 的 `Motorcycle-perfect` 图像对运行 demo。

```bash
STEREO_MODEL=model/LAS2_M_288x384.fp16.onnx \
STEREO_LEFT=/path/Motorcycle-perfect/im0.png \
STEREO_RIGHT=/path/Motorcycle-perfect/im1.png \
./run.sh
```

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `STEREO_MODEL` | tensor 模型路径 | 必填 |
| `STEREO_LEFT` / `STEREO_RIGHT` | 左右图路径 | 必填 |
| `STEREO_A100_CORES` | A100 核数 | 8 |
| `STEREO_INSTANCES` | 推理实例数 | 4 |
| `STEREO_OUTPUT` | 视差图输出路径 | `/tmp/stereo_matching_output/disparity.png` |

## C++ 视差接口

```cpp
#include <stereo_matching.h>

spacemit::stereo::Config config;
config.model_path = "./model/LAS2_M_288x384.fp16.onnx";
config.a100_core_count = 8;
config.instance_count = 4;

spacemit::stereo::StereoMatcher matcher;
if (!matcher.initialize(config)) {
    // 通过 matcher.last_error() 获取错误信息。
}
```

使用 CMake 时链接 `stereo_matching::stereo_matching`，或将 `include/` 加入
头文件搜索路径并链接 `lib/libstereo_matching.so`。`model_info()` 返回模型输入
尺寸与布局。`infer()` 接收数值范围为 0–255 的 RGB float32 张量，输入和输出
缓冲在 `infer()` 返回前须保持有效。

## 图片接口板端实测

K3，输入 `384×288`，预热 10 次，计时仅包含推理。

| 配置 | 并发 | 平均延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| 4 核 / 1 实例 | 1 | 50.04 ms | 19.98 FPS |
| 8 核 / 4 实例 | 4 | 92.44 ms | 43.11 FPS |
| 8 核 / 8 实例 | 8 | 160.03 ms | 49.70 FPS |

8 核 / 8 实例连续运行 800 次，P50 为 159.87 ms，P99 为 165.17 ms。

## ExternalStereo 深度接口

接收双目 NV12 DMA-BUF 和 Kalibr 标定，输出米制深度与对齐的左目 RGB。C++ 应用
使用 `<stereo_pipeline.h>` 并链接 `stereo_matching::external_input`。单独从源码构建时，
在支持 RVV 的 K3 工具链下开启 `-DSTEREO_MATCHING_BUILD_EXTERNAL_INPUT=ON`。

ExternalStereo 使用另一份同名模型。VSLAM 的模型下载和启动方式见
[spacemit-vslam README](https://github.com/spacemit-robotics/ros2_spacemit_vslam/blob/main/README.md)。

算法来源和许可见 [NOTICE](NOTICE)。
