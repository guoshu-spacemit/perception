# Stereo Matching

SpacemiT K3 LAS2 双目视差推理库。

## 目录

```text
stereo_matching/
├── include/
│   ├── stereo_matching.h          # tensor API
│   └── stereo_matching/
│       └── pipeline.h            # Camera / ExternalStereo API
├── lib/libstereo_matching.so
├── model/LAS2_M_288x384.fp16.onnx  # tensor API 模型
├── demo/demo.cpp                 # SDK 图片输入示例
├── src/
│   ├── stereo_pipeline.cpp        # 公共 API 实现与槽位生命周期
│   ├── stereo_pipeline_internal.h # 私有实现声明
│   ├── stereo_runtime.h          # DMA、ORT 推理及运行辅助类型
│   ├── stereo_preprocessing.h    # 标定校正、预处理及 OpenCL 处理器
│   └── ...                       # 校正、缓存与计时辅助头
├── cmake/
├── tests/
├── thirdparty/dmabufheap/
├── licenses/LiteAnyStereo-MIT.txt
├── CMakeLists.txt
├── NOTICE
├── UPSTREAM.json
├── RUNTIME_CONTRACT.json
├── README.md
└── run.sh
```

## 运行 demo

左右图作为外部输入：

```bash
STEREO_LEFT=/path/Motorcycle-perfect/im0.png \
STEREO_RIGHT=/path/Motorcycle-perfect/im1.png \
./run.sh
```

默认使用 `model/LAS2_M_288x384.fp16.onnx`，输入宽×高为
`384×288`。默认配置是 8 个 A100 核和 4 个实例。可通过以下环境变量
调整：

- `STEREO_MODEL`
- `STEREO_A100_CORES`
- `STEREO_INSTANCES`
- `STEREO_OUTPUT`

## API

```cpp
#include <stereo_matching.h>

spacemit::stereo::Config config;
config.model_path = "./model/LAS2_M_288x384.fp16.onnx";
config.a100_core_count = 8;
config.instance_count = 8;

spacemit::stereo::StereoMatcher matcher;
if (!matcher.initialize(config)) {
    // matcher.last_error()
}
```

客户程序只需增加：

```text
-I<stereo_matching>/include
-L<stereo_matching>/lib -lstereo_matching
```

`model_info()` 返回模型输入尺寸和布局。`infer()` 接收 RGB float32
输入，数值范围为 `0–255`。输入输出内存在 `infer()` 返回前必须保持
有效。

## 板端实测

K3，`384×288`，预热 10 次，纯推理：

| 配置 | 并发 | 平均延迟 | 吞吐 |
|---|---:|---:|---:|
| 4 核 / 1 实例 | 1 | 50.04 ms | 19.98 FPS |
| 8 核 / 4 实例 | 4 | 92.44 ms | 43.11 FPS |
| 8 核 / 8 实例 | 8 | 160.03 ms | 49.70 FPS |

8 核 / 8 实例连续 800 次：P50 `159.87 ms`，P99 `165.17 ms`。

上述 tensor API 的 OpenCV 仅用于 `demo/demo.cpp` 读取和保存图片；
下述 ExternalStereo 源码还使用 OpenCV 生成标定校正映射。


## Robot SDK 安装接口

该目录现在提供 CMake package `stereo_matching`，普通推理 target 为
`stereo_matching::stereo_matching`。设置 `STEREO_MATCHING_BUILD_DEMO=ON` 可构建图片 demo。

定位重建应用使用 NV12 输入、校正、深度和 RGB 输出契约，要求
`find_package(stereo_matching CONFIG REQUIRED COMPONENTS external_input)`。
公共接口位于 `include/stereo_matching/pipeline.h`，通过
`#include <stereo_matching/pipeline.h>` 使用，包含 Camera 和 ExternalStereo。
`src/stereo_pipeline.cpp` 实现公共接口，私有运行类型位于
`src/stereo_runtime.h`，标定校正和预处理处理器位于
`src/stereo_preprocessing.h`。这些私有头不作为公共 API 安装。

算法上游为 [Lite Any Stereo](https://github.com/TomTomTommi/LiteAnyStereo)，
采用 MIT 许可证。本目录的 K3 C++ 实现是 SDK 适配，分别记录算法来源和
平台集成贡献；源码结构与接口不等同于算法仓库的官方参考实现。
上游、论文引用及适配范围见 [UPSTREAM.json](UPSTREAM.json)，
作者与第三方署名见 [NOTICE](NOTICE)，原始 MIT 许可证见
[licenses/LiteAnyStereo-MIT.txt](licenses/LiteAnyStereo-MIT.txt)。

### 从源码构建 ExternalStereo

在 K3 上准备 SDK 的 OpenCV、OpenCL、yaml-cpp 开发依赖后，指定
外部匹配的 SpaceMIT ORT/EP 工件目录。目录需包含 `include/` 和 `lib/`；
版本、头文件和运行库 SHA-256 见 `RUNTIME_CONTRACT.json`。

```bash
cmake -S "$SROBOTIS_ROOT/components/perception/stereo_matching" \
  -B /tmp/las2-external-build \
  -DCMAKE_INSTALL_PREFIX=/path/to/product-prefix \
  -DSTEREO_MATCHING_BUILD_EXTERNAL_INPUT=ON \
  -DONNXRUNTIME_ROOT=/path/to/runtime/onnxruntime
cmake --build /tmp/las2-external-build -j4
cmake --install /tmp/las2-external-build
```

通过 SDK 构建入口时，将后两个 `-D` 参数逐行加入
`SROBOTIS_NATIVE_CMAKE_EXTRA_ARGS`。MPP 头文件默认复用当前 SDK
`components/multimedia/mpp`；独立构建可设置 `MPP_ROOT`。DMA heap 分配器
使用 `thirdparty/dmabufheap` 内最小 Apache-2.0 源码快照，也可通过
`DMABUFHEAP_ROOT` 指定兼容源码。默认保留 RVV 校正；
`LAS2_ENABLE_RVV_RECTIFICATION=OFF` 可用于标量对照。

构建后安装 `libstereo_matching_pipeline.so` 并导出
`stereo_matching::external_input`。源码后端默认启用
`STEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME`：从显式选择的
`ONNXRUNTIME_ROOT/lib` 将四类 ORT/EP/TCM 运行库及 ABI 符号链接安装到
SDK `lib/`，许可证和工件清单安装到 `share/stereo_matching/`。
后端以相对 `$ORIGIN` 查找这些依赖，安装包不记录构建机器的 ORT 绝对路径。
这些文件只进入构建安装/部署工件，不复制到 SDK 源码，也不安装私有开发头。
可在安装前缀目录运行
`sha256sum -c share/stereo_matching/external_input/runtime/DEPLOYED_RUNTIME_SHA256SUMS`
核验实际部署的文件；原工件清单保留为 `SOURCE_SHA256SUMS`。
如设 `STEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME=OFF`，部署方需自行提供
完全匹配的运行库，并在链接与运行阶段配置搜索路径。
也可用原有 `STEREO_MATCHING_EXTERNAL_INPUT_LIBRARY`
显式安装匹配的预编译后端，不能同时启用源码构建。预编译入口使用工具链的
`readelf` 检查 ELF；安装时同时保留标准库名和版本化 SONAME，输入可以是实际
库文件或其符号链接。无 SONAME 的库也可使用；带 SONAME 的库必须属于
`libstereo_matching_pipeline.so` 名称族。

ExternalStereo 必须使用其匹配的 FP16 模型：
`LAS2_M_288x384.fp16.onnx`，SHA-256 为
`aa4f22f0f2433dc317ae3fa028549b9e228200c117d95121e19cfa0f1615145a`。
它接受外部 ImageNet 归一化的 RGB，输入/输出张量为 FP16。
本目录 `model/` 下同名模型属于原 tensor API，SHA-256 为
`daef9f5c3e2181116bde4d045dcb8338fe519fc46ec5e82fc41961d590c3ba9a`；
两者不能因文件名相同而互换。应用通过配置提供模型和 Kalibr 标定路径。
可显式设置 `STEREO_MATCHING_EXTERNAL_INPUT_MODEL=/path/to/matched/model.onnx`
将其部署到 `share/stereo_matching/external_input/model/`；默认不安装该外部模型，
也不覆盖原 tensor API 的 `share/stereo_matching/model/`。

输出保持 float32 米制深度、与深度对齐的未归一化 RGB888、内参与
`left_from_depth`、帧序号和准备槽位的同步/生命周期语义。曝光时间关联
继续由应用维护；后端不创建时间偏移。MPP 仅供编译复用帧类型头文件，
ExternalStereo 不导入或传输 VB token，也不链接 MPP 运行库；相机/解码
的 MPP ABI 和整条 DMA-BUF 路径仍需板端联调。

非 RISC-V 目标默认不强制要求 TCM；配套工件包含 TCM 时仍会一并安装。
若所选 EP 在非 RISC-V 上通过动态加载使用 TCM，应显式设置
`STEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM=ON`，缺失时会拒绝打包。
RISC-V 的 TCM 必需性保持。

### 回归检查

只测试资源生命周期时，在开发主机执行：

```bash
cmake -S . -B build-lifecycle -DSTEREO_MATCHING_BUILD_LIFECYCLE_TESTS=ON
ctest --test-dir build-lifecycle -L lifecycle --output-on-failure
```

该检查需要 Python 3 和 C++17 编译器，不需要 OpenCL、ORT 或相机。测试从
生产文件提取实际预处理、process、warmup 函数进行编译，使用 DMA/OpenCL
替身注入失败，检查预热后接收不同对齐步长、重复预热和失败退出时的
私有输入/导入缓存生命周期。预热的紧凑 NV12 布局不约束真实帧；已建立的
真实流布局在预热前后保持。这些替身验证资源所有权和布局控制流，
不验证 MPP ABI、驱动或 GPU 数值。

主机可运行 `python3 tests/test_runtime_packaging.py`，用临时 ELF 库验证
运行库/链接/许可证/SHA 安装、模型隔离，以及移动安装前缀后的 consumer
真实链接、执行和 TCM 动态加载；该检查需要 C 编译器、CMake 和 readelf。
`python3 tests/test_prebuilt_backend.py` 使用实际包构建规则检查无 SONAME、
版本化 SONAME 和符号链接形式的预编译库安装，以及安装后的 consumer 链接运行。

`STEREO_MATCHING_BUILD_TESTS=ON` 构建 CPU 校正/缓存/计时检查及显式 GPU
探针。普通 `ctest --test-dir <build> -L cpu --output-on-failure` 不运行
GPU 探针；OpenCL GPU 数值回归须在目标板单独运行
`test_imagenet_opencl`、`test_fp16_opencl`、`test_preview_opencl`，参数为
本目录 `src/stereo_preprocessing.h`。CPU 检查通过不代表推理、DMA heap、
MPP ABI 或持续深度吞吐已经验证。
