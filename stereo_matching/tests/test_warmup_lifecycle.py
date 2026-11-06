#!/usr/bin/env python3
"""Compile the production warmup/preprocess/process bodies with CPU DMA/CL fakes.

No GPU, ORT, MPP binary, or installed OpenCL headers are required. The fakes
exercise ownership, copy sizes, cache keys, and exception exits, not GPU math.
"""

from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile


CPP_BRACES = re.compile(
    r'//[^\n]*|/\*.*?\*/|'
    r'R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"|'
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',
    re.DOTALL,
)


def production_function(source, declaration_pattern):
    """Extract one actual function, ignoring braces in comments and literals."""
    declaration = re.search(declaration_pattern, source)
    if declaration is None:
        raise ValueError("production function declaration not found")
    opening = source.index("{", declaration.end())
    depth = 0
    for token in CPP_BRACES.finditer(source, opening):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return source[declaration.start():token.end()]
    raise ValueError("production function closing brace not found")


HARNESS = r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <linux/dma-buf.h>

using U32 = std::uint32_t;
using cl_mem = int;
constexpr int kSyncWrite = 1;
std::mutex g_external_gpu_mutex;
void check(bool invalid, const std::string& message) {
    if (invalid) throw std::runtime_error(message);
}
void require(bool valid, const char* message) { check(!valid, message); }

struct DmaBuffer;
std::map<int, DmaBuffer*> live_buffers;
std::map<int, std::size_t> imported_buffers;
int invalid_release = 0;
int closed_while_imported = 0;
int fail_allocation = 0;
bool fail_sync = false;

struct DmaBuffer {
    DmaBuffer(void*, std::size_t size) : bytes(size) {
        if (fail_allocation && --fail_allocation == 0)
            throw std::runtime_error("injected allocation failure");
        descriptor = 10;
        while (live_buffers.count(descriptor)) ++descriptor;
        live_buffers.emplace(descriptor, this);
    }
    ~DmaBuffer() {
        if (imported_buffers.count(descriptor)) ++closed_while_imported;
        live_buffers.erase(descriptor);
    }
    int fd() const { return descriptor; }
    std::size_t size() const { return bytes.size(); }
    void* data() { return bytes.data(); }
    void sync_start(int) {
        if (std::exchange(fail_sync, false)) throw std::runtime_error("injected sync failure");
    }
    void sync_end(int) {}
    int descriptor;
    std::vector<std::uint8_t> bytes;
};
struct ScopedDmaMapping {
    ScopedDmaMapping(int fd, std::size_t size) : buffer(live_buffers.at(fd)) {
        require(buffer->size() >= size, "source mapping exceeds allocation");
    }
    const void* data() const { return buffer->bytes.data(); }
    DmaBuffer* buffer;
};
struct ScopedDmaAccess {
    ScopedDmaAccess(int fd, std::uint64_t) {
        require(live_buffers.count(fd), "sync of closed DMA fd");
    }
};
struct VideoFrameInfo {
    struct {
        struct { U32 u32Width = 0, u32Height = 0; } stCommFrameInfo;
    } stVdecFrameInfo;
    struct { U32 u32PlaneNum = 0, u32PlaneStride[2]{}, u32TotalSize = 0; } stVFrame;
};
struct StereoPipelineOptions { int width = 4, height = 2; };
struct StereoPipelineSlot {
    cl_mem cl_left_tensor{}, cl_right_tensor{}, cl_left_preview{}, cl_right_preview{};
    std::unique_ptr<DmaBuffer> left_preview_dma;
};
struct PreprocessedStereoFrame {
    std::vector<std::uint8_t> left_xrgb;
    int image_w{}, image_h{}, depth_w{}, depth_h{}, pad_left{}, pad_top{}, dma_fd{};
    double opencl_ms{};
};
std::vector<std::uint8_t> read_preview(DmaBuffer&, int, int) { return {}; }
struct FakeCl {
    bool fail_import = false, fail_run = false;
    int imports = 0, releases = 0, runs = 0;
    cl_mem import_input_dmabuf_cached(int fd, std::size_t size) {
        if (std::exchange(fail_import, false)) throw std::runtime_error("injected import failure");
        auto result = imported_buffers.emplace(fd, size);
        if (result.second) ++imports;
        require(result.first->second == size, "cached input size changed");
        require(live_buffers.at(fd)->size() == size, "incorrect import size");
        return fd;
    }
    void release_input_dmabuf_cached(int fd) {
        if (!live_buffers.count(fd)) ++invalid_release;
        releases += imported_buffers.erase(fd);
    }
    double run(cl_mem input, cl_mem, cl_mem, cl_mem, cl_mem,
               const VideoFrameInfo& frame, const StereoPipelineOptions&, int, int, bool,
               int, int, int, int) {
        if (std::exchange(fail_run, false)) throw std::runtime_error("injected run failure");
        const auto required = std::size_t(frame.stVFrame.u32PlaneStride[0]) *
                                  frame.stVdecFrameInfo.stCommFrameInfo.u32Height +
                              std::size_t(frame.stVFrame.u32PlaneStride[1]) *
                                  (frame.stVdecFrameInfo.stCommFrameInfo.u32Height / 2);
        require(imported_buffers.at(input) == required, "stride/import size mismatch");
        ++runs;
        return 0.25;
    }
};
struct Las2Handle {
    bool started = true, produce_rgb888 = false, nhwc = false;
    StereoPipelineOptions opt;
    int model_w = 4, model_h = 2;
    void* allocator = nullptr;
    std::unique_ptr<DmaBuffer> external_input_dma;
    std::unique_ptr<FakeCl> cl = std::make_unique<FakeCl>();
    std::vector<std::unique_ptr<StereoPipelineSlot>> slots;
    Las2Handle() { slots.push_back(std::make_unique<StereoPipelineSlot>()); }
};
namespace las2 {
struct Nv12DmaFrame {
    int dma_fd = -1, width{}, height{}, y_stride{}, uv_stride{}, frame_id{};
    std::size_t total_size{};
};
struct Frame {};
class ExternalStereo {
public:
    struct Impl {
        std::shared_ptr<Las2Handle> handle = std::make_shared<Las2Handle>();
        std::string error;
        double last_opencl_ms = 0;
    };
    std::unique_ptr<Impl> impl_ = std::make_unique<Impl>();
    bool process(const Nv12DmaFrame&, Frame&);
    bool warmup(int);
};
}
void infer_external_frame(Las2Handle*, StereoPipelineSlot&, PreprocessedStereoFrame&&, int, las2::Frame*) {}

// PRODUCTION_FUNCTIONS

las2::Nv12DmaFrame live_frame(DmaBuffer& input) {
    las2::Nv12DmaFrame frame;
    frame.dma_fd = input.fd();
    frame.width = 4;
    frame.height = 2;
    frame.y_stride = 8;
    frame.uv_stride = 12;
    frame.total_size = 28;
    return frame;
}
void dispose(las2::ExternalStereo& stereo) {
    auto& handle = *stereo.impl_->handle;
    if (handle.external_input_dma) {
        handle.cl->release_input_dmabuf_cached(handle.external_input_dma->fd());
        handle.external_input_dma.reset();
    }
    require(imported_buffers.empty(), "import leaked");
    require(invalid_release == 0, "released cache after fd close or wrong fd");
    require(closed_while_imported == 0, "DMA fd closed while import remained");
}
void success_tests() {
    las2::ExternalStereo stereo;
    auto& handle = *stereo.impl_->handle;
    require(stereo.warmup(2), "initial warmup failed");
    require(!handle.external_input_dma && imported_buffers.empty(), "warmup fixed live layout");
    require(handle.cl->imports == 1 && handle.cl->releases == 1, "wrong warmup cache key");
    require(live_buffers.empty(), "warmup DMA leaked");
    DmaBuffer input(nullptr, 28);
    std::fill(input.bytes.begin(), input.bytes.end(), 73);
    auto frame = live_frame(input);
    las2::Frame output;
    require(stereo.process(frame, output), "aligned real frame rejected after warmup");
    auto* original = handle.external_input_dma.get();
    require(original->bytes == input.bytes, "padded planes were not copied completely");
    const int live_fd = original->fd();
    require(stereo.process(frame, output), "repeated aligned frame failed");
    const int imports = handle.cl->imports;
    require(stereo.warmup(3), "warmup with live allocation failed");
    require(handle.external_input_dma.get() == original, "live allocation not restored");
    require(imported_buffers.size() == 1 && imported_buffers.count(live_fd), "live cache lost");
    require(stereo.process(frame, output), "live stream failed after repeated warmup");
    require(handle.cl->imports == imports + 1, "real stream was unnecessarily reimported");
    require(stereo.warmup(0), "zero warmup failed");
    require(!stereo.warmup(-1) && !stereo.warmup(101), "invalid iterations accepted");
    require(handle.external_input_dma.get() == original, "validation changed live allocation");
    dispose(stereo);
}
void failure_tests(bool existing_live) {
    for (const std::string failure : {"import", "run", "sync", "source_alloc", "copy_alloc"}) {
        las2::ExternalStereo stereo;
        auto& handle = *stereo.impl_->handle;
        DmaBuffer input(nullptr, 28);
        las2::Frame output;
        if (existing_live) require(stereo.process(live_frame(input), output), "setup failed");
        auto* original = handle.external_input_dma.get();
        const auto original_cache = imported_buffers;
        const auto original_live = live_buffers.size();
        handle.cl->fail_import = failure == "import";
        handle.cl->fail_run = failure == "run";
        fail_sync = failure == "sync";
        fail_allocation = failure == "source_alloc" ? 1 : failure == "copy_alloc" ? 2 : 0;
        require(!stereo.warmup(2), "injected warmup failure not reported");
        require(!stereo.impl_->error.empty(), "failure diagnostic lost");
        require(handle.external_input_dma.get() == original, "failure lost live allocation");
        require(imported_buffers == original_cache, "failure leaked/removed imports");
        require(live_buffers.size() == original_live, "failure leaked DMA allocation");
        require(stereo.process(live_frame(input), output), "real frame rejected after failure");
        dispose(stereo);
    }
}
int main() {
    try {
        success_tests();
        failure_tests(false);
        failure_tests(true);
        require(live_buffers.empty(), "test left DMA allocations alive");
        las2::ExternalStereo uninitialized;
        uninitialized.impl_->handle->started = false;
        require(!uninitialized.warmup(1), "uninitialized warmup accepted");
        std::cout << "PASS production warmup: padded planes, repeated/live warmup, "
                     "zero/invalid/uninitialized and ten injected failure paths\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
"""


def main():
    source = (Path(__file__).resolve().parents[1] /
              "src/stereo_pipeline.cpp").read_text()
    functions = [
        production_function(
            source, r"\bPreprocessedStereoFrame\s+preprocess_external_nv12\s*\("
        ),
        production_function(
            source, r"\bbool\s+las2::ExternalStereo::process\s*\("
        ),
        production_function(
            source, r"\bbool\s+las2::ExternalStereo::warmup\s*\("
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="las2-warmup-test-") as directory:
        temporary = Path(directory)
        unit = temporary / "test.cpp"
        binary = temporary / "test"
        unit.write_text(HARNESS.replace("// PRODUCTION_FUNCTIONS", "\n".join(functions)))
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                       ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                        str(unit), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
