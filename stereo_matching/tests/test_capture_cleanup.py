#!/usr/bin/env python3
"""Fault-inject the actual V4L2 capture class without camera/MPP/CL runtimes.

Only the capture class is extracted from its production translation unit. The
fixture supplies syscall fakes and the small frame/options types it references;
it does not duplicate the capture implementation or validate the MPP ABI.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile


COMPONENT = Path(__file__).resolve().parents[1]

PREAMBLE = r"""
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

bool fail_next_allocation = false;

void* operator new(std::size_t size) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (void* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

using U32 = std::uint32_t;
struct VideoFrameInfo {
    U32 u32Idx;
    struct { U32 u32Width; U32 u32Height; } stCommFrameInfo;
    struct {
        U32 u32PlaneStride[3];
        U32 u32TotalSize;
        U32 u32Fd[3];
    } stVFrame;
};
struct UsbOptions {
    std::string dev = "/fixture/video";
    int width = 64;
    int height = 48;
    int fps = 30;
};

struct Mock {
    int fail_at = 0;
    int calls = 0;
    int next_fd = 11;
    int stream_on_calls = 0;
    int stream_off_calls = 0;
    int qbuf_calls = 0;
    int duplicate_closes = 0;
    int invalid_unmaps = 0;
    bool bad_capability = false;
    bool bad_geometry = false;
    bool few_buffers = false;
    bool allocation_failure = false;
    bool zero_mapping = false;
    bool fail_stream_off = false;
    bool streaming = false;
    std::set<int> descriptors;
    std::map<void*, std::size_t> mappings;

    bool fail() {
        ++calls;
        if (calls != fail_at) return false;
        errno = EIO;
        return true;
    }
} mock;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int fake_open(const char*, int) {
    if (mock.fail()) return -1;
    mock.descriptors.insert(10);
    return 10;
}

int fake_close(int fd) {
    if (!mock.descriptors.erase(fd)) ++mock.duplicate_closes;
    if (fd == 10) mock.streaming = false;
    return 0;
}

int fake_ioctl(int fd, unsigned long request, void* argument) {
    expect(fd == 10 && mock.descriptors.count(fd), "ioctl on closed camera");
    if (request == VIDIOC_STREAMOFF) {
        ++mock.stream_off_calls;
        if (mock.fail_stream_off) {
            errno = EIO;
            return -1;
        }
        mock.streaming = false;
        return 0;
    }
    if (request == VIDIOC_DQBUF) {
        static_cast<v4l2_buffer*>(argument)->index = 0;
        return 0;
    }
    if (mock.fail()) return -1;
    if (request == VIDIOC_QUERYCAP) {
        auto* value = static_cast<v4l2_capability*>(argument);
        value->device_caps = mock.bad_capability ? 0 :
            V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
    } else if (request == VIDIOC_S_FMT) {
        auto* value = static_cast<v4l2_format*>(argument);
        if (mock.bad_geometry) ++value->fmt.pix.width;
        value->fmt.pix.bytesperline = value->fmt.pix.width * 2;
    } else if (request == VIDIOC_REQBUFS) {
        static_cast<v4l2_requestbuffers*>(argument)->count = mock.few_buffers ? 2 : 6;
        if (mock.allocation_failure) fail_next_allocation = true;
    } else if (request == VIDIOC_QUERYBUF) {
        auto* value = static_cast<v4l2_buffer*>(argument);
        value->length = 8192;
        value->m.offset = value->index * value->length;
    } else if (request == VIDIOC_EXPBUF) {
        auto* value = static_cast<v4l2_exportbuffer*>(argument);
        value->fd = mock.next_fd++;
        mock.descriptors.insert(value->fd);
    } else if (request == VIDIOC_QBUF) {
        ++mock.qbuf_calls;
    } else if (request == VIDIOC_STREAMON) {
        ++mock.stream_on_calls;
        mock.streaming = true;
    } else {
        expect(request == VIDIOC_S_PARM, "unexpected capture ioctl");
    }
    return 0;
}

void* fake_mmap(void*, std::size_t bytes, int, int, int fd, off_t offset) {
    expect(fd == 10 && mock.descriptors.count(fd), "mmap on closed camera");
    if (mock.fail()) return MAP_FAILED;
    void* address = mock.zero_mapping && offset == 0 ? nullptr :
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(offset) + 4096);
    expect(mock.mappings.emplace(address, bytes).second, "duplicate fixture mapping");
    return address;
}

int fake_munmap(void* address, std::size_t bytes) {
    auto found = mock.mappings.find(address);
    if (found == mock.mappings.end() || found->second != bytes) {
        ++mock.invalid_unmaps;
    } else {
        mock.mappings.erase(found);
    }
    return 0;
}

int fake_poll(pollfd*, nfds_t, int) { return 1; }

#define open fake_open
#define close fake_close
#define ioctl fake_ioctl
#define mmap fake_mmap
#define munmap fake_munmap
#define poll fake_poll
"""

TESTS = r"""
#undef open
#undef close
#undef ioctl
#undef mmap
#undef munmap
#undef poll

static_assert(std::is_nothrow_destructible_v<DirectV4l2YuyvCapture>);
static_assert(!std::is_copy_constructible_v<DirectV4l2YuyvCapture>);
static_assert(!std::is_copy_assignable_v<DirectV4l2YuyvCapture>);

void assert_released() {
    expect(mock.descriptors.empty(), "capture leaked a descriptor");
    expect(mock.mappings.empty(), "capture leaked an mmap region");
    expect(!mock.streaming, "camera still streaming after close");
    expect(mock.duplicate_closes == 0, "descriptor released twice");
    expect(mock.invalid_unmaps == 0, "mapping released incorrectly");
}

void expect_constructor_failure() {
    bool threw = false;
    try {
        DirectV4l2YuyvCapture capture(UsbOptions{});
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "fault did not propagate from constructor");
    assert_released();
    expect(mock.stream_on_calls == 0 && mock.stream_off_calls == 0,
           "failed constructor used unexpected streaming state");
}

int main() {
    try {
        int acquisition_calls = 0;
        {
            DirectV4l2YuyvCapture capture(UsbOptions{});
            acquisition_calls = mock.calls;
            expect(mock.descriptors.size() == 7 && mock.mappings.size() == 6,
                   "successful constructor did not retain every resource");
            expect(mock.streaming, "successful constructor did not start streaming");
            VideoFrameInfo frame{};
            int fd = -1;
            double capture_ms = -1;
            expect(capture.get_frame(frame, fd, capture_ms), "frame capture failed");
            expect(frame.stCommFrameInfo.u32Width == 64 &&
                   frame.stCommFrameInfo.u32Height == 48 &&
                   frame.stVFrame.u32PlaneStride[0] == 128 &&
                   frame.stVFrame.u32TotalSize == 8192 && fd == 11,
                   "capture frame metadata changed");
            capture.release_frame(frame);
            expect(mock.qbuf_calls == 7, "dequeued buffer was not returned");
        }
        assert_released();
        expect(mock.stream_on_calls == 1 && mock.stream_off_calls == 1,
               "successful destructor did not stop streaming exactly once");
        expect(acquisition_calls == 30, "capture acquisition sequence changed");

        for (int repeat = 0; repeat < 12; ++repeat) {
            for (int fail_at = 1; fail_at <= acquisition_calls; ++fail_at) {
                mock = Mock{};
                mock.fail_at = fail_at;
                mock.zero_mapping = true;
                expect_constructor_failure();
            }
        }

        for (int mode = 0; mode < 4; ++mode) {
            mock = Mock{};
            mock.bad_capability = mode == 0;
            mock.bad_geometry = mode == 1;
            mock.few_buffers = mode == 2;
            mock.allocation_failure = mode == 3;
            expect_constructor_failure();
        }

        for (bool fail_stream_off : {false, true}) {
            mock = Mock{};
            mock.zero_mapping = true;
            mock.fail_stream_off = fail_stream_off;
            {
                DirectV4l2YuyvCapture capture(UsbOptions{});
                expect(mock.mappings.count(nullptr), "zero-address mapping was not exercised");
            }
            assert_released();
            expect(mock.stream_off_calls == 1, "cleanup skipped STREAMOFF");
        }
        std::puts("PASS: 360 syscall failures, 4 validation/allocation failures, "
                  "successful capture and zero-address/STREAMOFF cleanup");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
"""


def main():
    source = (COMPONENT / "src/las2_usb_stereo.cpp").read_text()
    begin = source.index("class DirectV4l2YuyvCapture {")
    end = source.index("\n};", begin) + len("\n};")
    capture_class = source[begin:end]
    with tempfile.TemporaryDirectory(prefix="las2-capture-cleanup-") as temporary:
        root = Path(temporary)
        fixture = root / "capture_cleanup.cpp"
        fixture.write_text(PREAMBLE + capture_class + TESTS)
        executable = root / "capture_cleanup"
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(
            [*compiler, "-std=c++17", "-O0", "-g", "-Wall", "-Wextra",
             "-Wpedantic", "-Werror", str(fixture), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
