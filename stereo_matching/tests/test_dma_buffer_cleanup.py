#!/usr/bin/env python3
"""Compile and fault-inject the production DMA owner without a DMA heap device."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile


COMPONENT = Path(__file__).resolve().parents[1]

PREAMBLE = r"""
#include "BufferAllocatorWrapper.h"
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class BufferAllocator {};

void expect(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Mapping
{
    std::size_t size;
    int fd;
};

struct Mock
{
    BufferAllocator* allocator = nullptr;
    bool fail_alloc = false;
    bool fail_map = false;
    bool zero_map = false;
    std::size_t allocated_size = 0;
    int alloc_calls = 0;
    int map_calls = 0;
    int unmap_calls = 0;
    int close_calls = 0;
    int invalid_unmaps = 0;
    int invalid_closes = 0;
    int close_before_unmap = 0;
    std::set<int> descriptors;
    std::map<void*, Mapping> mappings;
    std::vector<SyncType> sync_starts;
    std::vector<SyncType> sync_ends;
} mock;

extern "C" int DmabufHeapAllocSystem(BufferAllocator* allocator, bool cpu_access,
    std::size_t size, unsigned int flags, std::size_t alignment)
{
    expect(allocator == mock.allocator, "allocator identity changed");
    expect(cpu_access && flags == 0 && alignment == 0, "DMA allocation options changed");
    ++mock.alloc_calls;
    mock.allocated_size = size;
    if (mock.fail_alloc)
    {
        errno = ENOMEM;
        return -1;
    }
    int descriptor = 0;
    while (mock.descriptors.count(descriptor))
        ++descriptor;
    mock.descriptors.insert(descriptor);
    return descriptor;
}

extern "C" int DmabufHeapCpuSyncStart(
    BufferAllocator* allocator, unsigned int descriptor, SyncType type)
{
    expect(allocator == mock.allocator && mock.descriptors.count(descriptor),
        "sync start used the wrong allocator or a closed fd");
    mock.sync_starts.push_back(type);
    return 0;
}

extern "C" int DmabufHeapCpuSyncEnd(
    BufferAllocator* allocator, unsigned int descriptor, SyncType type)
{
    expect(allocator == mock.allocator && mock.descriptors.count(descriptor),
        "sync end used the wrong allocator or a closed fd");
    mock.sync_ends.push_back(type);
    return 0;
}

void* fake_mmap(void* address, std::size_t size, int protection, int flags, int fd, off_t offset)
{
    expect(address == nullptr && protection == (PROT_READ | PROT_WRITE) &&
               flags == MAP_SHARED && offset == 0,
        "DMA mapping options changed");
    expect(mock.descriptors.count(fd) && size == mock.allocated_size,
        "mapping used a closed fd or incorrect length");
    ++mock.map_calls;
    if (mock.fail_map)
    {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    void* mapping = mock.zero_map ? nullptr :
        reinterpret_cast<void*>(std::uintptr_t(fd + 1) * 4096);
    expect(mock.mappings.emplace(mapping, Mapping{size, fd}).second, "duplicate mapping");
    return mapping;
}

int fake_munmap(void* address, std::size_t size)
{
    ++mock.unmap_calls;
    auto found = mock.mappings.find(address);
    if (found == mock.mappings.end() || found->second.size != size)
        ++mock.invalid_unmaps;
    else
        mock.mappings.erase(found);
    return 0;
}

int fake_close(int fd)
{
    ++mock.close_calls;
    for (const auto& item : mock.mappings)
        if (item.second.fd == fd)
            ++mock.close_before_unmap;
    if (!mock.descriptors.erase(fd))
        ++mock.invalid_closes;
    return 0;
}

#define mmap fake_mmap
#define munmap fake_munmap
#define close fake_close
"""

TESTS = r"""
#undef mmap
#undef munmap
#undef close

static_assert(std::is_nothrow_destructible_v<DmaBuffer>);
static_assert(!std::is_copy_constructible_v<DmaBuffer>);
static_assert(!std::is_copy_assignable_v<DmaBuffer>);

void assert_released()
{
    expect(mock.descriptors.empty(), "DMA allocation leaked a descriptor");
    expect(mock.mappings.empty(), "DMA allocation leaked an mmap region");
    expect(mock.invalid_closes == 0 && mock.invalid_unmaps == 0, "resource released twice");
    expect(mock.close_before_unmap == 0, "DMA fd closed before unmapping");
}

void successful_allocation(BufferAllocator& allocator, std::size_t requested, bool zero_map)
{
    mock.zero_map = zero_map;
    const auto closes = mock.close_calls;
    const auto unmaps = mock.unmap_calls;
    const auto sync_starts = mock.sync_starts.size();
    const auto sync_ends = mock.sync_ends.size();
    {
        DmaBuffer buffer(&allocator, requested);
        expect(buffer.fd() == 0, "fd zero was not retained/reused correctly");
        expect(buffer.size() == requested, "requested size changed");
        const std::size_t aligned = ((requested + 4095) / 4096) * 4096;
        expect(mock.allocated_size == aligned, "page alignment changed");
        const auto found = mock.mappings.find(buffer.data());
        expect(found != mock.mappings.end() && found->second.size == aligned,
            "mapped data/size mismatch");
        expect(!zero_map || buffer.data() == nullptr, "zero-address mapping not exercised");
        for (const auto type : {kSyncRead, kSyncWrite, kSyncReadWrite})
        {
            buffer.sync_start(type);
            buffer.sync_end(type);
            expect(mock.sync_starts.back() == type && mock.sync_ends.back() == type,
                "DMA sync access flags changed");
        }
    }
    assert_released();
    expect(mock.close_calls == closes + 1 && mock.unmap_calls == unmaps + 1,
        "successful DMA owner did not release exactly once");
    expect(mock.sync_starts.size() == sync_starts + 3 && mock.sync_ends.size() == sync_ends + 3,
        "sync calls were not forwarded exactly once");
}

int main()
{
    try
    {
        BufferAllocator allocator;
        mock.allocator = &allocator;
        for (bool map_failure : {false, true})
        {
            for (int repeat = 0; repeat < 100; ++repeat)
            {
                mock.fail_alloc = !map_failure;
                mock.fail_map = map_failure;
                const auto closes = mock.close_calls;
                const auto unmaps = mock.unmap_calls;
                bool threw = false;
                try
                {
                    DmaBuffer buffer(&allocator, 4097);
                }
                catch (const std::runtime_error& error)
                {
                    threw = true;
                    expect(std::string(error.what()).find(
                        map_failure ? "mmap dmabuf" : "DmabufHeapAllocSystem failed") !=
                            std::string::npos,
                        "construction failure diagnostic changed");
                }
                expect(threw, "injected construction failure did not propagate");
                assert_released();
                expect(mock.close_calls == closes + (map_failure ? 1 : 0),
                    "partial constructor did not close the acquired fd exactly once");
                expect(mock.unmap_calls == unmaps, "failed mapping must not be unmapped");
                mock.fail_alloc = mock.fail_map = false;
                successful_allocation(allocator, 4097, false);
            }
        }
        for (std::size_t size : {1, 4096, 4097})
            for (bool zero_map : {false, true})
                successful_allocation(allocator, size, zero_map);
        std::puts("PASS production DmaBuffer: 200 alloc/mmap failures with successful retries; "
                  "fd zero, zero-address maps, page alignment and sync forwarding");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
"""


def main():
    source = (COMPONENT / "src/stereo_runtime.h").read_text()
    helpers_begin = source.index("std::string errno_msg(")
    class_begin = source.index("class DmaBuffer", helpers_begin)
    class_end = source.index("\n};", class_begin) + len("\n};")
    implementation = source[helpers_begin:class_end]
    with tempfile.TemporaryDirectory(prefix="las2-dma-cleanup-") as temporary:
        root = Path(temporary)
        fixture = root / "dma_cleanup.cpp"
        fixture.write_text(PREAMBLE + implementation + TESTS)
        executable = root / "dma_cleanup"
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(
            [*compiler, "-std=c++17", "-O0", "-g", "-Wall", "-Wextra",
             "-Wpedantic", "-Werror", "-I", str(COMPONENT / "thirdparty/dmabufheap/include"),
             str(fixture), "-o", str(executable)], check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
