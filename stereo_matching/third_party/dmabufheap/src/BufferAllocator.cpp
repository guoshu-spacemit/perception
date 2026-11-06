/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BufferAllocator.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <dirent.h>

using namespace std;

static constexpr char kDmaHeapRoot[] = "/dev/dma_heap/";

int BufferAllocator::OpenDmabufHeap(const std::string& heap_name) {
    std::shared_lock<std::shared_mutex> slock(dmabuf_heap_fd_mutex_);

    /* Check if heap has already been opened. */
    auto it = dmabuf_heap_fds_.find(heap_name);
    if (it != dmabuf_heap_fds_.end())
        return it->second;

    slock.unlock();

    /*
     * Heap device needs to be opened, use a unique_lock since dmabuf_heap_fd_
     * needs to be modified.
     */
    std::unique_lock<std::shared_mutex> ulock(dmabuf_heap_fd_mutex_);

    /*
     * Check if we already opened this heap again to prevent racing threads from
     * opening the heap device multiple times.
     */
    it = dmabuf_heap_fds_.find(heap_name);
    if (it != dmabuf_heap_fds_.end()) return it->second;

    std::string heap_path = kDmaHeapRoot + heap_name;
    int fd = TEMP_FAILURE_RETRY(open(heap_path.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd < 0) return -errno;

    //cout << "Using DMA-BUF heap named: " << heap_name << ", fd: " << fd << endl;

    auto ret = dmabuf_heap_fds_.insert({heap_name, fd});
    return fd;
}

void BufferAllocator::CloseDmabufHeap() {
    for (auto& heap_fd:dmabuf_heap_fds_) {
        if (heap_fd.second >= 0) {
            close(heap_fd.second);
        }
    }
}

BufferAllocator::BufferAllocator() {
}

BufferAllocator::~BufferAllocator() {
    CloseDmabufHeap();
}

int BufferAllocator::DmabufAlloc(const std::string& heap_name, size_t len) {
    int fd = OpenDmabufHeap(heap_name);
    if (fd < 0) return fd;

    struct dma_heap_allocation_data heap_data{
        .len = len,  // length of data to be allocated in bytes
        .fd_flags = O_RDWR | O_CLOEXEC,  // permissions for the memory to be allocated
    };

    struct timespec timeStart, timeEnd;
    long timeCostms = 0;
    clock_gettime(CLOCK_MONOTONIC, &timeStart);

    auto ret = TEMP_FAILURE_RETRY(ioctl(fd, DMA_HEAP_IOCTL_ALLOC, &heap_data));
    if (ret < 0) {
        cout << "Unable to allocate from DMA-BUF heap: " << heap_name << endl;
        return ret;
    }

    clock_gettime(CLOCK_MONOTONIC, &timeEnd);
    timeCostms = ((timeEnd.tv_sec * 1000 + timeEnd.tv_nsec / 1000000) - (timeStart.tv_sec * 1000 + timeStart.tv_nsec / 1000000));
    if (timeCostms > 3000)
        cout << "Alloc " << len << "B dmabuf from " << heap_name << " cost " << timeCostms << "ms" << endl;

    return heap_data.fd;
}

int BufferAllocator::Alloc(const std::string& heap_name, size_t len,
                           unsigned int heap_flags, size_t legacy_align) {
    int fd = DmabufAlloc(heap_name, len);

    if (fd < 0)
        cout << "Alloc dma buf fail. len is " << len << endl;

    return fd;
}

int BufferAllocator::AllocSystem(bool cpu_access_needed, size_t len, unsigned int heap_flags,
                                 size_t legacy_align) {
    if (!cpu_access_needed) {
        /*
         * CPU does not need to access allocated buffer so we try to allocate in
         * the 'system-uncached' heap after querying for its existence.
         */
        static bool uncached_dmabuf_system_heap_support = [this]() -> bool {
            auto dmabuf_heap_list = this->GetDmabufHeapList();
            return (dmabuf_heap_list.find(kDmabufSystemUncachedHeapName) != dmabuf_heap_list.end());
        }();

        if (uncached_dmabuf_system_heap_support)
            return DmabufAlloc(kDmabufSystemUncachedHeapName, len);

        cout << "AllocSystem. don't support system-uncached dma buf." << endl;
    }

    /*
     * Either 1) CPU needs to access allocated buffer OR 2) CPU does not need to
     * access allocated buffer but the "system-uncached" heap is unsupported.
     */
    return Alloc(kDmabufSystemHeapName, len, heap_flags, legacy_align);
}

int BufferAllocator::DoSync(unsigned int dmabuf_fd, bool start, SyncType sync_type) {
    struct dma_buf_sync sync = {
        .flags = (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) |
                static_cast<uint64_t>(sync_type),
    };
    return TEMP_FAILURE_RETRY(ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync));
}

int BufferAllocator::CpuSyncStart(unsigned int dmabuf_fd, SyncType sync_type) {
    int ret = DoSync(dmabuf_fd, true /* start */, sync_type);

    if (ret) cout << "CpuSyncStart() failure" << endl;
    return ret;
}

int BufferAllocator::CpuSyncEnd(unsigned int dmabuf_fd, SyncType sync_type) {
    int ret = DoSync(dmabuf_fd, false /* start */, sync_type);
    if (ret) cout << "CpuSyncEnd() failure" << endl;

    return ret;
}

std::unordered_set<std::string> BufferAllocator::GetDmabufHeapList() {
    std::unordered_set<std::string> heap_list;
    std::unique_ptr<DIR, int (*)(DIR*)> dir(opendir(kDmaHeapRoot), closedir);

    if (dir) {
        struct dirent* dent;
        while ((dent = readdir(dir.get()))) {
            if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) continue;
            heap_list.insert(dent->d_name);
        }
    }

    return heap_list;
}
