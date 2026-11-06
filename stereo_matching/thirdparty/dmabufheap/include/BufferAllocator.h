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

#pragma once

#include "dmabufheap-defs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class BufferAllocator {
  public:
    BufferAllocator();
    ~BufferAllocator();

    /* Not copyable or movable */
    BufferAllocator(const BufferAllocator&) = delete;
    BufferAllocator& operator=(const BufferAllocator&) = delete;

    /* *
     * Returns a dmabuf fd if the allocation in one of the specified heaps is successful and
     * an error code otherwise. If dmabuf heaps are supported, tries to allocate in the
     * specified dmabuf heap. If allocation fails in the specified dmabuf heap and ion_fd is a
     * valid fd, goes through saved heap data to find a heap ID/mask to match the specified heap
     * names and allocates memory as per the specified parameters. For vendor defined heaps with a
     * legacy ION interface(no heap query support), MapNameToIonMask() must be called prior to
     * invocation of Alloc() to map a heap name to an equivalent heap mask and heap flag
     * configuration.
     * @heap_name: name of the heap to allocate in.
     * @len: size of the allocation.
     * @heap_flags: flags passed to heap.
     * @legacy_align: alignment value used only by legacy ION
     */
    int Alloc(const std::string& heap_name, size_t len, unsigned int heap_flags = 0, size_t legacy_align = 0);

    /* *
     * Returns a dmabuf fd if the allocation in system heap(cached/uncached) is successful and
     * an error code otherwise. Allocates in the 'system' heap if CPU access of
     * the buffer is expected and 'system-uncached' otherwise. If the 'system-uncached'
     * heap is not supported, falls back to the 'system' heap.
     * For vendor defined heaps with a legacy ION interface(no heap query support),
     * MapNameToIonMask() must be called prior to invocation of AllocSystem() to
     * map names 'system'(and optionally 'system-uncached' if applicable) to an
     * equivalent heap mask and heap flag configuration;
     * configuration.
     * @cpu_access: indicates if CPU access of the buffer is expected.
     * @len: size of the allocation.
     * @heap_flags: flags passed to heap.
     * @legacy_align: alignment value used only by legacy ION
     */
    int AllocSystem(bool cpu_access, size_t len, unsigned int heap_flags = 0,
                    size_t legacy_align = 0);

    /**
     * Must be invoked before CPU access of the allocated memory.
     * For a legacy ion interface, syncs a shared dmabuf fd with memory either using
     * ION_IOC_SYNC ioctl or using callback @legacy_ion_cpu_sync if specified. For
     * non-legacy ION and dmabuf heap interfaces, DMA_BUF_IOCTL_SYNC is used.
     * @fd: dmabuf fd. When the legacy version of ion is in use and a callback
     * function is supplied, this is passed as the second argument to legacy_ion_cpu_sync.
     * @sync_type: specifies if the sync is for read, write or read/write.
     * @legacy_ion_cpu_sync: optional callback for legacy ion interfaces. If
     * specified, will be invoked instead of ion_sync_fd()
     * to sync dmabuf_fd with memory. The paremeter will be ignored if the interface being
     * used is not legacy ion.
     * @legacy_ion_custom_data: When the legacy version of ion is in use and a callback
     * function is supplied, this pointer is passed as the third argument to
     * legacy_ion_cpu_sync. It is intended to point to data for performing the callback.
     *
     * Returns 0  on success and an error code otherwise.
     */
    int CpuSyncStart(unsigned int dmabuf_fd, SyncType sync_type = kSyncRead);

    /**
     * Must be invoked once CPU is done accessing the allocated memory.
     * For a legacy ion interface, syncs a shared dmabuf fd with memory using
     * either ION_IOC_SYNC ioctl or using callback @legacy_ion_cpu_sync if
     * specified. For non-legacy ION and dmabuf heap interfaces,
     * DMA_BUF_IOCTL_SYNC is used.
     * @dmabuf_fd: dmabuf fd. When the legacy version of ion is in use and a callback
     * function is supplied, this is passed as the second argument to legacy_ion_cpu_sync.
     * @sync_type: specifies if sync_type is for read, write or read/write.
     * @legacy_ion_cpu_sync: optional callback for legacy ion interfaces. If
     * specified, will be invoked instead of ion_sync_fd with a dup of ion_fd_ as its
     * argument. The parameter will be ignored if the interface being used is
     * not legacy ion.
     * @legacy_ion_custom_data: When the legacy version of ion is in use and a callback
     * function is supplied, this pointer is passed as the third argument to
     * legacy_ion_cpu_sync. It is intended to point to data for performing the callback.
     *
     * Returns 0 on success and an error code otherwise.
     */
    int CpuSyncEnd(unsigned int dmabuf_fd, SyncType sync_type = kSyncRead);

    /**
     * Query supported DMA-BUF heaps.
     *
     * @return the list of supported DMA-BUF heap names.
     */
    static std::unordered_set<std::string> GetDmabufHeapList();

  private:
    int OpenDmabufHeap(const std::string& name);
    int GetDmabufHeapFd(const std::string& name);
    bool DmabufHeapsSupported() { return !dmabuf_heap_fds_.empty(); }
    void LogInterface(const std::string& interface);
    int DmabufAlloc(const std::string& heap_name, size_t len);
    int DoSync(unsigned int dmabuf_fd, bool start, SyncType sync_type);
    void CloseDmabufHeap();

    /* Stores all open dmabuf_heap handles. */
    std::unordered_map<std::string, int> dmabuf_heap_fds_;
    /* Protects dma_buf_heap_fd_ from concurrent access */
    std::shared_mutex dmabuf_heap_fd_mutex_;
};
