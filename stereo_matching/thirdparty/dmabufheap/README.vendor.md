# dmabufheap vendor snapshot

This directory contains the minimal Android-compatible `libdmabufheap`
allocator sources required by the LAS2 DMA-BUF path.

The source files identify the Android Open Source Project as their copyright
holder and are distributed under Apache License 2.0. The license text is kept
in `LICENSE`. The upstream project is:

<https://android.googlesource.com/platform/system/memory/libdmabufheap/>

Only the following build inputs are vendored:

- `include/BufferAllocator.h`
- `include/BufferAllocatorWrapper.h`
- `include/dmabufheap-defs.h`
- `src/BufferAllocator.cpp`
- `src/BufferAllocatorWrapper.cpp`

Models, binaries, generated files, and unrelated Android build metadata are
not included.
