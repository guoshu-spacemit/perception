#!/usr/bin/env python3
"""Exercise default runtime selection with real package rules and host ELF stubs."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from test_runtime_packaging import run, shared_library


COMPONENT = Path(__file__).resolve().parents[1]


def make_project(root):
    """Copy build rules and replace inference/hardware code with a link fixture."""
    project = root / "project"
    project.mkdir()
    shutil.copy2(COMPONENT / "CMakeLists.txt", project)
    shutil.copytree(COMPONENT / "cmake", project / "cmake")
    for path in ("lib", "include", "src", "licenses", "thirdparty/onnxruntime/include",
                 "thirdparty/dmabufheap/src", "thirdparty/dmabufheap/include",
                 "mpp/include/sys", "cmake-fixtures"):
        (project / path).mkdir(parents=True, exist_ok=True)
    (project / "lib/libstereo_matching.so").write_bytes(b"unused tensor fixture")
    for name in ("NOTICE", "UPSTREAM.json", "licenses/LiteAnyStereo-MIT.txt",
                 "thirdparty/dmabufheap/LICENSE", "thirdparty/onnxruntime/LICENSE",
                 "licenses/spacemit-runtime-MIT.txt"):
        (project / name).write_text("fixture metadata/license\n")
    for name in ("onnxruntime_c_api.h", "onnxruntime_cxx_api.h", "spacemit_ort_env.h"):
        (project / "thirdparty/onnxruntime/include" / name).write_text("// fixture\n")
    (project / "mpp/include/sys/vb_type.h").write_text("// fixture\n")
    for name in ("BufferAllocator.cpp", "BufferAllocatorWrapper.cpp"):
        (project / "thirdparty/dmabufheap/src" / name).write_text("// fixture\n")
    (project / "src/stereo_pipeline.cpp").write_text(
        'extern "C" int fixture_runtime(void);\n'
        'extern "C" int fixture_backend(void) { return fixture_runtime() + 1; }\n')
    shared_library(project / "lib", "runtime.c", "libstereo_matching_runtime.so",
                   "libstereo_matching_runtime.so", "int fixture_runtime(void) { return 41; }\n")
    digest = hashlib.sha256((project / "lib/libstereo_matching_runtime.so").read_bytes()).hexdigest()
    (project / "RUNTIME_CONTRACT.json").write_text(json.dumps({
        "runtime": {"embedded_library_sha256": digest},
    }))
    # Keep this a packaging test: no OpenCV/OpenCL or capture code is executed.
    fixtures = project / "cmake-fixtures"
    (fixtures / "FindOpenCV.cmake").write_text("set(OpenCV_FOUND TRUE)\n")
    (fixtures / "FindOpenCL.cmake").write_text(
        "set(OpenCL_FOUND TRUE)\nadd_library(OpenCL::OpenCL INTERFACE IMPORTED)\n")
    (fixtures / "yaml-cpp-config.cmake").write_text(
        "add_library(yaml-cpp INTERFACE IMPORTED)\n")
    return project


def main():
    environment = os.environ.copy()
    for name in ("ONNXRUNTIME_ROOT", "LD_LIBRARY_PATH", "SROBOTIS_ROOT"):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="las2-embedded-package-") as temporary:
        root = Path(temporary)
        project = make_project(root)
        prefix = root / "prefix"
        model = root / "model.onnx"
        model.write_bytes(b"optional model fixture")
        common = [
            "-DSTEREO_MATCHING_BUILD_EXTERNAL_INPUT=ON",
            f"-DCMAKE_MODULE_PATH={project / 'cmake-fixtures'}",
            f"-Dyaml-cpp_DIR={project / 'cmake-fixtures'}",
            f"-DMPP_ROOT={project / 'mpp'}",
        ]
        build = root / "build"
        run("cmake", "-S", str(project), "-B", str(build), *common,
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DSTEREO_MATCHING_EXTERNAL_INPUT_MODEL={model}", env=environment)
        run("cmake", "--build", str(build), env=environment)
        run("cmake", "--install", str(build), env=environment)
        assert (prefix / "lib/libstereo_matching_runtime.so").is_file()
        assert not list((prefix / "lib").glob("libonnxruntime*.so*"))
        metadata = prefix / "share/stereo_matching"
        assert (metadata / "licenses/external-ort-LICENSE").is_file()
        assert (metadata / "licenses/onnxruntime-MIT.txt").is_file()
        assert (metadata / "external_input/model/model.onnx").read_bytes() == model.read_bytes()
        checksums = metadata / "external_input/runtime/DEPLOYED_RUNTIME_SHA256SUMS"
        lines = checksums.read_text().splitlines()
        assert len(lines) == 1
        checksum, name = lines[0].split("  ", 1)
        assert hashlib.sha256((prefix / name).read_bytes()).hexdigest() == checksum

        no_model_prefix = root / "no-model-prefix"
        no_model_build = root / "no-model-build"
        run("cmake", "-S", str(project), "-B", str(no_model_build), *common,
            f"-DCMAKE_INSTALL_PREFIX={no_model_prefix}", env=environment)
        run("cmake", "--build", str(no_model_build), env=environment)
        run("cmake", "--install", str(no_model_build), env=environment)
        assert not (no_model_prefix / "share/stereo_matching/external_input/model").exists()

        # Remove the source binary, then consume only the relocated installation.
        runtime = project / "lib/libstereo_matching_runtime.so"
        saved_runtime = runtime.read_bytes()
        runtime.unlink()
        relocated = root / "relocated"
        prefix.rename(relocated)
        consumer = root / "consumer"
        consumer.mkdir()
        (consumer / "main.c").write_text(
            "extern int fixture_backend(void);\n"
            "int main(void) { return fixture_backend() == 42 ? 0 : 1; }\n")
        (consumer / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES C)
find_package(stereo_matching CONFIG REQUIRED COMPONENTS external_input)
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE stereo_matching::external_input)
''')
        run("cmake", "-S", str(consumer), "-B", str(root / "consumer-build"),
            f"-DCMAKE_PREFIX_PATH={relocated}", env=environment)
        run("cmake", "--build", str(root / "consumer-build"), env=environment)
        run(str(root / "consumer-build/consumer"), env=environment)
        dynamic = run("readelf", "-d", str(relocated / "lib/libstereo_matching_pipeline.so"))
        assert "$ORIGIN" in dynamic and str(project) not in dynamic
        print("PASS default runtime/model install and relocated consumer without external runtime or LD_LIBRARY_PATH")

        failure = run("cmake", "-S", str(project), "-B", str(root / "missing-build"),
                      *common, env=environment, success=False)
        assert "Bundled LAS2 runtime is missing" in failure
        runtime.write_bytes(saved_runtime + b"tamper")
        failure = run("cmake", "-S", str(project), "-B", str(root / "tampered-build"),
                      *common, env=environment, success=False)
        assert "runtime SHA256 does not match" in failure
        runtime.write_bytes(saved_runtime)
        toolchain = root / "different-target.cmake"
        wrong_target = "x86_64" if platform.machine().startswith("riscv") else "riscv64"
        toolchain.write_text(
            "set(CMAKE_SYSTEM_NAME Linux)\n"
            f"set(CMAKE_SYSTEM_PROCESSOR {wrong_target})\n")
        failure = run("cmake", "-S", str(project), "-B", str(root / "wrong-target-build"),
                      *common, f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
                      env=environment, success=False)
        assert f"ELF machine does not match target {wrong_target}" in failure
        print("PASS missing/tampered runtime and mismatched target rejected before link")


if __name__ == "__main__":
    main()
