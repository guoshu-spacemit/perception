#!/usr/bin/env python3
"""Exercise runtime installation and a relocated consumer with host ELF stubs."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


COMPONENT = Path(__file__).resolve().parents[1]


def run(*command, cwd=None, success=True, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if (result.returncode == 0) != success:
        raise RuntimeError(f"{' '.join(map(str, command))}\n{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def shared_library(directory, name, version, soname, source, *arguments):
    path = directory / name
    path.write_text(source)
    output = directory / version
    run("cc", "-shared", "-fPIC", str(path), "-o", str(output),
        f"-Wl,-soname,{soname}", *arguments)
    if soname != version:
        (directory / soname).symlink_to(version)
    unversioned = version.split(".so")[0] + ".so"
    if unversioned != version:
        (directory / unversioned).symlink_to(soname)


def relocated_consumer(root, prefix, artifact):
    """Execute against the relocated install, including provider dlopen calls."""
    consumer = root / "consumer"
    consumer.mkdir()
    (consumer / "main.c").write_text(
        "extern int fixture_backend(void);\n"
        "int main(void) { return fixture_backend() == 12 ? 0 : 1; }\n")
    (consumer / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES C)
find_package(stereo_matching CONFIG REQUIRED COMPONENTS external_input)
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE stereo_matching::external_input)
''')
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    run("cmake", "-S", str(consumer), "-B", str(root / "consumer-build"),
        f"-DCMAKE_PREFIX_PATH={prefix}", env=environment)
    run("cmake", "--build", str(root / "consumer-build"), env=environment)
    run(str(root / "consumer-build/consumer"), env=environment)
    dynamic = run("readelf", "-d", str(prefix / "lib/libstereo_matching_pipeline.so"))
    assert "$ORIGIN" in dynamic and str(artifact) not in dynamic
    configuration = (prefix / "lib/cmake/stereo_matching/stereo_matchingConfig.cmake").read_text()
    assert str(artifact) not in configuration


def main():
    with tempfile.TemporaryDirectory(prefix="las2-runtime-package-") as temporary:
        root = Path(temporary)
        artifact = root / "selected-runtime"
        libraries = artifact / "lib"
        libraries.mkdir(parents=True)
        shared_library(libraries, "providers.c", "libonnxruntime_providers_shared.so",
                       "libonnxruntime_providers_shared.so", "int providers(void) { return 5; }\n")
        shared_library(libraries, "ort.c", "libonnxruntime.so.1.2.3", "libonnxruntime.so.1",
                       "extern int providers(void); int ort(void) { return providers(); }\n",
                       f"-L{libraries}", "-lonnxruntime_providers_shared", "-Wl,-rpath,$ORIGIN",
                       "-Wl,--disable-new-dtags")
        shared_library(libraries, "tcm.c", "libspine_tcm.so.0.2.0", "libspine_tcm.so.0",
                       "int tcm(void) { return 7; }\n")
        shared_library(libraries, "ep.c", "libspacemit_ep.so.2.0.4", "libspacemit_ep.so.2",
                       '#include <dlfcn.h>\nint ep(void) {\n'
                       'void *h = dlopen("libspine_tcm.so", RTLD_NOW);\n'
                       'if (!h) return -100;\nint (*f)(void) = dlsym(h, "tcm");\n'
                       'int value = f ? f() : -200; dlclose(h); return value; }\n',
                       "-ldl", "-Wl,-rpath,$ORIGIN", "-Wl,--disable-new-dtags")
        (artifact / "LICENSE").write_text("Fixture runtime license\n")
        (artifact / "SHA256SUMS").write_text("Fixture source manifest\n")
        (artifact / "README.vendor.md").write_text("Fixture selected runtime\n")
        model = root / "LAS2_M_288x384.fp16.onnx"
        model.write_text("external model\n")
        project = root / "project"
        project.mkdir()
        (project / "backend.c").write_text(
            "extern int ort(void); extern int ep(void);\n"
            "int fixture_backend(void) { return ort() + ep(); }\n")
        (project / "tensor-model.onnx").write_text("original tensor model\n")
        (project / "include").mkdir()
        (project / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.16)
project(runtime_fixture LANGUAGES C)
# Exercise target-architecture decisions with host ELF fixtures. This is not a
# cross-compiler or RISC-V ABI test.
if(DEFINED FIXTURE_TARGET_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR "${{FIXTURE_TARGET_PROCESSOR}}")
endif()
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
add_library(backend SHARED backend.c)
set_target_properties(backend PROPERTIES OUTPUT_NAME stereo_matching_pipeline INSTALL_RPATH "$ORIGIN")
target_link_libraries(backend PRIVATE "${{ONNXRUNTIME_ROOT}}/lib/libonnxruntime.so"
  "${{ONNXRUNTIME_ROOT}}/lib/libspacemit_ep.so")
target_link_options(backend PRIVATE -Wl,--no-undefined)
install(TARGETS backend LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
install(DIRECTORY include/ DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}})
install(FILES tensor-model.onnx DESTINATION share/stereo_matching/model
  RENAME LAS2_M_288x384.fp16.onnx)
set(STEREO_MATCHING_HAS_EXTERNAL_INPUT TRUE)
configure_package_config_file("{COMPONENT}/cmake/stereo_matchingConfig.cmake.in"
  "${{CMAKE_CURRENT_BINARY_DIR}}/stereo_matchingConfig.cmake"
  INSTALL_DESTINATION ${{CMAKE_INSTALL_LIBDIR}}/cmake/stereo_matching)
install(FILES "${{CMAKE_CURRENT_BINARY_DIR}}/stereo_matchingConfig.cmake"
  DESTINATION ${{CMAKE_INSTALL_LIBDIR}}/cmake/stereo_matching)
include("{COMPONENT}/cmake/ExternalRuntime.cmake")
''')
        prefix = root / "prefix"
        build = root / "build"
        run("cmake", "-S", str(project), "-B", str(build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}", f"-DONNXRUNTIME_ROOT={artifact}",
            f"-DSTEREO_MATCHING_EXTERNAL_INPUT_MODEL={model}",
            "-DFIXTURE_TARGET_PROCESSOR=x86_64")
        assert "STEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM:BOOL=OFF" in (
            build / "CMakeCache.txt").read_text()
        run("cmake", "--build", str(build))
        run("cmake", "--install", str(build))
        runtime_files = sorted(libraries.glob("*.so*"))
        assert len(runtime_files) == 10
        for path in runtime_files:
            installed = prefix / "lib" / path.name
            assert installed.read_bytes() == path.read_bytes(), path.name
            assert installed.is_symlink() == path.is_symlink(), path.name
            if path.is_symlink():
                assert installed.readlink() == path.readlink(), path.name
        metadata = prefix / "share/stereo_matching"
        assert (metadata / "licenses/external-ort-LICENSE").is_file()
        checksums = metadata / "external_input/runtime/DEPLOYED_RUNTIME_SHA256SUMS"
        for line in checksums.read_text().splitlines():
            checksum, relative = line.split("  ", 1)
            assert hashlib.sha256((prefix / relative).read_bytes()).hexdigest() == checksum
        assert (metadata / "model/LAS2_M_288x384.fp16.onnx").read_text() == "original tensor model\n"
        assert (metadata / "external_input/model" / model.name).read_text() == "external model\n"
        assert not (prefix / "include/onnxruntime_cxx_api.h").exists()
        # Supplied optional TCM was installed above; the same complete artifact
        # must also satisfy an explicit declaration of its provider's dlopen use.
        run("cmake", "-S", str(project), "-B", str(build),
            "-DSTEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM=ON")
        run("cmake", "--install", str(build))

        # A generic provider without TCM must configure, install and run. The
        # remaining runtime contracts (license, links and checksums) still apply.
        generic_artifact = root / "generic-runtime"
        shutil.copytree(artifact, generic_artifact, symlinks=True)
        generic_libraries = generic_artifact / "lib"
        for pattern in ("libspine_tcm.so*", "libspacemit_ep.so*"):
            for path in generic_libraries.glob(pattern):
                path.unlink()
        shared_library(generic_libraries, "ep.c", "libspacemit_ep.so.2.0.4",
                       "libspacemit_ep.so.2", "int ep(void) { return 7; }\n")
        generic_prefix = root / "generic-prefix"
        generic_build = root / "generic-build"
        run("cmake", "-S", str(project), "-B", str(generic_build),
            f"-DCMAKE_INSTALL_PREFIX={generic_prefix}",
            f"-DONNXRUNTIME_ROOT={generic_artifact}",
            "-DFIXTURE_TARGET_PROCESSOR=x86_64")
        cache = (generic_build / "CMakeCache.txt").read_text()
        assert "STEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM:BOOL=OFF" in cache
        run("cmake", "--build", str(generic_build))
        run("cmake", "--install", str(generic_build))
        assert not list((generic_prefix / "lib").glob("libspine_tcm.so*"))
        generic_checksums = (generic_prefix / "share/stereo_matching/external_input/runtime/"
                            "DEPLOYED_RUNTIME_SHA256SUMS").read_text()
        assert "spine_tcm" not in generic_checksums
        for line in generic_checksums.splitlines():
            checksum, relative = line.split("  ", 1)
            assert hashlib.sha256((generic_prefix / relative).read_bytes()).hexdigest() == checksum
        assert (generic_prefix / "share/stereo_matching/licenses/external-ort-LICENSE").is_file()
        for path in generic_libraries.glob("*.so*"):
            installed = generic_prefix / "lib" / path.name
            assert installed.read_bytes() == path.read_bytes()
            assert installed.is_symlink() == path.is_symlink()
            if path.is_symlink():
                assert installed.readlink() == path.readlink()

        # RISC-V requires TCM even if a caller tries to disable the declaration.
        # A non-RISC-V provider can explicitly declare an otherwise invisible
        # dlopen dependency; no DT_NEEDED heuristic may silently drop it.
        for processor, arguments in (
                ("riscv64", []),
                ("riscv64", ["-DSTEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM=OFF"]),
                ("x86_64", ["-DSTEREO_MATCHING_EXTERNAL_RUNTIME_REQUIRES_TCM=ON"])):
            missing_build = root / f"missing-tcm-{processor}-{len(arguments)}"
            failure = run("cmake", "-S", str(project), "-B", str(missing_build),
                          f"-DONNXRUNTIME_ROOT={generic_artifact}",
                          f"-DFIXTURE_TARGET_PROCESSOR={processor}", *arguments, success=False)
            assert "requires libspine_tcm.so" in failure
        print("PASS generic non-TCM runtime; RISC-V and declared dynamic TCM requirements enforced")

        generic_artifact.rename(root / "hidden-generic-runtime")
        generic_relocated = root / "generic-relocated"
        generic_prefix.rename(generic_relocated)
        generic_consumer = root / "generic-consumer-test"
        generic_consumer.mkdir()
        relocated_consumer(generic_consumer, generic_relocated, generic_artifact)
        print("PASS relocated generic consumer without TCM or LD_LIBRARY_PATH")

        # Remove access to the selected source artifact, move the deployment,
        # then link and run through the real installed package configuration.
        artifact.rename(root / "hidden-runtime")
        relocated = root / "relocated"
        prefix.rename(relocated)
        relocated_consumer(root, relocated, artifact)
        configuration = (relocated / "lib/cmake/stereo_matching/stereo_matchingConfig.cmake").read_text()
        assert str(artifact) not in configuration and str(prefix) not in configuration
        print("PASS four runtime libraries, six relative symlinks, license and deployment SHA manifest")
        print("PASS explicit external model preserved alongside unchanged tensor model")
        print("PASS relocated installed consumer links and runs, including TCM dlopen, without LD_LIBRARY_PATH")

        # Incomplete artifacts fail clearly; opting out retains explicit
        # deployment ownership without installing a partial runtime silently.
        incomplete = root / "incomplete"
        incomplete.mkdir()
        failure = run("cmake", "-S", str(project), "-B", str(root / "missing-build"),
                      f"-DONNXRUNTIME_ROOT={incomplete}", success=False)
        assert "requires libonnxruntime.so" in failure
        run("cmake", "-S", str(project), "-B", str(root / "runtime-off-build"),
            f"-DONNXRUNTIME_ROOT={root / 'hidden-runtime'}",
            "-DSTEREO_MATCHING_INSTALL_EXTERNAL_RUNTIME=OFF")
        print("PASS incomplete selected runtime rejected; explicit runtime deployment opt-out configures")


if __name__ == "__main__":
    main()
