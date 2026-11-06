#!/usr/bin/env python3
"""Install real prebuilt ELF backends and execute a CMake package consumer."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


COMPONENT = Path(__file__).resolve().parents[1]


def run(*command, success=True, env=None):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if (result.returncode == 0) != success:
        command_text = " ".join(map(str, command))
        raise RuntimeError(f"{command_text}\n{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def make_backend(root, filename, soname=None):
    source = root / "backend.c"
    source.write_text("int fixture_backend(void) { return 42; }\n")
    output = root / filename
    arguments = ["cc", "-shared", "-fPIC", str(source), "-o", str(output)]
    if soname:
        arguments.append(f"-Wl,-soname,{soname}")
    run(*arguments)
    return output


def exercise_backend(root, project, name, soname=None, use_symlink=False):
    case = root / name
    case.mkdir()
    artifact = case / "artifact"
    artifact.mkdir()
    backend = make_backend(artifact, "backend.so.1.0.0", soname)
    if use_symlink:
        link = artifact / "backend.so"
        link.symlink_to(backend.name)
        backend = link
    prefix = case / "prefix"
    run("cmake", "-S", str(project), "-B", str(case / "build"),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DSTEREO_MATCHING_EXTERNAL_INPUT_LIBRARY={backend}")
    run("cmake", "--install", str(case / "build"))
    if soname:
        assert (prefix / "lib" / soname).is_file()
    assert (prefix / "lib/libstereo_matching_pipeline.so").is_file()
    relocated = case / "relocated"
    prefix.rename(relocated)
    shutil.rmtree(artifact)
    consumer = case / "consumer"
    consumer.mkdir()
    (consumer / "main.c").write_text(
        "extern int fixture_backend(void);\n"
        "int main(void) { return fixture_backend() == 42 ? 0 : 1; }\n")
    (consumer / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(prebuilt_consumer LANGUAGES C)
find_package(stereo_matching CONFIG REQUIRED COMPONENTS external_input)
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE stereo_matching::external_input)
''')
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    run("cmake", "-S", str(consumer), "-B", str(case / "consumer-build"),
        f"-DCMAKE_PREFIX_PATH={relocated}", env=environment)
    run("cmake", "--build", str(case / "consumer-build"), env=environment)
    executable = case / "consumer-build/consumer"
    dynamic = run("readelf", "-d", str(executable))
    if soname:
        assert f"[{soname}]" in dynamic
    run(str(executable), env=environment)
    print(f"PASS prebuilt {name}: installed consumer links and runs "
          "without source artifacts")


def main():
    with tempfile.TemporaryDirectory(prefix="las2-prebuilt-package-") as temporary:
        root = Path(temporary)
        # Run the actual package CMake rules with lightweight tensor/model files.
        project = root / "project"
        project.mkdir()
        shutil.copy2(COMPONENT / "CMakeLists.txt", project)
        shutil.copytree(COMPONENT / "cmake", project / "cmake")
        (project / "include").mkdir()
        (project / "lib").mkdir()
        (project / "model").mkdir()
        (project / "lib/libstereo_matching.so").write_bytes(b"unused tensor fixture")
        (project / "model/model.onnx").write_bytes(b"unused model fixture")
        (project / "NOTICE").write_text("Fixture notice\n")
        exercise_backend(root, project, "unversioned")
        exercise_backend(root, project, "soname", "libstereo_matching_pipeline.so.1")
        exercise_backend(root, project, "symlink",
                         "libstereo_matching_pipeline.so.1", True)
        invalid = root / "invalid"
        invalid.mkdir()
        wrong_soname = make_backend(invalid, "wrong.so", "libunrelated_backend.so.1")
        result = run("cmake", "-S", str(project), "-B", str(root / "invalid-build"),
                     f"-DSTEREO_MATCHING_EXTERNAL_INPUT_LIBRARY={wrong_soname}",
                     success=False)
        message = "SONAME must use the libstereo_matching_pipeline.so family"
        assert message in " ".join(result.split())
        not_elf = invalid / "not-elf.so"
        not_elf.write_text("not an ELF library\n")
        result = run("cmake", "-S", str(project), "-B", str(root / "not-elf-build"),
                     f"-DSTEREO_MATCHING_EXTERNAL_INPUT_LIBRARY={not_elf}",
                     success=False)
        assert "Cannot inspect prebuilt ExternalStereo ELF" in result
        print("PASS mismatched SONAME and non-ELF inputs rejected")


if __name__ == "__main__":
    main()
