// Exercise the exact production calibration loader without creating an ORT
// session, opening a camera, or allocating DMA. Unused demo code is link-pruned.
#define LAS2_USB_STEREO_NO_MAIN
#include "../src/usb_mpp_las2_demo.cpp"

namespace {
void require_test(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct TemporaryCalibration {
    std::string directory;
    std::string path;
    TemporaryCalibration() {
        char pattern[] = "/tmp/las2-map-test-XXXXXX";
        const char *result = mkdtemp(pattern);
        require_test(result != nullptr, "mkdtemp failed");
        directory = result;
        path = directory + "/calibration.yaml";
    }
    ~TemporaryCalibration() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(directory, error);
    }
    void write(const std::string &contents) {
        std::ofstream file(path, std::ios::binary);
        file << contents;
        require_test(file.good(), "write test calibration");
    }
};

std::string fixture(const char *distortion, int width, int height, double fx) {
    std::ostringstream yaml;
    for (int camera = 0; camera < 2; ++camera) {
        yaml << "cam" << camera << ":\n"
             << "  camera_model: pinhole\n  distortion_model: " << distortion << '\n'
             << "  intrinsics: [" << fx << ", " << fx << ", " << width * .49 << ", " << height * .51
             << "]\n"
             << "  distortion_coeffs: [-0.04, 0.003, 0, 0]\n"
             << "  resolution: [" << width << ", " << height << "]\n";
        if (camera == 1)
            yaml << "  T_cn_cnm1: [[1,0,0,-0.065],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n";
    }
    return yaml.str();
}

void cache_tests() {
    TemporaryCalibration file;
    for (const char *model : {"radtan", "equidistant"}) {
        file.write(fixture(model, 64, 48, 45));
        auto first = load_shared_rectify_maps(file.path, 64, 48, 0);
        auto second = load_shared_rectify_maps(file.path, 64, 48, 0);
        require_test(first.get() == second.get(), "map cache did not share storage");
        auto other_alpha = load_shared_rectify_maps(file.path, 64, 48, .5);
        require_test(other_alpha.get() != first.get(), "alpha missing from cache key");
        file.write(fixture(model, 64, 48, 47));
        auto changed = load_shared_rectify_maps(file.path, 64, 48, 0);
        require_test(changed.get() != first.get() && changed->fx != first->fx,
                     "in-place calibration edit reused stale maps");
        bool rejected = false;
        try {
            load_shared_rectify_maps(file.path, 65, 48, 0);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require_test(rejected, "wrong dimensions reused cached geometry");
        TemporaryCalibration same_contents;
        same_contents.write(fixture(model, 64, 48, 47));
        auto shared_path = load_shared_rectify_maps(same_contents.path, 64, 48, 0);
        require_test(shared_path.get() == changed.get(), "identical contents not shared");
        std::weak_ptr<const RectifyMaps> lifetime = first;
        first.reset();
        second.reset();
        require_test(lifetime.expired(), "cache retained maps without consumers");
    }
    file.write(fixture("equidistant", 64, 48, 49));
    std::promise<void> begin;
    const auto gate = begin.get_future().share();
    std::vector<std::future<std::shared_ptr<const RectifyMaps>>> futures;
    for (int index = 0; index < 8; ++index)
        futures.push_back(std::async(std::launch::async, [&] {
            gate.wait();
            return load_shared_rectify_maps(file.path, 64, 48, 0);
        }));
    begin.set_value();
    auto first = futures.front().get();
    for (size_t index = 1; index < futures.size(); ++index)
        require_test(futures[index].get().get() == first.get(), "concurrent maps duplicated");
    std::cout << "PASS calibration content/alpha/dimensions/lifetime/concurrent cache\n";
}

void reference(const RectifyMaps &maps, const std::string &mode, const char *path) {
    const bool write = mode == "--write-reference";
    require_test(write || mode == "--verify-reference", "invalid reference mode");
    std::fstream file(path,
                      std::ios::binary | (write ? std::ios::out | std::ios::trunc : std::ios::in));
    require_test(file.good(), "reference file open failed");
    std::vector<double> geometry{maps.fx, maps.fy, maps.cx, maps.cy, maps.baseline_m};
    geometry.insert(geometry.end(), maps.rectified_from_left.begin(),
                    maps.rectified_from_left.end());
    double maximum = 0;
    auto original_geometry = geometry;
    if (write)
        file.write(reinterpret_cast<const char *>(geometry.data()), geometry.size() * 8);
    else
        file.read(reinterpret_cast<char *>(geometry.data()), geometry.size() * 8);
    require_test(file.good(), "reference geometry I/O failed");
    for (size_t n = 0; n < geometry.size(); ++n)
        require_test(std::abs(geometry[n] - original_geometry[n]) <= 1e-9,
                     "FOV/intrinsics/extrinsics changed");
    for (const auto *values : {&maps.left_x, &maps.left_y, &maps.right_x, &maps.right_y}) {
        if (write)
            file.write(reinterpret_cast<const char *>(values->data()), values->size() * 4);
        else {
            std::vector<float> expected(values->size());
            file.read(reinterpret_cast<char *>(expected.data()), expected.size() * 4);
            for (size_t n = 0; n < expected.size(); ++n) {
                require_test(std::isfinite(expected[n]) && std::isfinite((*values)[n]),
                             "nonfinite production map");
                maximum = std::max(maximum, std::abs(double(expected[n]) - (*values)[n]));
            }
        }
        require_test(file.good(), "reference maps I/O failed");
    }
    require_test(maximum <= .00025, "production map coordinate error > 0.00025px");
    std::cout << "PASS production reference geometry/max_error_px=" << maximum << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::cout << std::unitbuf;
        cache_tests();
        require_test(argc == 1 || argc == 2 || argc == 4,
                     "usage: [KALIBR [--write-reference|--verify-reference FILE]]");
        if (argc > 1) {
            const auto root = YAML::LoadFile(argv[1]);
            const auto dimensions = yaml_vector(root["cam0"], "resolution", 2);
            const int width = std::lround(dimensions[0]), height = std::lround(dimensions[1]);
            auto start = std::chrono::steady_clock::now();
            auto first = load_shared_rectify_maps(argv[1], width, height, 0);
            double cold_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            start = std::chrono::steady_clock::now();
            auto second = load_shared_rectify_maps(argv[1], width, height, 0);
            double hit_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            require_test(first.get() == second.get(), "production maps not shared");
            std::cout << "MAP_INIT cold_ms=" << cold_ms << " hit_ms=" << hit_ms
                      << " shared_bytes=" << first->left_x.size() * sizeof(float) * 4 << '\n';
            if (argc == 4)
                reference(*first, argv[2], argv[3]);
        }
        std::cout << "PASS backend=" << (LAS2_HAS_RVV_RECTIFICATION ? "rvv-fp64" : "opencv")
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
