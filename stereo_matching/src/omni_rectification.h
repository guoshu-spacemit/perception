#ifndef OMNI_RECTIFICATION_H
#define OMNI_RECTIFICATION_H

// Included after RectifyMaps and Kalibr YAML helpers in the implementation.
RectifyMaps load_omni_rectify_maps(
    const std::string& path, const las2::ExternalStereoInputConfig& config)
{
    const auto root = YAML::LoadFile(path);
    auto camera = [&](int index)
    {
        const auto node = root["cam" + std::to_string(index)];
        check(!node || node["camera_model"].as<std::string>() != "omni" ||
                node["distortion_model"].as<std::string>() != "radtan",
            "stacked input requires Kalibr omni+radtan cameras");
        const auto resolution = yaml_vector(node, "resolution", 2);
        check(resolution[0] != config.eye_width || resolution[1] != config.eye_height,
            "omni source dimensions differ from calibration");
        return node;
    };
    const auto left = camera(config.left_camera), right = camera(config.right_camera);
    const auto t0 = kalibr_transform(camera(0), "T_cam_imu");
    const auto ta = kalibr_transform(left, "T_cam_imu");
    const auto tb = kalibr_transform(right, "T_cam_imu");
    const cv::Mat ra = ta(cv::Rect(0, 0, 3, 3)), rb = tb(cv::Rect(0, 0, 3, 3));
    const cv::Mat ca = -ra.t() * ta(cv::Rect(3, 0, 1, 3));
    const cv::Mat cb = -rb.t() * tb(cv::Rect(3, 0, 1, 3));
    auto vector = [](const cv::Mat& value)
    { return cv::Vec3d(value.at<double>(0), value.at<double>(1), value.at<double>(2)); };
    const cv::Vec3d baseline = vector(cb - ca);
    const double length = cv::norm(baseline);
    check(!std::isfinite(length) || length < 1e-6, "invalid omni stereo baseline");
    const cv::Vec3d ex = baseline / length;
    const cv::Vec3d forward(ra.at<double>(2, 0) + rb.at<double>(2, 0),
        ra.at<double>(2, 1) + rb.at<double>(2, 1), ra.at<double>(2, 2) + rb.at<double>(2, 2));
    cv::Vec3d ez = forward - ex * ex.dot(forward);
    check(cv::norm(ez) < 1e-6, "omni pair has no common forward direction");
    ez /= cv::norm(ez);
    const cv::Vec3d ey = ez.cross(ex);
    cv::Mat basis(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
    {
        basis.at<double>(row, 0) = ex[row];
        basis.at<double>(row, 1) = ey[row];
        basis.at<double>(row, 2) = ez[row];
    }
    RectifyMaps maps;
    maps.width = config.rectified_width;
    maps.height = config.rectified_height;
    maps.fx = maps.fy = maps.focal_px =
        maps.width / (2.0 * std::tan(config.horizontal_fov_degrees * CV_PI / 360.0));
    maps.cx = (maps.width - 1) * .5;
    maps.cy = (maps.height - 1) * .5;
    const int crop_rows = static_cast<int>(std::floor(maps.height * config.crop_bottom_ratio));
    const int retained_height = maps.height - crop_rows;
    check(retained_height < 2, "bottom crop must retain at least two rectified rows");
    // Fuse crop/resize into the original remap. Use pixel-center resize geometry.
    const double scale_y = static_cast<double>(maps.height) / retained_height;
    maps.fy *= scale_y;
    maps.cy = (maps.cy + 0.5) * scale_y - 0.5;
    maps.baseline_m = length;
    maps.full_precision = true;
    const cv::Mat r0 = t0(cv::Rect(0, 0, 3, 3));
    const cv::Mat rotation = (r0 * basis).t();
    const cv::Mat origin = r0 * ca + t0(cv::Rect(3, 0, 1, 3));
    for (int row = 0; row < 3; ++row)
    {
        maps.depth_origin_in_left[row] = origin.at<double>(row);
        for (int col = 0; col < 3; ++col)
            maps.rectified_from_left[row * 3 + col] = rotation.at<double>(row, col);
    }
    auto project = [&](const YAML::Node& node, const cv::Mat& r, std::vector<float>& mx,
                        std::vector<float>& my)
    {
        const auto k = yaml_vector(node, "intrinsics", 5);
        const auto d = yaml_vector(node, "distortion_coeffs", 4);
        check(k[0] < 0 || k[1] <= 0 || k[2] <= 0, "invalid omni intrinsics");
        cv::Mat rot = r * basis;
        mx.resize(maps.width * maps.height);
        my.resize(mx.size());
        std::size_t valid_count = 0;
        for (int y = 0; y < maps.height; ++y)
            for (int x = 0; x < maps.width; ++x)
            {
                const cv::Vec3d ray((x - maps.cx) / maps.fx, (y - maps.cy) / maps.fy, 1.0);
                cv::Vec3d v;
                for (int row = 0; row < 3; ++row)
                    v[row] = rot.at<double>(row, 0) * ray[0] + rot.at<double>(row, 1) * ray[1] +
                        rot.at<double>(row, 2);
                const double norm = cv::norm(v), denom = v[2] + k[0] * norm;
                const double px = v[0] / denom, py = v[1] / denom, r2 = px * px + py * py;
                const double radial = 1 + d[0] * r2 + d[1] * r2 * r2;
                const double u =
                    k[1] * (px * radial + 2 * d[2] * px * py + d[3] * (r2 + 2 * px * px)) + k[3];
                const double w =
                    k[2] * (py * radial + d[2] * (r2 + 2 * py * py) + 2 * d[3] * px * py) + k[4];
                const bool valid = std::isfinite(u) && std::isfinite(w) &&
                    v[2] / norm > (k[0] > 1 ? -1 / k[0] : -k[0]) && u >= 1 &&
                    u < config.eye_width - 2 && w >= 1 && w < config.eye_height - 2;
                const auto i = static_cast<std::size_t>(y) * maps.width + x;
                mx[i] = valid ? static_cast<float>(u) : -1.0F;
                my[i] = valid ? static_cast<float>(w) : -1.0F;
                valid_count += valid;
            }
        check(!valid_count, "omni virtual view has no valid source pixels");
    };
    project(left, ra, maps.left_x, maps.left_y);
    project(right, rb, maps.right_x, maps.right_y);
    std::printf("[CALIB] omni pair=%d/%d output=%dx%d hfov=%.2f fx=%.9f baseline=%.9f\n",
        config.left_camera, config.right_camera, maps.width, maps.height,
        config.horizontal_fov_degrees, maps.fx, maps.baseline_m);
    std::printf("[CALIB] bottom_crop=%.6f rows=%d retained=%d fy=%.9f cy=%.9f\n",
        config.crop_bottom_ratio, crop_rows, retained_height, maps.fy, maps.cy);
    return maps;
}

#endif // OMNI_RECTIFICATION_H
