#ifndef PREVIEW_REFERENCE_H
#define PREVIEW_REFERENCE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace preview_test
{
struct Rgb
{
    std::uint8_t r, g, b;
};
inline std::uint8_t byte(float value)
{
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}
inline Rgb color(float value, bool disparity, float minimum, float maximum)
{
    if (!std::isfinite(value) || (disparity ? value <= 0 : (value <= minimum || value > maximum)))
        return {0, 0, 0};
    const float t = std::clamp((value - minimum) / (maximum - minimum), 0.0F, 1.0F);
    const auto channel = [t](std::array<float, 6> c)
    {
        float p = c[5];
        for (int i = 4; i >= 0; --i)
            p = p * t + c[i];
        return byte(p * 255.0F);
    };
    return {channel({0.13572138F, 4.61539260F, -42.66032258F, 132.13108234F, -152.94239396F,
                59.28637943F}),
        channel({0.09140261F, 2.19418839F, 4.84296658F, -14.18503333F, 4.27729857F, 2.82956604F}),
        channel({0.10667330F, 12.64194608F, -60.58204836F, 110.36276771F, -89.90310912F,
            27.34824973F})};
}
// Independent reference for the CPU preview reference (including RGB8 rounding).
inline void compose(const std::uint8_t* rgb, int rgb_stride, const float* values, int value_stride,
    int w, int h, bool disparity, float minimum, float maximum, std::uint8_t* output, int ys,
    int uvs)
{
    const auto pixel = [&](int x, int y)
    {
        if (x < w)
        {
            const auto* p = rgb + y * rgb_stride + x * 3;
            return Rgb{p[0], p[1], p[2]};
        }
        const auto* row = reinterpret_cast<const float*>(
            reinterpret_cast<const std::uint8_t*>(values) + y * value_stride);
        return color(row[x - w], disparity, minimum, maximum);
    };
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w * 2; ++x)
        {
            const auto c = pixel(x, y);
            output[y * ys + x] = ((66 * c.r + 129 * c.g + 25 * c.b + 128) >> 8) + 16;
        }
    }
    for (int y = 0; y < h; y += 2)
    {
        for (int x = 0; x < w * 2; x += 2)
        {
            int r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                {
                    const auto c = pixel(x + dx, y + dy);
                    r += c.r;
                    g += c.g;
                    b += c.b;
                }
            r /= 4;
            g /= 4;
            b /= 4;
            auto* uv = output + ys * h + (y / 2) * uvs + x;
            uv[0] = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            uv[1] = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        }
    }
}
} // namespace preview_test

#endif // PREVIEW_REFERENCE_H
