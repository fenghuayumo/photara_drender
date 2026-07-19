#include <iostream>

#include "asdiff_render/asdiff_render.hpp"

int main() {
    const auto devices = asdiff_render::Context::enumerate_devices();
    std::cout << devices.size() << '\n';
    if (devices.empty()) {
        return 1;
    }
    asdiff_render::Context context;
    asdiff_render::Rasterizer rasterizer(context);
    const float positions[] = {
        -0.5F, -0.5F, 0.0F, 1.0F,
         0.5F, -0.5F, 0.0F, 1.0F,
         0.0F,  0.5F, 0.0F, 1.0F,
    };
    const std::uint32_t indices[] = {0, 1, 2};
    const auto output = rasterizer.forward(positions, indices, {8, 8});
    return output.raster.empty() ? 1 : 0;
}
