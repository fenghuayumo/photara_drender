#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "asdiff_render/asdiff_render.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float sum_channel(const asdiff_render::RasterizeOutput& output, std::uint32_t channel) {
    float sum = 0.0F;
    for (std::size_t i = channel; i < output.raster.size(); i += 4) {
        sum += output.raster[i];
    }
    return sum;
}

} // namespace

int main() {
    try {
        asdiff_render::Context context;
        asdiff_render::Rasterizer rasterizer(context);
        const std::vector<float> positions{
            -0.75F, -0.75F, 0.0F, 1.0F,
             0.75F, -0.75F, 0.0F, 1.0F,
             0.00F,  0.75F, 0.0F, 1.0F,
        };
        const std::vector<std::uint32_t> indices{0, 1, 2};
        const asdiff_render::RasterizeOptions options{32, 32, asdiff_render::CullMode::none, true};
        const auto output = rasterizer.forward(positions, indices, options);
        require(output.raster.size() == 32 * 32 * 4, "Unexpected raster size");
        require(output.barycentric_derivatives.size() == output.raster.size(), "Unexpected derivative size");
        require(sum_channel(output, 3) > 100.0F, "Triangle did not cover the expected pixels");

        std::vector<float> grad_raster(output.raster.size(), 0.0F);
        for (std::size_t i = 0; i < grad_raster.size(); i += 4) {
            grad_raster[i] = output.raster[i + 3] > 0.0F ? 1.0F : 0.0F;
        }
        const auto analytical_gradient = rasterizer.backward(positions, indices, output, grad_raster);
        require(analytical_gradient.size() == positions.size(), "Unexpected position gradient size");
        require(std::ranges::all_of(analytical_gradient, [](float value) { return std::isfinite(value); }),
                "Position gradient contains non-finite values");

        const float epsilon = 1e-3F;
        auto positive_positions = positions;
        auto negative_positions = positions;
        positive_positions[0] += epsilon;
        negative_positions[0] -= epsilon;
        const float positive_loss = sum_channel(rasterizer.forward(positive_positions, indices, options), 0);
        const float negative_loss = sum_channel(rasterizer.forward(negative_positions, indices, options), 0);
        const float numerical_gradient = (positive_loss - negative_loss) / (2.0F * epsilon);
        require(std::abs(analytical_gradient[0] - numerical_gradient) < 0.2F,
                "Analytical raster gradient failed finite-difference validation");

        const std::vector<float> attributes{
            1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 1.0F,
        };
        const auto interpolated = rasterizer.interpolate_forward(attributes, 3, indices, output);
        require(interpolated.values.size() == 32 * 32 * 3, "Unexpected interpolation output size");
        std::vector<float> grad_interpolated(interpolated.values.size(), 1.0F);
        const auto interpolation_gradients =
            rasterizer.interpolate_backward(attributes, 3, indices, output, grad_interpolated);
        require(interpolation_gradients.attributes.size() == attributes.size(),
                "Unexpected attribute gradient size");
        require(interpolation_gradients.raster.size() == output.raster.size(),
                "Unexpected interpolation raster gradient size");
        require(std::ranges::all_of(interpolation_gradients.attributes, [](float value) { return std::isfinite(value); }),
                "Attribute gradient contains non-finite values");

        std::cout << "device=" << context.device_info().name << '\n';
        std::cout << "finite_difference=" << numerical_gradient
                  << " analytical=" << analytical_gradient[0] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
