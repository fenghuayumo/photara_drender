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

        const std::vector<float> vertex_uv{
            0.20F, 0.20F,
            0.80F, 0.20F,
            0.50F, 0.80F,
        };
        const auto interpolated_uv = rasterizer.interpolate_forward(vertex_uv, 2, indices, output);
        std::vector<float> texture(4 * 4 * 2);
        for (std::uint32_t y = 0; y < 4; ++y) {
            for (std::uint32_t x = 0; x < 4; ++x) {
                texture[(y * 4 + x) * 2 + 0] = static_cast<float>(x + y * 2);
                texture[(y * 4 + x) * 2 + 1] = static_cast<float>(static_cast<std::int32_t>(x * 3) - static_cast<std::int32_t>(y));
            }
        }
        const asdiff_render::TextureDesc texture_desc{4, 4, 2, asdiff_render::AddressMode::clamp};
        const auto sampled = rasterizer.texture_forward(texture, texture_desc, interpolated_uv.values, output);
        require(sampled.values.size() == 32 * 32 * 2, "Unexpected texture sample output size");
        std::vector<float> grad_sampled(sampled.values.size(), 0.0F);
        std::size_t covered_pixel = 0;
        while (covered_pixel < output.raster.size() / 4 && output.raster[covered_pixel * 4 + 3] == 0.0F) {
            ++covered_pixel;
        }
        require(covered_pixel < output.raster.size() / 4, "No covered pixel found for texture gradient test");
        grad_sampled[covered_pixel * 2] = 1.0F;
        const auto texture_gradients =
            rasterizer.texture_backward(texture, texture_desc, interpolated_uv.values, output, grad_sampled);
        require(texture_gradients.texture.size() == texture.size(), "Unexpected texture gradient size");
        require(texture_gradients.uv.size() == interpolated_uv.values.size(), "Unexpected UV gradient size");

        auto positive_uv = interpolated_uv.values;
        auto negative_uv = interpolated_uv.values;
        positive_uv[covered_pixel * 2] += epsilon;
        negative_uv[covered_pixel * 2] -= epsilon;
        const auto positive_sample = rasterizer.texture_forward(texture, texture_desc, positive_uv, output);
        const auto negative_sample = rasterizer.texture_forward(texture, texture_desc, negative_uv, output);
        const float numerical_uv_gradient =
            (positive_sample.values[covered_pixel * 2] - negative_sample.values[covered_pixel * 2]) / (2.0F * epsilon);
        require(std::abs(texture_gradients.uv[covered_pixel * 2] - numerical_uv_gradient) < 1e-3F,
                "Analytical UV gradient failed finite-difference validation");

        std::size_t gradient_texel = 0;
        while (gradient_texel < texture_gradients.texture.size() &&
               std::abs(texture_gradients.texture[gradient_texel]) < 1e-8F) {
            ++gradient_texel;
        }
        require(gradient_texel < texture_gradients.texture.size(), "No texture gradient contribution was generated");
        auto positive_texture = texture;
        auto negative_texture = texture;
        positive_texture[gradient_texel] += epsilon;
        negative_texture[gradient_texel] -= epsilon;
        const auto positive_texture_sample =
            rasterizer.texture_forward(positive_texture, texture_desc, interpolated_uv.values, output);
        const auto negative_texture_sample =
            rasterizer.texture_forward(negative_texture, texture_desc, interpolated_uv.values, output);
        const float numerical_texture_gradient =
            (positive_texture_sample.values[covered_pixel * 2] - negative_texture_sample.values[covered_pixel * 2]) /
            (2.0F * epsilon);
        require(std::abs(texture_gradients.texture[gradient_texel] - numerical_texture_gradient) < 1e-3F,
                "Analytical texture gradient failed finite-difference validation");

        auto viewport_options = options;
        viewport_options.viewport = asdiff_render::Viewport{8.0F, 4.0F, 16.0F, 20.0F};
        const auto viewport_output = rasterizer.forward(positions, indices, viewport_options);
        for (std::uint32_t y = 0; y < viewport_output.height; ++y) {
            for (std::uint32_t x = 0; x < viewport_output.width; ++x) {
                const bool outside_viewport = x < 8 || x >= 24 || y < 4 || y >= 24;
                if (outside_viewport) {
                    require(viewport_output.raster[(y * viewport_output.width + x) * 4 + 3] == 0.0F,
                            "Viewport rasterization wrote outside the viewport");
                }
            }
        }
        const auto viewport_gradient = rasterizer.backward(positions, indices, viewport_output, grad_raster);
        require(std::ranges::all_of(viewport_gradient, [](float value) { return std::isfinite(value); }),
                "Viewport position gradient contains non-finite values");

        const std::vector<float> world_positions{
            -0.75F, -0.75F, 0.0F,
             0.75F, -0.75F, 0.0F,
             0.00F,  0.75F, 0.0F,
        };
        const std::vector<float> world_normals{
            0.0F, 0.0F, 1.0F,
            0.0F, 0.0F, 1.0F,
            0.0F, 0.0F, 1.0F,
        };
        const std::vector<float> bake_uv{
            0.125F, 0.125F,
            0.875F, 0.125F,
            0.500F, 0.875F,
        };
        if (asdiff_render::has_uv_atlas_backend()) {
            asdiff_render::UvAtlasOptions unwrap_options;
            unwrap_options.width = 64;
            unwrap_options.height = 64;
            unwrap_options.gutter = 2.0F;
            const auto unwrapped = asdiff_render::unwrap_uv(world_positions, indices, unwrap_options);
            require(!unwrapped.positions.empty() && unwrapped.uv.size() / 2 == unwrapped.positions.size() / 3,
                    "UVAtlas returned invalid vertex data");
            require(unwrapped.indices.size() == indices.size(), "UVAtlas returned an invalid index count");
            require(unwrapped.face_chart_ids.size() == indices.size() / 3, "UVAtlas omitted face chart IDs");

            unwrap_options.parallel_partitions = 2;
            const auto parallel_unwrapped =
                asdiff_render::unwrap_uv(world_positions, indices, unwrap_options);
            require(parallel_unwrapped.indices.size() == indices.size(),
                    "parallel UVAtlas returned an invalid index count");
            require(parallel_unwrapped.face_chart_ids.size() == indices.size() / 3,
                    "parallel UVAtlas omitted face chart IDs");
            require(parallel_unwrapped.vertex_remap.size() ==
                        parallel_unwrapped.positions.size() / 3,
                    "parallel UVAtlas returned an invalid vertex remap");
            require(parallel_unwrapped.partition_count >= 1,
                    "parallel UVAtlas omitted partition_count");
            require(
                std::all_of(
                    parallel_unwrapped.indices.begin(),
                    parallel_unwrapped.indices.end(),
                    [&](std::uint32_t index) {
                        return index < parallel_unwrapped.positions.size() / 3;
                    }),
                "parallel UVAtlas returned an out-of-range index");
            require(
                std::all_of(
                    parallel_unwrapped.uv.begin(),
                    parallel_unwrapped.uv.end(),
                    [](float value) {
                        return std::isfinite(value) && value >= -1e-3F && value <= 1.0F + 1e-3F;
                    }),
                "parallel UVAtlas returned UV coordinates outside the atlas");

            const std::vector<float> non_manifold_positions{
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 1.0F,
            };
            const std::vector<std::uint32_t> non_manifold_indices{
                0, 1, 2, 1, 0, 3, 0, 1, 4,
            };
            bool rejected_non_manifold = false;
            try {
                static_cast<void>(asdiff_render::unwrap_uv(
                    non_manifold_positions, non_manifold_indices, unwrap_options));
            } catch (const std::invalid_argument&) {
                rejected_non_manifold = true;
            }
            require(rejected_non_manifold, "UVAtlas accepted a non-manifold edge");
        }

        asdiff_render::ProjectionView projection_view;
        projection_view.width = 16;
        projection_view.height = 16;
        projection_view.channel_count = 4;
        projection_view.image.resize(16 * 16 * 4);
        for (std::size_t pixel = 0; pixel < 16 * 16; ++pixel) {
            projection_view.image[pixel * 4 + 0] = 0.8F;
            projection_view.image[pixel * 4 + 1] = 0.2F;
            projection_view.image[pixel * 4 + 2] = 0.1F;
            projection_view.image[pixel * 4 + 3] = 1.0F;
        }
        projection_view.world_to_clip = {
            1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F,
        };
        projection_view.camera_position = {0.0F, 0.0F, 2.0F};
        asdiff_render::TextureBakeOptions bake_options;
        bake_options.width = 16;
        bake_options.height = 16;
        bake_options.blend_mode = asdiff_render::ProjectionBlendMode::weighted_average;
        asdiff_render::TextureBaker texture_baker(context);
        const auto baked = texture_baker.bake(
            world_positions, world_normals, bake_uv, indices,
            std::span<const asdiff_render::ProjectionView>(&projection_view, 1), bake_options);
        require(baked.color.size() == 16 * 16 * 4, "Unexpected baked atlas size");
        require(std::ranges::any_of(baked.valid_mask, [](float value) { return value > 0.5F; }),
                "Texture projection did not produce valid atlas texels");
        const auto first_valid = std::ranges::find_if(baked.valid_mask, [](float value) { return value > 0.5F; });
        const auto valid_index = static_cast<std::size_t>(first_valid - baked.valid_mask.begin());
        require(std::abs(baked.color[valid_index * 4 + 0] - 0.8F) < 1e-4F,
                "Texture projection returned an incorrect color");
        if (context.device_info().supports_ray_query) {
            bake_options.visibility_mode = asdiff_render::VisibilityMode::hybrid_ray_query;
            bake_options.allow_visibility_fallback = false;
            const auto ray_baked = texture_baker.bake(
                world_positions, world_normals, bake_uv, indices,
                std::span<const asdiff_render::ProjectionView>(&projection_view, 1), bake_options);
            require(ray_baked.used_ray_query, "Hybrid texture projection did not use Vulkan ray queries");
            require(std::ranges::any_of(ray_baked.valid_mask, [](float value) { return value > 0.5F; }),
                    "Ray-query texture projection rejected every visible texel");
        }

        std::cout << "device=" << context.device_info().name << '\n';
        std::cout << "finite_difference=" << numerical_gradient
                  << " analytical=" << analytical_gradient[0] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
