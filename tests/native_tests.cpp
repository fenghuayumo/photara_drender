#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "photara_drender/photara_drender.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr std::uint32_t k_padding_test_size = 24;

std::size_t padding_test_index(std::uint32_t x, std::uint32_t y) {
    return static_cast<std::size_t>(y) * k_padding_test_size + x;
}

float sum_channel(const photara_drender::RasterizeOutput& output, std::uint32_t channel) {
    float sum = 0.0F;
    for (std::size_t i = channel; i < output.raster.size(); i += 4) {
        sum += output.raster[i];
    }
    return sum;
}

// Atlas padding runs on the CPU, so it is validated before any Vulkan work.
void test_texture_padding() {
    constexpr std::uint32_t size = k_padding_test_size;
    const auto index_of = [](std::uint32_t x, std::uint32_t y) { return padding_test_index(x, y); };
    const std::array<float, 4> source_color{0.25F, 0.5F, 0.75F, 1.0F};

    photara_drender::TextureBakeOutput output;
    output.width = size;
    output.height = size;
    output.color.assign(static_cast<std::size_t>(size) * size * 4, 0.0F);
    output.valid_mask.assign(static_cast<std::size_t>(size) * size, 0.0F);
    output.coverage_mask.assign(static_cast<std::size_t>(size) * size, 0.0F);
    // A 4x4 block of projected texels ...
    for (std::uint32_t y = 2; y < 6; ++y) {
        for (std::uint32_t x = 2; x < 6; ++x) {
            const std::size_t pixel = index_of(x, y);
            output.valid_mask[pixel] = 1.0F;
            output.coverage_mask[pixel] = 1.0F;
            std::copy(source_color.begin(), source_color.end(), output.color.begin() + pixel * 4);
        }
    }
    // ... and a rasterized chart that never received a sample (masked out,
    // grazing, or occluded in every view).
    for (std::uint32_t y = 10; y < 18; ++y) {
        for (std::uint32_t x = 10; x < 18; ++x) {
            output.coverage_mask[index_of(x, y)] = 1.0F;
        }
    }

    photara_drender::TexturePaddingOptions options;
    options.margin = 3;
    options.fill_unobserved = true;
    const auto stats = photara_drender::pad_texture_atlas(output, options);
    require(stats.valid_texels == 16, "Padding counted the wrong number of valid texels");
    require(stats.filled_texels == stats.gutter_texels + stats.unobserved_texels,
            "Padding statistics are inconsistent");
    require(stats.gutter_texels > 0, "Padding did not fill any gutter texel");
    require(stats.unobserved_texels == 64, "Padding did not fill the unobserved chart texels");
    require(stats.remaining_unobserved == 0, "Padding left chart texels unfilled");

    const auto matches_source = [&](std::uint32_t x, std::uint32_t y) {
        const auto* pixel = output.color.data() + index_of(x, y) * 4;
        return std::abs(pixel[0] - source_color[0]) < 1e-6F &&
               std::abs(pixel[1] - source_color[1]) < 1e-6F &&
               std::abs(pixel[2] - source_color[2]) < 1e-6F && pixel[3] == 1.0F;
    };
    require(matches_source(2, 6), "Gutter texel inside the margin was not extended");
    require(matches_source(10, 10), "Unobserved chart texel was not filled from the nearest valid texel");
    require(output.filled_mask[index_of(10, 10)] > 0.5F, "Filled texel was not recorded in filled_mask");
    require(output.filled_mask[index_of(2, 2)] < 0.5F, "Valid texels must not be marked as filled");
    require(output.valid_mask[index_of(10, 10)] < 0.5F,
            "Padding must not widen the projection-valid mask");
    require(output.color[index_of(2, 12) * 4 + 0] == 0.0F,
            "Padding wrote outside the guard band and the chart coverage");

    photara_drender::TextureBakeOutput untouched;
    untouched.width = size;
    untouched.height = size;
    untouched.color.assign(static_cast<std::size_t>(size) * size * 4, 0.0F);
    untouched.valid_mask.assign(static_cast<std::size_t>(size) * size, 0.0F);
    untouched.coverage_mask.assign(static_cast<std::size_t>(size) * size, 0.0F);
    for (std::uint32_t y = 2; y < 6; ++y) {
        for (std::uint32_t x = 2; x < 6; ++x) {
            const std::size_t pixel = index_of(x, y);
            untouched.valid_mask[pixel] = 1.0F;
            untouched.coverage_mask[pixel] = 1.0F;
        }
    }
    for (std::uint32_t y = 10; y < 18; ++y) {
        for (std::uint32_t x = 10; x < 18; ++x) {
            untouched.coverage_mask[index_of(x, y)] = 1.0F;
        }
    }
    photara_drender::TexturePaddingOptions disabled;
    disabled.margin = 0;
    disabled.fill_unobserved = false;
    const auto disabled_stats = photara_drender::pad_texture_atlas(untouched, disabled);
    require(disabled_stats.filled_texels == 0, "Disabled padding must not fill any texel");
    require(disabled_stats.remaining_unobserved == 64,
            "Disabled padding must report the chart texels it left untouched");

    const std::vector<float> vertices{0.0F, 0.0F, 0.0F};
    const std::vector<float> cameras{0.0F, 0.0F, 2.0F};
    const float scale = photara_drender::suggest_softmax_scale(vertices, cameras);
    require(std::abs(scale - 80.0F) < 1e-3F, "Softmax scale heuristic returned an unexpected value");
}

} // namespace

int main() {
    try {
        test_texture_padding();
        photara_drender::Context context;
        photara_drender::Rasterizer rasterizer(context);
        const std::vector<float> positions{
            -0.75F, -0.75F, 0.0F, 1.0F,
             0.75F, -0.75F, 0.0F, 1.0F,
             0.00F,  0.75F, 0.0F, 1.0F,
        };
        const std::vector<std::uint32_t> indices{0, 1, 2};
        const photara_drender::RasterizeOptions options{32, 32, photara_drender::CullMode::none, true};
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
        const photara_drender::TextureDesc texture_desc{4, 4, 2, photara_drender::AddressMode::clamp};
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
        viewport_options.viewport = photara_drender::Viewport{8.0F, 4.0F, 16.0F, 20.0F};
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
        if (photara_drender::has_uv_atlas_backend()) {
            photara_drender::UvAtlasOptions unwrap_options;
            unwrap_options.width = 64;
            unwrap_options.height = 64;
            unwrap_options.gutter = 2.0F;
            const auto unwrapped = photara_drender::unwrap_uv(world_positions, indices, unwrap_options);
            require(!unwrapped.positions.empty() && unwrapped.uv.size() / 2 == unwrapped.positions.size() / 3,
                    "UVAtlas returned invalid vertex data");
            require(unwrapped.indices.size() == indices.size(), "UVAtlas returned an invalid index count");
            require(unwrapped.face_chart_ids.size() == indices.size() / 3, "UVAtlas omitted face chart IDs");

            unwrap_options.parallel_partitions = 2;
            const auto parallel_unwrapped =
                photara_drender::unwrap_uv(world_positions, indices, unwrap_options);
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
                static_cast<void>(photara_drender::unwrap_uv(
                    non_manifold_positions, non_manifold_indices, unwrap_options));
            } catch (const std::invalid_argument&) {
                rejected_non_manifold = true;
            }
            require(rejected_non_manifold, "UVAtlas accepted a non-manifold edge");
        }

        photara_drender::ProjectionView projection_view;
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
        photara_drender::TextureBakeOptions bake_options;
        bake_options.width = 16;
        bake_options.height = 16;
        bake_options.blend_mode = photara_drender::ProjectionBlendMode::weighted_average;
        photara_drender::TextureBaker texture_baker(context);
        const auto baked = texture_baker.bake(
            world_positions, world_normals, bake_uv, indices,
            std::span<const photara_drender::ProjectionView>(&projection_view, 1), bake_options);
        require(baked.color.size() == 16 * 16 * 4, "Unexpected baked atlas size");
        require(std::ranges::any_of(baked.valid_mask, [](float value) { return value > 0.5F; }),
                "Texture projection did not produce valid atlas texels");
        if (context.device_info().supports_ray_query) {
            require(baked.used_ray_query, "Default texture projection did not use Vulkan ray queries");
        }
        const auto first_valid = std::ranges::find_if(baked.valid_mask, [](float value) { return value > 0.5F; });
        const auto valid_index = static_cast<std::size_t>(first_valid - baked.valid_mask.begin());
        require(std::abs(baked.color[valid_index * 4 + 0] - 0.8F) < 1e-4F,
                "Texture projection returned an incorrect color");
        if (context.device_info().supports_ray_query) {
            bake_options.visibility_mode = photara_drender::VisibilityMode::ray_query;
            bake_options.allow_visibility_fallback = false;
            const auto ray_baked = texture_baker.bake(
                world_positions, world_normals, bake_uv, indices,
                std::span<const photara_drender::ProjectionView>(&projection_view, 1), bake_options);
            require(ray_baked.used_ray_query, "Texture projection did not use Vulkan ray queries");
            require(std::ranges::any_of(ray_baked.valid_mask, [](float value) { return value > 0.5F; }),
                    "Ray-query texture projection rejected every visible texel");
        }

        // Pixel-footprint softmax: the near view must dominate the atlas instead
        // of being averaged with a distant, lower-resolution view.
        photara_drender::ProjectionView far_view = projection_view;
        far_view.camera_position = {0.0F, 0.0F, 6.0F};
        for (std::size_t pixel = 0; pixel < 16 * 16; ++pixel) {
            far_view.image[pixel * 4 + 0] = 0.1F;
            far_view.image[pixel * 4 + 1] = 0.2F;
            far_view.image[pixel * 4 + 2] = 0.8F;
        }
        const std::array<photara_drender::ProjectionView, 2> two_views{projection_view, far_view};
        const std::span<const photara_drender::ProjectionView> two_view_span(two_views.data(), two_views.size());
        auto average_options = bake_options;
        average_options.blend_mode = photara_drender::ProjectionBlendMode::weighted_average;
        average_options.softmax_scale = 0.0F;
        const auto averaged =
            texture_baker.bake(world_positions, world_normals, bake_uv, indices, two_view_span, average_options);
        auto softmax_options = bake_options;
        softmax_options.blend_mode = photara_drender::ProjectionBlendMode::softmax;
        softmax_options.softmax_scale = 0.0F;
        const auto softmax_baked =
            texture_baker.bake(world_positions, world_normals, bake_uv, indices, two_view_span, softmax_options);
        require(softmax_baked.valid_mask.size() == softmax_baked.coverage_mask.size(),
                "Coverage mask was not reported with the same shape as the valid mask");
        const auto softmax_valid = std::ranges::find_if(
            softmax_baked.valid_mask, [](float value) { return value > 0.5F; });
        require(softmax_valid != softmax_baked.valid_mask.end(), "Softmax blending produced no valid texel");
        const auto softmax_index = static_cast<std::size_t>(softmax_valid - softmax_baked.valid_mask.begin());
        require(softmax_baked.coverage_mask[softmax_index] > 0.5F,
                "Rasterized texels must be reported as covered");
        require(softmax_baked.color[softmax_index * 4 + 0] > averaged.color[softmax_index * 4 + 0] + 0.05F,
                "Softmax blending did not favour the near view over the linear average");
        require(std::abs(softmax_baked.color[softmax_index * 4 + 0] - 0.8F) < 0.05F,
                "Softmax blending is not dominated by the best-resolving view");

        // Mask floor: object masks hide the volume behind a subject, so a
        // masked-out but visible surface must keep its texture as a weak sample
        // instead of being vetoed outright.
        photara_drender::ProjectionView masked_view = projection_view;
        masked_view.visibility_mask.assign(16 * 16, 0.0F);
        const auto masked_span =
            std::span<const photara_drender::ProjectionView>(&masked_view, 1);
        auto strict_mask_options = bake_options;
        strict_mask_options.blend_mode = photara_drender::ProjectionBlendMode::softmax;
        strict_mask_options.mask_floor = 0.0F;
        const auto strict_masked_baked =
            texture_baker.bake(world_positions, world_normals, bake_uv, indices, masked_span, strict_mask_options);
        require(std::ranges::none_of(
                    strict_masked_baked.valid_mask, [](float value) { return value > 0.0F; }),
                "A strict mask must reject every masked pixel");
        auto floored_mask_options = strict_mask_options;
        floored_mask_options.mask_floor = 0.1F;
        const auto floored_masked_baked =
            texture_baker.bake(world_positions, world_normals, bake_uv, indices, masked_span, floored_mask_options);
        const auto floored_valid = std::ranges::find_if(
            floored_masked_baked.valid_mask, [](float value) { return value > 0.5F; });
        require(floored_valid != floored_masked_baked.valid_mask.end(),
                "A mask floor must keep masked pixels as weak samples");
        const auto floored_index =
            static_cast<std::size_t>(floored_valid - floored_masked_baked.valid_mask.begin());
        require(std::abs(floored_masked_baked.color[floored_index * 4 + 0] - 0.8F) < 1e-3F,
                "Mask-floor samples must carry the photograph colour");

        std::cout << "device=" << context.device_info().name << '\n';
        std::cout << "finite_difference=" << numerical_gradient
                  << " analytical=" << analytical_gradient[0] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
