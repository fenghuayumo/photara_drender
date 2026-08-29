#include "aether_drender/texture_baker.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "aether_drender/rasterizer.hpp"
#include "context_internal.hpp"

namespace aether_drender {
namespace {

constexpr std::uint32_t PROJECTION_BLOCK_SIZE = 64;

std::uint32_t divide_round_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

VkDescriptorBufferInfo descriptor(const Buffer& buffer) {
    return {buffer.handle, 0, buffer.size};
}

Viewport resolve_viewport(const ProjectionView& view) {
    const auto viewport = view.viewport.value_or(
        Viewport{0.0F, 0.0F, static_cast<float>(view.width), static_cast<float>(view.height)});
    if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
        !std::isfinite(viewport.height) || viewport.width <= 0.0F || viewport.height <= 0.0F) {
        throw std::invalid_argument("projection viewport must contain finite values and positive dimensions");
    }
    return viewport;
}

void validate_mesh(
    std::span<const float> positions,
    std::span<const float> normals,
    std::span<const float> uv,
    std::span<const std::uint32_t> indices) {
    if (positions.empty() || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    const auto vertex_count = positions.size() / 3;
    if (normals.size() != positions.size()) {
        throw std::invalid_argument("normals must have shape [vertex_count, 3]");
    }
    if (uv.size() != vertex_count * 2) {
        throw std::invalid_argument("uv must have shape [vertex_count, 2]");
    }
    if (indices.empty() || indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    if (std::ranges::any_of(indices, [vertex_count](std::uint32_t index) { return index >= vertex_count; })) {
        throw std::out_of_range("triangle_indices contains an invalid vertex index");
    }
}

void validate_view(const ProjectionView& view) {
    if (view.width == 0 || view.height == 0 || (view.channel_count != 3 && view.channel_count != 4)) {
        throw std::invalid_argument("projection image must have nonzero dimensions and 3 or 4 channels");
    }
    const auto pixel_count = static_cast<std::size_t>(view.width) * view.height;
    if (view.image.size() != pixel_count * view.channel_count) {
        throw std::invalid_argument("projection image data size does not match its descriptor");
    }
    if (!view.visibility_mask.empty() && view.visibility_mask.size() != pixel_count) {
        throw std::invalid_argument("projection visibility_mask must have shape [height, width]");
    }
    if (!std::isfinite(view.weight) || view.weight < 0.0F || !std::isfinite(view.depth_bias) ||
        view.depth_bias < 0.0F || !std::isfinite(view.min_view_cosine) || view.min_view_cosine < -1.0F ||
        view.min_view_cosine > 1.0F) {
        throw std::invalid_argument("projection confidence parameters are invalid");
    }
    resolve_viewport(view);
}

std::vector<float> make_atlas_clip_positions(std::span<const float> uv) {
    std::vector<float> result(uv.size() / 2 * 4);
    for (std::size_t vertex = 0; vertex < uv.size() / 2; ++vertex) {
        result[vertex * 4 + 0] = uv[vertex * 2 + 0] * 2.0F - 1.0F;
        result[vertex * 4 + 1] = uv[vertex * 2 + 1] * 2.0F - 1.0F;
        result[vertex * 4 + 2] = 0.0F;
        result[vertex * 4 + 3] = 1.0F;
    }
    return result;
}

std::vector<float> transform_positions(
    std::span<const float> positions,
    const std::array<float, 16>& matrix) {
    std::vector<float> result(positions.size() / 3 * 4);
    for (std::size_t vertex = 0; vertex < positions.size() / 3; ++vertex) {
        const float x = positions[vertex * 3 + 0];
        const float y = positions[vertex * 3 + 1];
        const float z = positions[vertex * 3 + 2];
        for (std::uint32_t row = 0; row < 4; ++row) {
            result[vertex * 4 + row] = matrix[row * 4 + 0] * x + matrix[row * 4 + 1] * y +
                                       matrix[row * 4 + 2] * z + matrix[row * 4 + 3];
        }
    }
    return result;
}

struct ProjectionPushConstants {
    std::array<float, 16> world_to_clip;
    std::array<float, 4> camera_position_weight;
    std::array<float, 4> viewport;
    std::array<std::uint32_t, 4> dimensions;
    std::array<float, 4> settings;
};

static_assert(sizeof(ProjectionPushConstants) == 128);

struct RayVisibilityPushConstants {
    std::array<float, 4> camera_position_bias;
    std::array<std::uint32_t, 4> dimensions;
};

static_assert(sizeof(RayVisibilityPushConstants) == 32);

} // namespace

class TextureBaker::Impl {
public:
    Impl(Context::Impl& context, Context& public_context)
        : context_(context), rasterizer_(public_context),
          projection_pipeline_(context_.create_pipeline("project_texture.hlsl.spv", 11, sizeof(ProjectionPushConstants))) {
        if (context_.device_info.supports_ray_query) {
            const std::array descriptor_types{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            };
            ray_visibility_pipeline_.emplace(context_.create_pipeline(
                "project_ray_visibility.hlsl.spv", descriptor_types, sizeof(RayVisibilityPushConstants)));
        }
    }

    Context::Impl& context_;
    Rasterizer rasterizer_;
    ComputePipeline projection_pipeline_;
    std::optional<ComputePipeline> ray_visibility_pipeline_;
};

TextureBaker::TextureBaker(Context& context) : impl_(std::make_unique<Impl>(*context.impl_, context)) {}
TextureBaker::~TextureBaker() = default;
TextureBaker::TextureBaker(TextureBaker&&) noexcept = default;
TextureBaker& TextureBaker::operator=(TextureBaker&&) noexcept = default;

TextureBakeOutput TextureBaker::bake(
    std::span<const float> positions,
    std::span<const float> normals,
    std::span<const float> uv,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const ProjectionView> views,
    const TextureBakeOptions& options) {
    validate_mesh(positions, normals, uv, triangle_indices);
    if (views.empty()) {
        throw std::invalid_argument("at least one projection view is required");
    }
    if (options.width == 0 || options.height == 0 || options.pcf_radius > 4 ||
        !std::isfinite(options.ray_origin_bias) || options.ray_origin_bias <= 0.0F) {
        throw std::invalid_argument("atlas dimensions are invalid or pcf_radius exceeds 4");
    }
    const bool ray_visibility_requested = options.visibility_mode != VisibilityMode::shadow_map;
    if (ray_visibility_requested && !impl_->context_.device_info.supports_ray_query &&
        !options.allow_visibility_fallback) {
        throw std::runtime_error("ray-query visibility was requested but the selected Vulkan device does not support it");
    }
    for (const auto& view : views) {
        validate_view(view);
    }

    const auto atlas_clip_positions = make_atlas_clip_positions(uv);
    RasterizeOptions atlas_raster_options;
    atlas_raster_options.width = options.width;
    atlas_raster_options.height = options.height;
    atlas_raster_options.output_barycentric_derivatives = false;
    const auto atlas_raster = impl_->rasterizer_.forward(atlas_clip_positions, triangle_indices, atlas_raster_options);
    const auto atlas_positions = impl_->rasterizer_.interpolate_forward(
        positions, 3, triangle_indices, atlas_raster);
    const auto atlas_normals = impl_->rasterizer_.interpolate_forward(
        normals, 3, triangle_indices, atlas_raster);
    const auto atlas_pixel_count = static_cast<std::size_t>(options.width) * options.height;

    auto atlas_position_buffer = impl_->context_.create_buffer(atlas_positions.values.size() * sizeof(float), 0);
    auto atlas_normal_buffer = impl_->context_.create_buffer(atlas_normals.values.size() * sizeof(float), 0);
    auto atlas_raster_buffer = impl_->context_.create_buffer(atlas_raster.raster.size() * sizeof(float), 0);
    auto accumulated_color_buffer = impl_->context_.create_buffer(atlas_pixel_count * 3 * sizeof(float), 0);
    auto accumulated_weight_buffer = impl_->context_.create_buffer(atlas_pixel_count * sizeof(float), 0);
    auto best_confidence_buffer = impl_->context_.create_buffer(atlas_pixel_count * sizeof(float), 0);
    auto source_view_buffer = impl_->context_.create_buffer(atlas_pixel_count * sizeof(std::uint32_t), 0);
    std::vector<float> ray_visibility(atlas_pixel_count, 1.0F);
    auto ray_visibility_buffer = impl_->context_.create_buffer(ray_visibility.size() * sizeof(float), 0);
    atlas_position_buffer.upload(atlas_positions.values.data(), atlas_positions.values.size() * sizeof(float));
    atlas_normal_buffer.upload(atlas_normals.values.data(), atlas_normals.values.size() * sizeof(float));
    atlas_raster_buffer.upload(atlas_raster.raster.data(), atlas_raster.raster.size() * sizeof(float));
    ray_visibility_buffer.upload(ray_visibility.data(), ray_visibility.size() * sizeof(float));
    std::vector<float> zero_color(atlas_pixel_count * 3, 0.0F);
    std::vector<float> zero_scalar(atlas_pixel_count, 0.0F);
    accumulated_color_buffer.upload(zero_color.data(), zero_color.size() * sizeof(float));
    accumulated_weight_buffer.upload(zero_scalar.data(), zero_scalar.size() * sizeof(float));
    best_confidence_buffer.upload(zero_scalar.data(), zero_scalar.size() * sizeof(float));
    std::vector<std::uint32_t> no_source(atlas_pixel_count, std::numeric_limits<std::uint32_t>::max());
    source_view_buffer.upload(no_source.data(), no_source.size() * sizeof(std::uint32_t));
    const bool use_ray_query = ray_visibility_requested && impl_->context_.device_info.supports_ray_query;
    const auto effective_visibility_mode = use_ray_query
        ? options.visibility_mode
        : VisibilityMode::shadow_map;
    const bool use_shadow_map = effective_visibility_mode != VisibilityMode::ray_query;
    std::optional<RayQueryScene> ray_scene;
    if (use_ray_query) {
        ray_scene.emplace(impl_->context_.create_ray_query_scene(positions, triangle_indices));
    }

    for (std::uint32_t view_index = 0; view_index < views.size(); ++view_index) {
        const auto& view = views[view_index];
        if (use_ray_query) {
            RayVisibilityPushConstants ray_push{};
            ray_push.camera_position_bias = {
                view.camera_position[0], view.camera_position[1], view.camera_position[2], options.ray_origin_bias};
            ray_push.dimensions = {options.width, options.height, 0, 0};
            impl_->context_.dispatch_ray_query(
                *impl_->ray_visibility_pipeline_, atlas_position_buffer, atlas_normal_buffer, atlas_raster_buffer,
                ray_scene->top_level, ray_visibility_buffer, &ray_push, sizeof(ray_push),
                divide_round_up(static_cast<std::uint32_t>(atlas_pixel_count), PROJECTION_BLOCK_SIZE));
        }
        std::vector<float> shadow_raster_data{0.0F};
        if (use_shadow_map) {
            const auto clip_positions = transform_positions(positions, view.world_to_clip);
            RasterizeOptions shadow_options;
            shadow_options.width = view.width;
            shadow_options.height = view.height;
            shadow_options.output_barycentric_derivatives = false;
            shadow_options.viewport = view.viewport;
            auto shadow_raster = impl_->rasterizer_.forward(clip_positions, triangle_indices, shadow_options);
            shadow_raster_data = std::move(shadow_raster.raster);
        }
        std::vector<float> default_mask{1.0F};
        const std::span<const float> mask = view.visibility_mask.empty()
            ? std::span<const float>(default_mask)
            : std::span<const float>(view.visibility_mask);
        auto shadow_buffer = impl_->context_.create_buffer(shadow_raster_data.size() * sizeof(float), 0);
        auto photo_buffer = impl_->context_.create_buffer(view.image.size() * sizeof(float), 0);
        auto mask_buffer = impl_->context_.create_buffer(mask.size() * sizeof(float), 0);
        shadow_buffer.upload(shadow_raster_data.data(), shadow_raster_data.size() * sizeof(float));
        photo_buffer.upload(view.image.data(), view.image.size() * sizeof(float));
        mask_buffer.upload(mask.data(), mask.size() * sizeof(float));

        const auto viewport = resolve_viewport(view);
        ProjectionPushConstants push_constants{};
        push_constants.world_to_clip = view.world_to_clip;
        push_constants.camera_position_weight = {
            view.camera_position[0], view.camera_position[1], view.camera_position[2], view.weight};
        push_constants.viewport = {viewport.x, viewport.y, viewport.width, viewport.height};
        push_constants.dimensions = {options.width, options.height, view.width, view.height};
        const std::uint32_t packed = view.channel_count |
            (static_cast<std::uint32_t>(options.blend_mode) << 8U) |
            (options.pcf_radius << 16U) |
            (view.visibility_mask.empty() ? 0U : (1U << 24U)) |
            (static_cast<std::uint32_t>(effective_visibility_mode) << 25U);
        push_constants.settings = {
            view.depth_bias,
            view.min_view_cosine,
            std::bit_cast<float>(packed),
            std::bit_cast<float>(view_index),
        };
        const std::array descriptors{
            descriptor(atlas_position_buffer), descriptor(atlas_normal_buffer), descriptor(atlas_raster_buffer),
            descriptor(shadow_buffer), descriptor(photo_buffer), descriptor(mask_buffer), descriptor(ray_visibility_buffer),
            descriptor(accumulated_color_buffer), descriptor(accumulated_weight_buffer),
            descriptor(best_confidence_buffer), descriptor(source_view_buffer),
        };
        impl_->context_.dispatch(
            impl_->projection_pipeline_, descriptors, &push_constants, sizeof(push_constants),
            divide_round_up(static_cast<std::uint32_t>(atlas_pixel_count), PROJECTION_BLOCK_SIZE));
    }

    TextureBakeOutput output;
    output.width = options.width;
    output.height = options.height;
    output.color.resize(atlas_pixel_count * 4);
    output.confidence.resize(atlas_pixel_count);
    output.source_view.resize(atlas_pixel_count);
    output.valid_mask.resize(atlas_pixel_count);
    std::vector<float> accumulated_color(atlas_pixel_count * 3);
    accumulated_color_buffer.download(accumulated_color.data(), accumulated_color.size() * sizeof(float));
    accumulated_weight_buffer.download(output.confidence.data(), output.confidence.size() * sizeof(float));
    source_view_buffer.download(output.source_view.data(), output.source_view.size() * sizeof(std::uint32_t));
    for (std::size_t pixel = 0; pixel < atlas_pixel_count; ++pixel) {
        const float weight = output.confidence[pixel];
        output.valid_mask[pixel] = weight > 0.0F ? 1.0F : 0.0F;
        const float divisor = options.blend_mode == ProjectionBlendMode::weighted_average && weight > 0.0F ? weight : 1.0F;
        output.color[pixel * 4 + 0] = accumulated_color[pixel * 3 + 0] / divisor;
        output.color[pixel * 4 + 1] = accumulated_color[pixel * 3 + 1] / divisor;
        output.color[pixel * 4 + 2] = accumulated_color[pixel * 3 + 2] / divisor;
        output.color[pixel * 4 + 3] = output.valid_mask[pixel];
    }
    output.used_ray_query = use_ray_query;
    return output;
}

} // namespace aether_drender
