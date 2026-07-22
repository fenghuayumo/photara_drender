#include "asdiff_render/rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "context_internal.hpp"

namespace asdiff_render {
namespace {

constexpr std::uint32_t FORWARD_BLOCK_WIDTH = 8;
constexpr std::uint32_t FORWARD_BLOCK_HEIGHT = 8;
constexpr std::uint32_t BACKWARD_BLOCK_SIZE = 64;
constexpr std::uint32_t RASTER_TILE_SIZE = 16;

std::uint32_t divide_round_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

VkDescriptorBufferInfo descriptor(const Buffer& buffer) {
    return {buffer.handle, 0, buffer.size};
}

void validate_mesh(std::span<const float> positions, std::span<const std::uint32_t> indices) {
    if (positions.empty() || positions.size() % 4 != 0) {
        throw std::invalid_argument("clip_positions must have shape [vertex_count, 4]");
    }
    if (indices.empty() || indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    const auto vertex_count = positions.size() / 4;
    if (std::ranges::any_of(indices, [vertex_count](std::uint32_t index) { return index >= vertex_count; })) {
        throw std::out_of_range("triangle_indices contains an invalid vertex index");
    }
}

struct TileBins {
    std::uint32_t tile_count_x = 0;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> triangle_indices;
};

Viewport resolve_viewport(
    std::uint32_t output_width,
    std::uint32_t output_height,
    const std::optional<Viewport>& requested_viewport) {
    const Viewport viewport = requested_viewport.value_or(
        Viewport{0.0F, 0.0F, static_cast<float>(output_width), static_cast<float>(output_height)});
    if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
        !std::isfinite(viewport.height) || viewport.width <= 0.0F || viewport.height <= 0.0F) {
        throw std::invalid_argument("viewport must contain finite values and positive dimensions");
    }
    return viewport;
}

TileBins build_tile_bins(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    std::uint32_t width,
    std::uint32_t height,
    const Viewport& viewport) {
    TileBins result;
    result.tile_count_x = divide_round_up(width, RASTER_TILE_SIZE);
    const auto tile_count_y = divide_round_up(height, RASTER_TILE_SIZE);
    std::vector<std::vector<std::uint32_t>> bins(
        static_cast<std::size_t>(result.tile_count_x) * tile_count_y);
    const auto triangle_count = static_cast<std::uint32_t>(indices.size() / 3);
    for (std::uint32_t triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        bool valid = true;
        for (std::uint32_t corner = 0; corner < 3; ++corner) {
            const auto vertex_index = indices[triangle_index * 3 + corner];
            const float w = positions[vertex_index * 4 + 3];
            if (!(w > 1e-8F) || !std::isfinite(w)) {
                valid = false;
                break;
            }
            const float x = positions[vertex_index * 4 + 0] / w;
            const float y = positions[vertex_index * 4 + 1] / w;
            if (!std::isfinite(x) || !std::isfinite(y)) {
                valid = false;
                break;
            }
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
        if (!valid || max_x < -1.0F || min_x > 1.0F || max_y < -1.0F || min_y > 1.0F) {
            continue;
        }

        const auto pixel_min_x = std::clamp(
            static_cast<std::int32_t>(std::floor((min_x * 0.5F + 0.5F) * viewport.width + viewport.x)),
            0,
            static_cast<std::int32_t>(width - 1));
        const auto pixel_max_x = std::clamp(
            static_cast<std::int32_t>(std::ceil((max_x * 0.5F + 0.5F) * viewport.width + viewport.x)) - 1,
            0,
            static_cast<std::int32_t>(width - 1));
        const auto pixel_min_y = std::clamp(
            static_cast<std::int32_t>(std::floor((min_y * 0.5F + 0.5F) * viewport.height + viewport.y)),
            0,
            static_cast<std::int32_t>(height - 1));
        const auto pixel_max_y = std::clamp(
            static_cast<std::int32_t>(std::ceil((max_y * 0.5F + 0.5F) * viewport.height + viewport.y)) - 1,
            0,
            static_cast<std::int32_t>(height - 1));
        for (std::uint32_t tile_y = static_cast<std::uint32_t>(pixel_min_y) / RASTER_TILE_SIZE;
             tile_y <= static_cast<std::uint32_t>(pixel_max_y) / RASTER_TILE_SIZE;
             ++tile_y) {
            for (std::uint32_t tile_x = static_cast<std::uint32_t>(pixel_min_x) / RASTER_TILE_SIZE;
                 tile_x <= static_cast<std::uint32_t>(pixel_max_x) / RASTER_TILE_SIZE;
                 ++tile_x) {
                bins[static_cast<std::size_t>(tile_y) * result.tile_count_x + tile_x].push_back(triangle_index);
            }
        }
    }

    result.offsets.reserve(bins.size() + 1);
    result.offsets.push_back(0);
    for (const auto& bin : bins) {
        if (result.triangle_indices.size() + bin.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Tile triangle list exceeds backend limits");
        }
        result.triangle_indices.insert(result.triangle_indices.end(), bin.begin(), bin.end());
        result.offsets.push_back(static_cast<std::uint32_t>(result.triangle_indices.size()));
    }
    return result;
}

struct VertexAdjacency {
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> corners;
};

VertexAdjacency build_vertex_adjacency(
    std::span<const std::uint32_t> indices,
    std::uint32_t vertex_count) {
    VertexAdjacency result;
    result.offsets.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
    for (const auto vertex_index : indices) {
        ++result.offsets[vertex_index + 1];
    }
    for (std::uint32_t vertex_index = 1; vertex_index <= vertex_count; ++vertex_index) {
        result.offsets[vertex_index] += result.offsets[vertex_index - 1];
    }
    result.corners.resize(indices.size());
    auto cursors = result.offsets;
    for (std::uint32_t corner = 0; corner < indices.size(); ++corner) {
        result.corners[cursors[indices[corner]]++] = corner;
    }
    return result;
}

std::int32_t address_coordinate(std::int32_t coordinate, std::uint32_t size, AddressMode address_mode) {
    const auto signed_size = static_cast<std::int32_t>(size);
    if (address_mode == AddressMode::clamp) {
        return std::clamp(coordinate, 0, signed_size - 1);
    }
    if (address_mode == AddressMode::wrap) {
        const auto remainder = coordinate % signed_size;
        return remainder < 0 ? remainder + signed_size : remainder;
    }
    const auto period = signed_size * 2;
    auto mirrored = coordinate % period;
    if (mirrored < 0) {
        mirrored += period;
    }
    return mirrored < signed_size ? mirrored : period - 1 - mirrored;
}

struct TextureSampleAdjacency {
    std::vector<std::uint32_t> texel_offsets;
    std::vector<std::uint32_t> sample_entries;
    std::vector<float> sample_weights;
};

TextureSampleAdjacency build_texture_sample_adjacency(
    std::span<const float> uv,
    const RasterizeOutput& raster_output,
    const TextureDesc& texture_desc) {
    const auto pixel_count = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    const auto texel_count = static_cast<std::size_t>(texture_desc.width) * texture_desc.height;
    TextureSampleAdjacency result;
    result.texel_offsets.assign(texel_count + 1, 0);
    result.sample_weights.assign(pixel_count * 4, 0.0F);
    std::vector<std::uint32_t> sample_texels(pixel_count * 4, 0);
    std::size_t valid_entry_count = 0;
    for (std::size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
        if (raster_output.raster[pixel_index * 4 + 3] == 0.0F) {
            continue;
        }
        const float u = uv[pixel_index * 2 + 0];
        const float v = uv[pixel_index * 2 + 1];
        if (!std::isfinite(u) || !std::isfinite(v)) {
            throw std::invalid_argument("uv contains a non-finite value in a covered pixel");
        }
        const double texel_x = static_cast<double>(u) * texture_desc.width - 0.5;
        const double texel_y = static_cast<double>(v) * texture_desc.height - 0.5;
        if (std::abs(texel_x) > 1.0e9 || std::abs(texel_y) > 1.0e9) {
            throw std::out_of_range("uv is too large for portable texture addressing");
        }
        const auto base_x = static_cast<std::int32_t>(std::floor(texel_x));
        const auto base_y = static_cast<std::int32_t>(std::floor(texel_y));
        const float fraction_x = static_cast<float>(texel_x - std::floor(texel_x));
        const float fraction_y = static_cast<float>(texel_y - std::floor(texel_y));
        const std::int32_t x[2] = {
            address_coordinate(base_x, texture_desc.width, texture_desc.address_mode),
            address_coordinate(base_x + 1, texture_desc.width, texture_desc.address_mode),
        };
        const std::int32_t y[2] = {
            address_coordinate(base_y, texture_desc.height, texture_desc.address_mode),
            address_coordinate(base_y + 1, texture_desc.height, texture_desc.address_mode),
        };
        const float weights[4] = {
            (1.0F - fraction_x) * (1.0F - fraction_y),
            fraction_x * (1.0F - fraction_y),
            (1.0F - fraction_x) * fraction_y,
            fraction_x * fraction_y,
        };
        const std::uint32_t texels[4] = {
            static_cast<std::uint32_t>(y[0]) * texture_desc.width + static_cast<std::uint32_t>(x[0]),
            static_cast<std::uint32_t>(y[0]) * texture_desc.width + static_cast<std::uint32_t>(x[1]),
            static_cast<std::uint32_t>(y[1]) * texture_desc.width + static_cast<std::uint32_t>(x[0]),
            static_cast<std::uint32_t>(y[1]) * texture_desc.width + static_cast<std::uint32_t>(x[1]),
        };
        for (std::uint32_t corner = 0; corner < 4; ++corner) {
            const auto sample_entry = pixel_index * 4 + corner;
            result.sample_weights[sample_entry] = weights[corner];
            sample_texels[sample_entry] = texels[corner];
            ++result.texel_offsets[texels[corner] + 1];
            ++valid_entry_count;
        }
    }
    for (std::size_t texel_index = 1; texel_index <= texel_count; ++texel_index) {
        result.texel_offsets[texel_index] += result.texel_offsets[texel_index - 1];
    }
    result.sample_entries.resize(valid_entry_count);
    auto cursors = result.texel_offsets;
    for (std::size_t sample_entry = 0; sample_entry < sample_texels.size(); ++sample_entry) {
        const auto pixel_index = sample_entry / 4;
        if (raster_output.raster[pixel_index * 4 + 3] == 0.0F) {
            continue;
        }
        const auto texel_index = sample_texels[sample_entry];
        result.sample_entries[cursors[texel_index]++] = static_cast<std::uint32_t>(sample_entry);
    }
    return result;
}

void validate_texture_inputs(
    std::span<const float> texture,
    const TextureDesc& texture_desc,
    std::span<const float> uv,
    const RasterizeOutput& raster_output) {
    if (texture_desc.width == 0 || texture_desc.height == 0 || texture_desc.channel_count == 0) {
        throw std::invalid_argument("Texture width, height, and channel_count must be positive");
    }
    if (texture_desc.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() / 2) ||
        texture_desc.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() / 2)) {
        throw std::out_of_range("Texture dimensions exceed portable addressing limits");
    }
    const auto texture_value_count = static_cast<std::size_t>(texture_desc.width) * texture_desc.height *
                                     texture_desc.channel_count;
    if (texture.size() != texture_value_count) {
        throw std::invalid_argument("texture must have shape [texture_height, texture_width, channel_count]");
    }
    const auto pixel_count = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    if (raster_output.width == 0 || raster_output.height == 0 || raster_output.raster.size() != pixel_count * 4) {
        throw std::invalid_argument("raster_output has inconsistent dimensions");
    }
    if (uv.size() != pixel_count * 2) {
        throw std::invalid_argument("uv must have shape [image_height, image_width, 2]");
    }
    if (pixel_count > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::size_t>(texture_desc.width) * texture_desc.height >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Texture operation exceeds backend element limits");
    }
}

} // namespace

class Rasterizer::Impl {
public:
    explicit Impl(Context::Impl& context)
        : context_(context),
          forward_pipeline_(context.create_pipeline("rasterize_forward.hlsl.spv", 6, sizeof(ForwardPushConstants))),
          backward_pipeline_(context.create_pipeline("rasterize_backward.hlsl.spv", 6, sizeof(BackwardPushConstants))),
          reduce_pipeline_(context.create_pipeline("reduce_vertex_gradients.hlsl.spv", 4, sizeof(ReducePushConstants))),
          interpolate_forward_pipeline_(
              context.create_pipeline("interpolate_forward.hlsl.spv", 4, sizeof(InterpolatePushConstants))),
          interpolate_grad_attributes_pipeline_(
              context.create_pipeline("interpolate_grad_attributes.hlsl.spv", 4, sizeof(InterpolatePushConstants))),
          interpolate_grad_raster_pipeline_(
              context.create_pipeline("interpolate_grad_raster.hlsl.spv", 5, sizeof(InterpolatePushConstants))),
          texture_forward_pipeline_(context.create_pipeline("texture_forward.hlsl.spv", 4, sizeof(TexturePushConstants))),
          texture_grad_texture_pipeline_(
              context.create_pipeline("texture_grad_texture.hlsl.spv", 5, sizeof(TextureGradTexturePushConstants))),
          texture_grad_uv_pipeline_(context.create_pipeline("texture_grad_uv.hlsl.spv", 5, sizeof(TexturePushConstants))) {}

    struct ForwardPushConstants {
        std::uint32_t vertex_count;
        std::uint32_t triangle_count;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t cull_mode;
        std::uint32_t output_derivatives;
        std::uint32_t tile_count_x;
        std::uint32_t tile_size;
        float viewport[4];
    };

    struct BackwardPushConstants {
        std::uint32_t vertex_count;
        std::uint32_t triangle_count;
        std::uint32_t width;
        std::uint32_t height;
        float viewport[4];
    };

    struct ReducePushConstants {
        std::uint32_t vertex_count;
    };

    struct InterpolatePushConstants {
        std::uint32_t vertex_count;
        std::uint32_t triangle_count;
        std::uint32_t pixel_count;
        std::uint32_t attribute_count;
    };

    struct TexturePushConstants {
        std::uint32_t pixel_count;
        std::uint32_t texture_width;
        std::uint32_t texture_height;
        std::uint32_t channel_count;
        std::uint32_t address_mode;
    };

    struct TextureGradTexturePushConstants {
        std::uint32_t texel_count;
        std::uint32_t pixel_count;
        std::uint32_t channel_count;
    };

    Context::Impl& context_;
    ComputePipeline forward_pipeline_;
    ComputePipeline backward_pipeline_;
    ComputePipeline reduce_pipeline_;
    ComputePipeline interpolate_forward_pipeline_;
    ComputePipeline interpolate_grad_attributes_pipeline_;
    ComputePipeline interpolate_grad_raster_pipeline_;
    ComputePipeline texture_forward_pipeline_;
    ComputePipeline texture_grad_texture_pipeline_;
    ComputePipeline texture_grad_uv_pipeline_;
};

Rasterizer::Rasterizer(Context& context) : impl_(std::make_unique<Impl>(*context.impl_)) {}
Rasterizer::~Rasterizer() = default;
Rasterizer::Rasterizer(Rasterizer&&) noexcept = default;
Rasterizer& Rasterizer::operator=(Rasterizer&&) noexcept = default;

RasterizeOutput Rasterizer::forward(
    std::span<const float> clip_positions,
    std::span<const std::uint32_t> triangle_indices,
    const RasterizeOptions& options) {
    validate_mesh(clip_positions, triangle_indices);
    if (options.width == 0 || options.height == 0) {
        throw std::invalid_argument("Rasterization width and height must be positive");
    }

    const auto pixel_count = static_cast<std::size_t>(options.width) * options.height;
    if (pixel_count > (std::numeric_limits<std::size_t>::max() / (4 * sizeof(float)))) {
        throw std::overflow_error("Rasterization output size overflow");
    }
    const auto position_bytes = clip_positions.size_bytes();
    const auto index_bytes = triangle_indices.size_bytes();
    const auto output_bytes = pixel_count * 4 * sizeof(float);
    const auto viewport = resolve_viewport(options.width, options.height, options.viewport);
    const auto tile_bins = build_tile_bins(clip_positions, triangle_indices, options.width, options.height, viewport);
    const auto tile_offset_bytes = tile_bins.offsets.size() * sizeof(std::uint32_t);
    const auto tile_triangle_bytes = tile_bins.triangle_indices.size() * sizeof(std::uint32_t);
    auto position_buffer = impl_->context_.create_buffer(position_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto index_buffer = impl_->context_.create_buffer(index_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto tile_offset_buffer = impl_->context_.create_buffer(tile_offset_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto tile_triangle_buffer = impl_->context_.create_buffer(tile_triangle_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto raster_buffer = impl_->context_.create_buffer(output_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto derivative_buffer = impl_->context_.create_buffer(output_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    position_buffer.upload(clip_positions.data(), position_bytes);
    index_buffer.upload(triangle_indices.data(), index_bytes);
    tile_offset_buffer.upload(tile_bins.offsets.data(), tile_offset_bytes);
    tile_triangle_buffer.upload(tile_bins.triangle_indices.data(), tile_triangle_bytes);

    const Impl::ForwardPushConstants push_constants{
        static_cast<std::uint32_t>(clip_positions.size() / 4),
        static_cast<std::uint32_t>(triangle_indices.size() / 3),
        options.width,
        options.height,
        static_cast<std::uint32_t>(options.cull_mode),
        options.output_barycentric_derivatives ? 1U : 0U,
        tile_bins.tile_count_x,
        RASTER_TILE_SIZE,
        {viewport.x, viewport.y, viewport.width, viewport.height},
    };
    const std::vector<VkDescriptorBufferInfo> descriptors{
        descriptor(position_buffer),
        descriptor(index_buffer),
        descriptor(tile_offset_buffer),
        descriptor(tile_triangle_buffer),
        descriptor(raster_buffer),
        descriptor(derivative_buffer),
    };
    impl_->context_.dispatch(
        impl_->forward_pipeline_,
        descriptors,
        &push_constants,
        sizeof(push_constants),
        divide_round_up(options.width, FORWARD_BLOCK_WIDTH),
        divide_round_up(options.height, FORWARD_BLOCK_HEIGHT));

    RasterizeOutput result;
    result.width = options.width;
    result.height = options.height;
    result.raster.resize(pixel_count * 4);
    raster_buffer.download(result.raster.data(), output_bytes);
    if (options.output_barycentric_derivatives) {
        result.barycentric_derivatives.resize(pixel_count * 4);
        derivative_buffer.download(result.barycentric_derivatives.data(), output_bytes);
    }
    result.viewport = viewport;
    return result;
}

std::vector<float> Rasterizer::backward(
    std::span<const float> clip_positions,
    std::span<const std::uint32_t> triangle_indices,
    const RasterizeOutput& forward_output,
    std::span<const float> grad_raster,
    std::span<const float> grad_barycentric_derivatives) {
    validate_mesh(clip_positions, triangle_indices);
    const auto pixel_count = static_cast<std::size_t>(forward_output.width) * forward_output.height;
    const auto output_value_count = pixel_count * 4;
    if (forward_output.width == 0 || forward_output.height == 0 || forward_output.raster.size() != output_value_count) {
        throw std::invalid_argument("forward_output has inconsistent dimensions");
    }
    if (grad_raster.size() != output_value_count) {
        throw std::invalid_argument("grad_raster must have shape [height, width, 4]");
    }
    if (!grad_barycentric_derivatives.empty()) {
        throw std::invalid_argument(
            "Gradient propagation through barycentric derivatives is not implemented in the portable baseline backend");
    }

    const auto vertex_count = static_cast<std::uint32_t>(clip_positions.size() / 4);
    const auto triangle_count = static_cast<std::uint32_t>(triangle_indices.size() / 3);
    const auto position_bytes = clip_positions.size_bytes();
    const auto index_bytes = triangle_indices.size_bytes();
    const auto raster_bytes = forward_output.raster.size() * sizeof(float);
    const auto corner_gradient_bytes = triangle_indices.size() * 4 * sizeof(float);
    const auto vertex_gradient_bytes = clip_positions.size_bytes();
    const auto viewport = resolve_viewport(
        forward_output.width,
        forward_output.height,
        forward_output.viewport.width > 0.0F ? std::optional<Viewport>(forward_output.viewport) : std::nullopt);
    auto position_buffer = impl_->context_.create_buffer(position_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto index_buffer = impl_->context_.create_buffer(index_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto grad_raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto grad_derivative_buffer = impl_->context_.create_buffer(4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto corner_gradient_buffer = impl_->context_.create_buffer(corner_gradient_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto vertex_gradient_buffer = impl_->context_.create_buffer(vertex_gradient_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    const auto adjacency = build_vertex_adjacency(triangle_indices, vertex_count);
    const auto adjacency_offset_bytes = adjacency.offsets.size() * sizeof(std::uint32_t);
    const auto adjacency_corner_bytes = adjacency.corners.size() * sizeof(std::uint32_t);
    auto adjacency_offset_buffer =
        impl_->context_.create_buffer(adjacency_offset_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto adjacency_corner_buffer =
        impl_->context_.create_buffer(adjacency_corner_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    position_buffer.upload(clip_positions.data(), position_bytes);
    index_buffer.upload(triangle_indices.data(), index_bytes);
    raster_buffer.upload(forward_output.raster.data(), raster_bytes);
    grad_raster_buffer.upload(grad_raster.data(), raster_bytes);
    adjacency_offset_buffer.upload(adjacency.offsets.data(), adjacency_offset_bytes);
    adjacency_corner_buffer.upload(adjacency.corners.data(), adjacency_corner_bytes);

    const Impl::BackwardPushConstants backward_push_constants{
        vertex_count,
        triangle_count,
        forward_output.width,
        forward_output.height,
        {viewport.x, viewport.y, viewport.width, viewport.height},
    };
    const std::vector<VkDescriptorBufferInfo> backward_descriptors{
        descriptor(position_buffer),
        descriptor(index_buffer),
        descriptor(raster_buffer),
        descriptor(grad_raster_buffer),
        descriptor(grad_derivative_buffer),
        descriptor(corner_gradient_buffer),
    };
    impl_->context_.dispatch(
        impl_->backward_pipeline_,
        backward_descriptors,
        &backward_push_constants,
        sizeof(backward_push_constants),
        divide_round_up(triangle_count, BACKWARD_BLOCK_SIZE));

    const Impl::ReducePushConstants reduce_push_constants{
        vertex_count,
    };
    const std::vector<VkDescriptorBufferInfo> reduce_descriptors{
        descriptor(adjacency_offset_buffer),
        descriptor(adjacency_corner_buffer),
        descriptor(corner_gradient_buffer),
        descriptor(vertex_gradient_buffer),
    };
    impl_->context_.dispatch(
        impl_->reduce_pipeline_,
        reduce_descriptors,
        &reduce_push_constants,
        sizeof(reduce_push_constants),
        divide_round_up(vertex_count, BACKWARD_BLOCK_SIZE));

    std::vector<float> result(clip_positions.size());
    vertex_gradient_buffer.download(result.data(), vertex_gradient_bytes);
    return result;
}

InterpolateOutput Rasterizer::interpolate_forward(
    std::span<const float> vertex_attributes,
    std::uint32_t attribute_count,
    std::span<const std::uint32_t> triangle_indices,
    const RasterizeOutput& raster_output) {
    if (attribute_count == 0 || vertex_attributes.empty() || vertex_attributes.size() % attribute_count != 0) {
        throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
    }
    if (triangle_indices.empty() || triangle_indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    const auto vertex_count = static_cast<std::uint32_t>(vertex_attributes.size() / attribute_count);
    if (std::ranges::any_of(triangle_indices, [vertex_count](std::uint32_t index) { return index >= vertex_count; })) {
        throw std::out_of_range("triangle_indices contains an invalid attribute vertex index");
    }
    const auto pixel_count_size = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    if (raster_output.width == 0 || raster_output.height == 0 || raster_output.raster.size() != pixel_count_size * 4) {
        throw std::invalid_argument("raster_output has inconsistent dimensions");
    }
    if (pixel_count_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Interpolation pixel count exceeds backend limits");
    }

    const auto output_value_count = pixel_count_size * attribute_count;
    const auto attribute_bytes = vertex_attributes.size_bytes();
    const auto index_bytes = triangle_indices.size_bytes();
    const auto raster_bytes = raster_output.raster.size() * sizeof(float);
    const auto output_bytes = output_value_count * sizeof(float);
    auto attribute_buffer = impl_->context_.create_buffer(attribute_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto index_buffer = impl_->context_.create_buffer(index_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto output_buffer = impl_->context_.create_buffer(output_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    attribute_buffer.upload(vertex_attributes.data(), attribute_bytes);
    index_buffer.upload(triangle_indices.data(), index_bytes);
    raster_buffer.upload(raster_output.raster.data(), raster_bytes);

    const Impl::InterpolatePushConstants push_constants{
        vertex_count,
        static_cast<std::uint32_t>(triangle_indices.size() / 3),
        static_cast<std::uint32_t>(pixel_count_size),
        attribute_count,
    };
    const std::vector<VkDescriptorBufferInfo> descriptors{
        descriptor(attribute_buffer),
        descriptor(index_buffer),
        descriptor(raster_buffer),
        descriptor(output_buffer),
    };
    impl_->context_.dispatch(
        impl_->interpolate_forward_pipeline_,
        descriptors,
        &push_constants,
        sizeof(push_constants),
        divide_round_up(static_cast<std::uint32_t>(output_value_count), BACKWARD_BLOCK_SIZE));

    InterpolateOutput result;
    result.width = raster_output.width;
    result.height = raster_output.height;
    result.attribute_count = attribute_count;
    result.values.resize(output_value_count);
    output_buffer.download(result.values.data(), output_bytes);
    return result;
}

InterpolateGradients Rasterizer::interpolate_backward(
    std::span<const float> vertex_attributes,
    std::uint32_t attribute_count,
    std::span<const std::uint32_t> triangle_indices,
    const RasterizeOutput& raster_output,
    std::span<const float> grad_interpolated) {
    if (attribute_count == 0 || vertex_attributes.empty() || vertex_attributes.size() % attribute_count != 0) {
        throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
    }
    const auto vertex_count = static_cast<std::uint32_t>(vertex_attributes.size() / attribute_count);
    if (triangle_indices.empty() || triangle_indices.size() % 3 != 0 ||
        std::ranges::any_of(triangle_indices, [vertex_count](std::uint32_t index) { return index >= vertex_count; })) {
        throw std::invalid_argument("triangle_indices is incompatible with vertex_attributes");
    }
    const auto pixel_count_size = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    if (raster_output.width == 0 || raster_output.height == 0 ||
        raster_output.raster.size() != pixel_count_size * 4) {
        throw std::invalid_argument("raster_output has inconsistent dimensions");
    }
    if (grad_interpolated.size() != pixel_count_size * attribute_count) {
        throw std::invalid_argument("grad_interpolated must have shape [height, width, attribute_count]");
    }
    if (pixel_count_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Interpolation pixel count exceeds backend limits");
    }

    const auto attribute_bytes = vertex_attributes.size_bytes();
    const auto index_bytes = triangle_indices.size_bytes();
    const auto raster_bytes = raster_output.raster.size() * sizeof(float);
    const auto output_gradient_bytes = grad_interpolated.size_bytes();
    auto attribute_buffer = impl_->context_.create_buffer(attribute_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto index_buffer = impl_->context_.create_buffer(index_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto output_gradient_buffer = impl_->context_.create_buffer(output_gradient_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto attribute_gradient_buffer = impl_->context_.create_buffer(attribute_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto raster_gradient_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    attribute_buffer.upload(vertex_attributes.data(), attribute_bytes);
    index_buffer.upload(triangle_indices.data(), index_bytes);
    raster_buffer.upload(raster_output.raster.data(), raster_bytes);
    output_gradient_buffer.upload(grad_interpolated.data(), output_gradient_bytes);

    const Impl::InterpolatePushConstants push_constants{
        vertex_count,
        static_cast<std::uint32_t>(triangle_indices.size() / 3),
        static_cast<std::uint32_t>(pixel_count_size),
        attribute_count,
    };
    const std::vector<VkDescriptorBufferInfo> attribute_descriptors{
        descriptor(index_buffer),
        descriptor(raster_buffer),
        descriptor(output_gradient_buffer),
        descriptor(attribute_gradient_buffer),
    };
    impl_->context_.dispatch(
        impl_->interpolate_grad_attributes_pipeline_,
        attribute_descriptors,
        &push_constants,
        sizeof(push_constants),
        divide_round_up(static_cast<std::uint32_t>(vertex_attributes.size()), BACKWARD_BLOCK_SIZE));

    const std::vector<VkDescriptorBufferInfo> raster_descriptors{
        descriptor(attribute_buffer),
        descriptor(index_buffer),
        descriptor(raster_buffer),
        descriptor(output_gradient_buffer),
        descriptor(raster_gradient_buffer),
    };
    impl_->context_.dispatch(
        impl_->interpolate_grad_raster_pipeline_,
        raster_descriptors,
        &push_constants,
        sizeof(push_constants),
        divide_round_up(static_cast<std::uint32_t>(pixel_count_size), BACKWARD_BLOCK_SIZE));

    InterpolateGradients result;
    result.attributes.resize(vertex_attributes.size());
    result.raster.resize(raster_output.raster.size());
    attribute_gradient_buffer.download(result.attributes.data(), attribute_bytes);
    raster_gradient_buffer.download(result.raster.data(), raster_bytes);
    return result;
}

TextureOutput Rasterizer::texture_forward(
    std::span<const float> texture,
    const TextureDesc& texture_desc,
    std::span<const float> uv,
    const RasterizeOutput& raster_output) {
    validate_texture_inputs(texture, texture_desc, uv, raster_output);
    const auto pixel_count = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    const auto output_value_count = pixel_count * texture_desc.channel_count;
    if (output_value_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Sampled texture output exceeds backend element limits");
    }
    const auto texture_bytes = texture.size_bytes();
    const auto uv_bytes = uv.size_bytes();
    const auto raster_bytes = raster_output.raster.size() * sizeof(float);
    const auto output_bytes = output_value_count * sizeof(float);
    auto texture_buffer = impl_->context_.create_buffer(texture_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto uv_buffer = impl_->context_.create_buffer(uv_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto output_buffer = impl_->context_.create_buffer(output_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    texture_buffer.upload(texture.data(), texture_bytes);
    uv_buffer.upload(uv.data(), uv_bytes);
    raster_buffer.upload(raster_output.raster.data(), raster_bytes);

    const Impl::TexturePushConstants push_constants{
        static_cast<std::uint32_t>(pixel_count),
        texture_desc.width,
        texture_desc.height,
        texture_desc.channel_count,
        static_cast<std::uint32_t>(texture_desc.address_mode),
    };
    const std::vector<VkDescriptorBufferInfo> descriptors{
        descriptor(texture_buffer),
        descriptor(uv_buffer),
        descriptor(raster_buffer),
        descriptor(output_buffer),
    };
    impl_->context_.dispatch(
        impl_->texture_forward_pipeline_,
        descriptors,
        &push_constants,
        sizeof(push_constants),
        divide_round_up(static_cast<std::uint32_t>(output_value_count), BACKWARD_BLOCK_SIZE));

    TextureOutput result;
    result.width = raster_output.width;
    result.height = raster_output.height;
    result.channel_count = texture_desc.channel_count;
    result.values.resize(output_value_count);
    output_buffer.download(result.values.data(), output_bytes);
    return result;
}

TextureGradients Rasterizer::texture_backward(
    std::span<const float> texture,
    const TextureDesc& texture_desc,
    std::span<const float> uv,
    const RasterizeOutput& raster_output,
    std::span<const float> grad_sampled,
    bool compute_uv_gradient) {
    validate_texture_inputs(texture, texture_desc, uv, raster_output);
    const auto pixel_count = static_cast<std::size_t>(raster_output.width) * raster_output.height;
    const auto texel_count = static_cast<std::size_t>(texture_desc.width) * texture_desc.height;
    const auto sampled_value_count = pixel_count * texture_desc.channel_count;
    if (grad_sampled.size() != sampled_value_count) {
        throw std::invalid_argument("grad_sampled must have shape [image_height, image_width, channel_count]");
    }
    if (sampled_value_count > std::numeric_limits<std::uint32_t>::max() ||
        texture.size() > std::numeric_limits<std::uint32_t>::max() ||
        pixel_count > std::numeric_limits<std::uint32_t>::max() / 4) {
        throw std::overflow_error("Texture gradient operation exceeds backend element limits");
    }

    const auto adjacency = build_texture_sample_adjacency(uv, raster_output, texture_desc);
    const auto texture_bytes = texture.size_bytes();
    const auto uv_bytes = uv.size_bytes();
    const auto raster_bytes = raster_output.raster.size() * sizeof(float);
    const auto grad_sampled_bytes = grad_sampled.size_bytes();
    const auto texel_offset_bytes = adjacency.texel_offsets.size() * sizeof(std::uint32_t);
    const auto sample_entry_bytes = adjacency.sample_entries.size() * sizeof(std::uint32_t);
    const auto sample_weight_bytes = adjacency.sample_weights.size() * sizeof(float);
    auto grad_sampled_buffer = impl_->context_.create_buffer(grad_sampled_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto texel_offset_buffer = impl_->context_.create_buffer(texel_offset_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto sample_entry_buffer = impl_->context_.create_buffer(sample_entry_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto sample_weight_buffer = impl_->context_.create_buffer(sample_weight_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto grad_texture_buffer = impl_->context_.create_buffer(texture_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    grad_sampled_buffer.upload(grad_sampled.data(), grad_sampled_bytes);
    texel_offset_buffer.upload(adjacency.texel_offsets.data(), texel_offset_bytes);
    sample_entry_buffer.upload(adjacency.sample_entries.data(), sample_entry_bytes);
    sample_weight_buffer.upload(adjacency.sample_weights.data(), sample_weight_bytes);

    const Impl::TextureGradTexturePushConstants grad_texture_push_constants{
        static_cast<std::uint32_t>(texel_count),
        static_cast<std::uint32_t>(pixel_count),
        texture_desc.channel_count,
    };
    const std::vector<VkDescriptorBufferInfo> grad_texture_descriptors{
        descriptor(grad_sampled_buffer),
        descriptor(texel_offset_buffer),
        descriptor(sample_entry_buffer),
        descriptor(sample_weight_buffer),
        descriptor(grad_texture_buffer),
    };
    impl_->context_.dispatch(
        impl_->texture_grad_texture_pipeline_,
        grad_texture_descriptors,
        &grad_texture_push_constants,
        sizeof(grad_texture_push_constants),
        divide_round_up(static_cast<std::uint32_t>(texture.size()), BACKWARD_BLOCK_SIZE));

    TextureGradients result;
    result.texture.resize(texture.size());
    grad_texture_buffer.download(result.texture.data(), texture_bytes);
    if (compute_uv_gradient) {
        auto texture_buffer = impl_->context_.create_buffer(texture_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto uv_buffer = impl_->context_.create_buffer(uv_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto raster_buffer = impl_->context_.create_buffer(raster_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto grad_uv_buffer = impl_->context_.create_buffer(uv_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        texture_buffer.upload(texture.data(), texture_bytes);
        uv_buffer.upload(uv.data(), uv_bytes);
        raster_buffer.upload(raster_output.raster.data(), raster_bytes);
        const Impl::TexturePushConstants grad_uv_push_constants{
            static_cast<std::uint32_t>(pixel_count),
            texture_desc.width,
            texture_desc.height,
            texture_desc.channel_count,
            static_cast<std::uint32_t>(texture_desc.address_mode),
        };
        const std::vector<VkDescriptorBufferInfo> grad_uv_descriptors{
            descriptor(texture_buffer),
            descriptor(uv_buffer),
            descriptor(raster_buffer),
            descriptor(grad_sampled_buffer),
            descriptor(grad_uv_buffer),
        };
        impl_->context_.dispatch(
            impl_->texture_grad_uv_pipeline_,
            grad_uv_descriptors,
            &grad_uv_push_constants,
            sizeof(grad_uv_push_constants),
            divide_round_up(static_cast<std::uint32_t>(pixel_count), BACKWARD_BLOCK_SIZE));
        result.uv.resize(uv.size());
        grad_uv_buffer.download(result.uv.data(), uv_bytes);
    }
    return result;
}

} // namespace asdiff_render
