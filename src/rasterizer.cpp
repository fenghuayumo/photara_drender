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

TileBins build_tile_bins(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    std::uint32_t width,
    std::uint32_t height) {
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
            static_cast<std::int32_t>(std::floor((min_x * 0.5F + 0.5F) * width)),
            0,
            static_cast<std::int32_t>(width - 1));
        const auto pixel_max_x = std::clamp(
            static_cast<std::int32_t>(std::ceil((max_x * 0.5F + 0.5F) * width)) - 1,
            0,
            static_cast<std::int32_t>(width - 1));
        const auto pixel_min_y = std::clamp(
            static_cast<std::int32_t>(std::floor((min_y * 0.5F + 0.5F) * height)),
            0,
            static_cast<std::int32_t>(height - 1));
        const auto pixel_max_y = std::clamp(
            static_cast<std::int32_t>(std::ceil((max_y * 0.5F + 0.5F) * height)) - 1,
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
              context.create_pipeline("interpolate_grad_raster.hlsl.spv", 5, sizeof(InterpolatePushConstants))) {}

    struct ForwardPushConstants {
        std::uint32_t vertex_count;
        std::uint32_t triangle_count;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t cull_mode;
        std::uint32_t output_derivatives;
        std::uint32_t tile_count_x;
        std::uint32_t tile_size;
    };

    struct BackwardPushConstants {
        std::uint32_t vertex_count;
        std::uint32_t triangle_count;
        std::uint32_t width;
        std::uint32_t height;
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

    Context::Impl& context_;
    ComputePipeline forward_pipeline_;
    ComputePipeline backward_pipeline_;
    ComputePipeline reduce_pipeline_;
    ComputePipeline interpolate_forward_pipeline_;
    ComputePipeline interpolate_grad_attributes_pipeline_;
    ComputePipeline interpolate_grad_raster_pipeline_;
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
    const auto tile_bins = build_tile_bins(clip_positions, triangle_indices, options.width, options.height);
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

} // namespace asdiff_render
