#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "aether_drender/context.hpp"
#include "aether_drender/types.hpp"

namespace aether_drender {

class Rasterizer {
public:
    explicit Rasterizer(Context& context);
    ~Rasterizer();

    Rasterizer(Rasterizer&&) noexcept;
    Rasterizer& operator=(Rasterizer&&) noexcept;
    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    [[nodiscard]] RasterizeOutput forward(
        std::span<const float> clip_positions,
        std::span<const std::uint32_t> triangle_indices,
        const RasterizeOptions& options);

    [[nodiscard]] std::vector<float> backward(
        std::span<const float> clip_positions,
        std::span<const std::uint32_t> triangle_indices,
        const RasterizeOutput& forward_output,
        std::span<const float> grad_raster,
        std::span<const float> grad_barycentric_derivatives = {});

    [[nodiscard]] InterpolateOutput interpolate_forward(
        std::span<const float> vertex_attributes,
        std::uint32_t attribute_count,
        std::span<const std::uint32_t> triangle_indices,
        const RasterizeOutput& raster_output);

    [[nodiscard]] InterpolateGradients interpolate_backward(
        std::span<const float> vertex_attributes,
        std::uint32_t attribute_count,
        std::span<const std::uint32_t> triangle_indices,
        const RasterizeOutput& raster_output,
        std::span<const float> grad_interpolated);

    [[nodiscard]] TextureOutput texture_forward(
        std::span<const float> texture,
        const TextureDesc& texture_desc,
        std::span<const float> uv,
        const RasterizeOutput& raster_output);

    [[nodiscard]] TextureGradients texture_backward(
        std::span<const float> texture,
        const TextureDesc& texture_desc,
        std::span<const float> uv,
        const RasterizeOutput& raster_output,
        std::span<const float> grad_sampled,
        bool compute_uv_gradient = true);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aether_drender
