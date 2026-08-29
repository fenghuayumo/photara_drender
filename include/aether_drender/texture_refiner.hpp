#pragma once

#include <memory>
#include <span>

#include "aether_drender/context.hpp"
#include "aether_drender/types.hpp"

namespace aether_drender {

class TextureRefiner {
public:
    explicit TextureRefiner(Context& context);
    ~TextureRefiner();

    TextureRefiner(TextureRefiner&&) noexcept;
    TextureRefiner& operator=(TextureRefiner&&) noexcept;
    TextureRefiner(const TextureRefiner&) = delete;
    TextureRefiner& operator=(const TextureRefiner&) = delete;

    [[nodiscard]] TextureRefineOutput refine(
        std::span<const float> initial_texture,
        std::span<const float> positions,
        std::span<const float> uv,
        std::span<const std::uint32_t> triangle_indices,
        std::span<const ProjectionView> views,
        std::span<const float> seam_uv_pairs = {},
        const TextureRefineOptions& options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aether_drender
