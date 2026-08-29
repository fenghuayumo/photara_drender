#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "aether_drender/context.hpp"
#include "aether_drender/types.hpp"

namespace aether_drender {

class TextureBaker {
public:
    explicit TextureBaker(Context& context);
    ~TextureBaker();

    TextureBaker(TextureBaker&&) noexcept;
    TextureBaker& operator=(TextureBaker&&) noexcept;
    TextureBaker(const TextureBaker&) = delete;
    TextureBaker& operator=(const TextureBaker&) = delete;

    [[nodiscard]] TextureBakeOutput bake(
        std::span<const float> positions,
        std::span<const float> normals,
        std::span<const float> uv,
        std::span<const std::uint32_t> triangle_indices,
        std::span<const ProjectionView> views,
        const TextureBakeOptions& options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aether_drender
