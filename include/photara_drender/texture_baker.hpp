#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "photara_drender/context.hpp"
#include "photara_drender/types.hpp"

namespace photara_drender {

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

} // namespace photara_drender
