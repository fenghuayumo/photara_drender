#pragma once

#include <cstdint>
#include <span>

#include "aether_drender/types.hpp"

namespace aether_drender {

[[nodiscard]] bool has_uv_atlas_backend() noexcept;

[[nodiscard]] UvAtlasOutput unwrap_uv(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const UvAtlasOptions& options = {});

} // namespace aether_drender
