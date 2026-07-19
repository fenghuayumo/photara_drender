#pragma once

#include <cstdint>
#include <span>

#include "asdiff_render/types.hpp"

namespace asdiff_render {

[[nodiscard]] bool has_uv_atlas_backend() noexcept;

[[nodiscard]] UvAtlasOutput unwrap_uv(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const UvAtlasOptions& options = {});

} // namespace asdiff_render
