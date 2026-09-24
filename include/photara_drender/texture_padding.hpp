#pragma once

#include <cstdint>
#include <span>

#include "photara_drender/types.hpp"

namespace photara_drender {

// Atlas gutter/guard fill. Colour is extrapolated from texels that received a
// real projection sample into
//   * the empty gutter around every chart (within `margin` texels), and
//   * chart texels that were rasterized but never received a sample (masked
//     out, rejected by the facing test, or occluded in every view).
// Without this step filtered (bilinear/mipmapped) sampling and any seaming
// viewer show the untouched texels as black cracks, and unseen surfaces stay
// pure black even when a neighbouring chart has a plausible colour.
struct TexturePaddingOptions {
    // Guard band in texels copied outwards from valid texels. 0 disables the
    // gutter guard while keeping the in-chart fill.
    std::uint32_t margin = 4;
    // Fill chart texels that rasterized but hold no projection sample.
    bool fill_unobserved = true;
    // Optional cap for the in-chart fill distance in texels (0 = unlimited).
    // A cap keeps long-range colour bleeding between distant charts bounded.
    std::uint32_t maximum_fill_distance = 0;
    // A texel counts as a colour source when valid_mask >= valid_threshold.
    float valid_threshold = 0.5F;
};

struct TexturePaddingStats {
    std::uint64_t valid_texels = 0;
    // Empty gutter texels filled within `margin` of a valid texel.
    std::uint64_t gutter_texels = 0;
    // Rasterized chart texels filled because they had no sample at all.
    std::uint64_t unobserved_texels = 0;
    // Chart texels that still have no colour (fill disabled or out of range).
    std::uint64_t remaining_unobserved = 0;
    // Texels whose colour was extrapolated (gutter_texels + unobserved_texels).
    std::uint64_t filled_texels = 0;
    double seconds = 0.0;
};

// Extends `output.color` in place and records the guard texels in
// output.filled_mask. output.valid_mask keeps its projection-valid meaning and
// is never widened, so callers can still tell measured texels from guard texels.
[[nodiscard]] TexturePaddingStats pad_texture_atlas(
    TextureBakeOutput& output, const TexturePaddingOptions& options = {});

// Scene-relative exponential scale for ProjectionBlendMode::softmax, matching
// the reference implementation's heuristic: 20 * (minimum squared distance
// between a mesh vertex sample and any camera). Vertices are subsampled to
// keep the query O(1) in mesh size. Returns 0 when the input is degenerate.
[[nodiscard]] float suggest_softmax_scale(
    std::span<const float> positions,
    std::span<const float> camera_positions);

} // namespace photara_drender
