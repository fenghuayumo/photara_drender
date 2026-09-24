#include "photara_drender/texture_padding.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace photara_drender {
namespace {

// Atlas side lengths are limited to keep neighbour offsets representable as
// int16. Every practical atlas is far below this.
constexpr std::int32_t k_maximum_atlas_side = 16384;
// Offset used for "no source texel" cells. Large enough to dominate any real
// distance and small enough that the squared value stays exact in int64.
constexpr std::int32_t k_far_offset = 12000;

struct NearestField {
    std::int32_t width = 0;
    std::int32_t height = 0;
    // Offset from each texel to its nearest source texel.
    std::vector<std::int16_t> offset_x;
    std::vector<std::int16_t> offset_y;
};

[[nodiscard]] bool valid_source(std::span<const float> mask, std::size_t index, float threshold) {
    return mask[index] >= threshold;
}

// Danielsson 8SSEDT: two sweeps propagate the nearest source texel so every
// texel knows an (almost exact) Euclidean nearest source.
void build_nearest_field(NearestField& field, std::span<const float> valid_mask, float threshold) {
    const std::int32_t width = field.width;
    const std::int32_t height = field.height;
    const auto index_of = [width](std::int32_t x, std::int32_t y) {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    };

    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < width; ++x) {
            const std::size_t index = index_of(x, y);
            const bool source = valid_source(valid_mask, index, threshold);
            field.offset_x[index] = source ? 0 : static_cast<std::int16_t>(k_far_offset);
            field.offset_y[index] = source ? 0 : static_cast<std::int16_t>(k_far_offset);
        }
    }

    const auto relax = [&](std::int32_t x, std::int32_t y, std::int32_t neighbour_x, std::int32_t neighbour_y) {
        if (neighbour_x < 0 || neighbour_y < 0 || neighbour_x >= width || neighbour_y >= height) {
            return;
        }
        const std::size_t current = index_of(x, y);
        const std::size_t neighbour = index_of(neighbour_x, neighbour_y);
        const std::int32_t candidate_x =
            static_cast<std::int32_t>(field.offset_x[neighbour]) + (neighbour_x - x);
        const std::int32_t candidate_y =
            static_cast<std::int32_t>(field.offset_y[neighbour]) + (neighbour_y - y);
        const std::int32_t current_x = field.offset_x[current];
        const std::int32_t current_y = field.offset_y[current];
        const std::int64_t candidate_squared =
            static_cast<std::int64_t>(candidate_x) * candidate_x +
            static_cast<std::int64_t>(candidate_y) * candidate_y;
        const std::int64_t current_squared =
            static_cast<std::int64_t>(current_x) * current_x +
            static_cast<std::int64_t>(current_y) * current_y;
        if (candidate_squared >= current_squared) {
            return;
        }
        field.offset_x[current] = static_cast<std::int16_t>(std::clamp(
            candidate_x, -k_maximum_atlas_side, k_maximum_atlas_side));
        field.offset_y[current] = static_cast<std::int16_t>(std::clamp(
            candidate_y, -k_maximum_atlas_side, k_maximum_atlas_side));
    };

    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < width; ++x) {
            relax(x, y, x - 1, y);
            relax(x, y, x, y - 1);
            relax(x, y, x - 1, y - 1);
            relax(x, y, x + 1, y - 1);
        }
    }
    for (std::int32_t y = height - 1; y >= 0; --y) {
        for (std::int32_t x = width - 1; x >= 0; --x) {
            relax(x, y, x + 1, y);
            relax(x, y, x, y + 1);
            relax(x, y, x + 1, y + 1);
            relax(x, y, x - 1, y + 1);
        }
    }
}

} // namespace

float suggest_softmax_scale(
    std::span<const float> positions,
    std::span<const float> camera_positions) {
    if (positions.size() < 3 || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    if (camera_positions.size() < 3 || camera_positions.size() % 3 != 0) {
        throw std::invalid_argument("camera_positions must have shape [camera_count, 3]");
    }
    const auto vertex_count = positions.size() / 3;
    const auto camera_count = camera_positions.size() / 3;
    // Subsample large meshes: the scale only needs the closest approach of the
    // capture rig, which a bounded vertex sample captures reliably.
    constexpr std::size_t k_maximum_samples = 10000;
    const auto stride = std::max<std::size_t>(1, vertex_count / k_maximum_samples);
    double minimum_squared_distance = std::numeric_limits<double>::infinity();
    for (std::size_t vertex = 0; vertex < vertex_count; vertex += stride) {
        const double x = positions[vertex * 3 + 0];
        const double y = positions[vertex * 3 + 1];
        const double z = positions[vertex * 3 + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            continue;
        }
        for (std::size_t camera = 0; camera < camera_count; ++camera) {
            const double dx = camera_positions[camera * 3 + 0] - x;
            const double dy = camera_positions[camera * 3 + 1] - y;
            const double dz = camera_positions[camera * 3 + 2] - z;
            const double squared = dx * dx + dy * dy + dz * dz;
            minimum_squared_distance = std::min(minimum_squared_distance, squared);
        }
    }
    if (!std::isfinite(minimum_squared_distance) || minimum_squared_distance <= 0.0) {
        return 0.0F;
    }
    // Matches the reference softmax definition: exponent = scale * |n.v| / d^2,
    // with the scale keeping the peak exponent near 20 for the best view.
    return static_cast<float>(20.0 * minimum_squared_distance);
}

TexturePaddingStats pad_texture_atlas(
    TextureBakeOutput& output, const TexturePaddingOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    const std::uint32_t width = output.width;
    const std::uint32_t height = output.height;
    if (width == 0 || height == 0) {
        throw std::invalid_argument("cannot pad an empty atlas");
    }
    if (width > static_cast<std::uint32_t>(k_maximum_atlas_side) ||
        height > static_cast<std::uint32_t>(k_maximum_atlas_side)) {
        throw std::invalid_argument("atlas dimensions exceed the supported maximum");
    }
    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    if (output.color.size() % pixel_count != 0) {
        throw std::invalid_argument("atlas color size does not match its dimensions");
    }
    const std::size_t channel_count = output.color.size() / pixel_count;
    if (channel_count != 3 && channel_count != 4) {
        throw std::invalid_argument("atlas color must have 3 or 4 channels");
    }
    if (output.valid_mask.size() != pixel_count) {
        throw std::invalid_argument("atlas valid_mask must match the atlas dimensions");
    }
    if (!std::isfinite(options.valid_threshold) || options.valid_threshold <= 0.0F ||
        options.valid_threshold > 1.0F) {
        throw std::invalid_argument("valid_threshold must be within (0, 1]");
    }

    TexturePaddingStats stats;
    output.filled_mask.assign(pixel_count, 0.0F);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        if (valid_source(output.valid_mask, pixel, options.valid_threshold)) {
            ++stats.valid_texels;
        }
    }
    if (stats.valid_texels == 0) {
        // Nothing to extrapolate from; keep the atlas untouched.
        stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return stats;
    }

    NearestField field;
    field.width = static_cast<std::int32_t>(width);
    field.height = static_cast<std::int32_t>(height);
    field.offset_x.assign(pixel_count, 0);
    field.offset_y.assign(pixel_count, 0);
    build_nearest_field(field, output.valid_mask, options.valid_threshold);

    const bool has_coverage = output.coverage_mask.size() == pixel_count;
    const std::int64_t margin_squared = static_cast<std::int64_t>(options.margin) * options.margin;
    const std::int64_t fill_limit_squared =
        options.maximum_fill_distance == 0
        ? std::numeric_limits<std::int64_t>::max()
        : static_cast<std::int64_t>(options.maximum_fill_distance) * options.maximum_fill_distance;
    // Texels that never relaxed keep the untouched sentinel offset, whose
    // squared length is exactly twice k_far_offset squared.
    const std::int64_t no_source_squared =
        2LL * static_cast<std::int64_t>(k_far_offset) * k_far_offset;

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            if (valid_source(output.valid_mask, pixel, options.valid_threshold)) {
                continue;
            }
            const std::int32_t offset_x = field.offset_x[pixel];
            const std::int32_t offset_y = field.offset_y[pixel];
            const std::int64_t distance_squared =
                static_cast<std::int64_t>(offset_x) * offset_x +
                static_cast<std::int64_t>(offset_y) * offset_y;
            if (distance_squared >= no_source_squared) {
                continue;
            }
            const std::int32_t source_x = static_cast<std::int32_t>(x) + offset_x;
            const std::int32_t source_y = static_cast<std::int32_t>(y) + offset_y;
            if (source_x < 0 || source_y < 0 || source_x >= field.width || source_y >= field.height) {
                continue;
            }
            const std::size_t source =
                static_cast<std::size_t>(source_y) * width + static_cast<std::size_t>(source_x);
            const bool covered = has_coverage && output.coverage_mask[pixel] > 0.5F;
            bool fill = false;
            bool is_gutter = false;
            if (!covered) {
                is_gutter = true;
                fill = options.margin > 0 && distance_squared <= margin_squared;
            } else if (options.fill_unobserved) {
                fill = distance_squared <= fill_limit_squared;
            }
            if (!fill) {
                if (covered) {
                    ++stats.remaining_unobserved;
                }
                continue;
            }
            for (std::size_t channel = 0; channel < channel_count; ++channel) {
                output.color[pixel * channel_count + channel] = output.color[source * channel_count + channel];
            }
            if (channel_count == 4) {
                output.color[pixel * channel_count + 3] = 1.0F;
            }
            output.filled_mask[pixel] = 1.0F;
            ++stats.filled_texels;
            if (is_gutter) {
                ++stats.gutter_texels;
            } else {
                ++stats.unobserved_texels;
            }
        }
    }
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}

} // namespace photara_drender
