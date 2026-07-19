#include "asdiff_render/uv_atlas.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if ASDIFF_HAS_UVATLAS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <UVAtlas.h>
#endif

namespace asdiff_render {
namespace {

void validate_input(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    const UvAtlasOptions& options) {
    if (positions.empty() || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    if (indices.empty() || indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    const auto vertex_count = positions.size() / 3;
    if (std::ranges::any_of(indices, [vertex_count](std::uint32_t index) { return index >= vertex_count; })) {
        throw std::out_of_range("triangle_indices contains an invalid vertex index");
    }
    if (options.width == 0 || options.height == 0 || options.gutter < 0.0F ||
        options.max_stretch < 0.0F || options.max_stretch > 1.0F) {
        throw std::invalid_argument("UV atlas dimensions, gutter, or max_stretch are invalid");
    }
}

#if ASDIFF_HAS_UVATLAS
struct EdgeKey {
    std::uint32_t first;
    std::uint32_t second;

    bool operator==(const EdgeKey&) const = default;
};

struct EdgeHash {
    std::size_t operator()(const EdgeKey& edge) const noexcept {
        return (static_cast<std::size_t>(edge.first) << 32U) ^ edge.second;
    }
};

std::vector<std::uint32_t> build_adjacency(std::span<const std::uint32_t> indices) {
    constexpr std::uint32_t UNUSED = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> adjacency(indices.size(), UNUSED);
    std::unordered_map<EdgeKey, std::pair<std::uint32_t, std::uint32_t>, EdgeHash> unmatched;
    const auto face_count = static_cast<std::uint32_t>(indices.size() / 3);
    for (std::uint32_t face = 0; face < face_count; ++face) {
        for (std::uint32_t edge = 0; edge < 3; ++edge) {
            const auto a = indices[face * 3 + edge];
            const auto b = indices[face * 3 + (edge + 1) % 3];
            const EdgeKey key{std::min(a, b), std::max(a, b)};
            const auto iterator = unmatched.find(key);
            if (iterator == unmatched.end()) {
                unmatched.emplace(key, std::pair{face, edge});
            } else {
                const auto [other_face, other_edge] = iterator->second;
                adjacency[face * 3 + edge] = other_face;
                adjacency[other_face * 3 + other_edge] = face;
                unmatched.erase(iterator);
            }
        }
    }
    return adjacency;
}
#endif

} // namespace

bool has_uv_atlas_backend() noexcept {
#if ASDIFF_HAS_UVATLAS
    return true;
#else
    return false;
#endif
}

UvAtlasOutput unwrap_uv(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const UvAtlasOptions& options) {
    validate_input(positions, triangle_indices, options);
#if !ASDIFF_HAS_UVATLAS
    throw std::runtime_error(
        "Microsoft UVAtlas support was not built; configure with ASDIFF_ENABLE_UVATLAS=ON and install uvatlas");
#else
    std::vector<DirectX::XMFLOAT3> directx_positions(positions.size() / 3);
    std::memcpy(directx_positions.data(), positions.data(), positions.size_bytes());
    const auto adjacency = build_adjacency(triangle_indices);
    std::vector<DirectX::UVAtlasVertex> output_vertices;
    std::vector<std::uint8_t> output_index_bytes;
    std::vector<std::uint32_t> face_partitions;
    std::vector<std::uint32_t> vertex_remap;
    float output_stretch = 0.0F;
    std::size_t output_chart_count = 0;
    const auto atlas_flags = options.quality ? DirectX::UVATLAS_GEODESIC_QUALITY : DirectX::UVATLAS_GEODESIC_FAST;
    const HRESULT result = DirectX::UVAtlasCreate(
        directx_positions.data(),
        directx_positions.size(),
        triangle_indices.data(),
        DXGI_FORMAT_R32_UINT,
        triangle_indices.size() / 3,
        options.max_chart_count,
        options.max_stretch,
        options.width,
        options.height,
        options.gutter,
        adjacency.data(),
        nullptr,
        nullptr,
        {},
        DirectX::UVATLAS_DEFAULT_CALLBACK_FREQUENCY,
        atlas_flags,
        output_vertices,
        output_index_bytes,
        &face_partitions,
        &vertex_remap,
        &output_stretch,
        &output_chart_count);
    if (FAILED(result)) {
        throw std::runtime_error("UVAtlasCreate failed with HRESULT " + std::to_string(result));
    }
    if (output_index_bytes.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("UVAtlas returned an unexpected index format");
    }

    UvAtlasOutput output;
    output.positions.resize(output_vertices.size() * 3);
    output.uv.resize(output_vertices.size() * 2);
    for (std::size_t vertex = 0; vertex < output_vertices.size(); ++vertex) {
        const auto& source = output_vertices[vertex];
        output.positions[vertex * 3 + 0] = source.pos.x;
        output.positions[vertex * 3 + 1] = source.pos.y;
        output.positions[vertex * 3 + 2] = source.pos.z;
        output.uv[vertex * 2 + 0] = source.uv.x;
        output.uv[vertex * 2 + 1] = source.uv.y;
    }
    output.indices.resize(output_index_bytes.size() / sizeof(std::uint32_t));
    std::memcpy(output.indices.data(), output_index_bytes.data(), output_index_bytes.size());
    output.vertex_remap = std::move(vertex_remap);
    output.face_chart_ids = std::move(face_partitions);
    output.chart_count = static_cast<std::uint32_t>(output_chart_count);
    output.max_stretch = output_stretch;
    return output;
#endif
}

} // namespace asdiff_render
