// Optional GPL/commercial CGAL mesh tooling. Not part of the MIT aether_drender library.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "aether_mesh/types.hpp"
#include "aether_drender/texture_baker.hpp"
#include "aether_drender/types.hpp"

namespace aether_mesh {

struct PrepareOptions {
    DecimateOptions decimate;
    RemeshOptions remesh;
    bool use_instant_remesh = false;
    aether_drender::UvAtlasOptions atlas;
};

struct PreparedBakeMesh {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uv;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> vertex_remap;
    std::vector<std::uint32_t> face_chart_ids;
    std::uint32_t chart_count = 0;
    std::uint32_t partition_count = 1;
    float max_stretch = 0.0F;
    TriangleMesh remeshed;
    TriangleMesh manifold;
};

struct BakePipelineOptions {
    PrepareOptions prepare;
    aether_drender::TextureBakeOptions bake;
};

struct BakePipelineResult {
    PreparedBakeMesh mesh;
    aether_drender::TextureBakeOutput texture;
};

[[nodiscard]] std::vector<float> compute_vertex_normals(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices);

[[nodiscard]] PreparedBakeMesh prepare_for_baking(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const PrepareOptions& options = {});

[[nodiscard]] aether_drender::TextureBakeOutput bake_texture_atlas(
    aether_drender::TextureBaker& baker,
    const PreparedBakeMesh& mesh,
    std::span<const aether_drender::ProjectionView> views,
    const aether_drender::TextureBakeOptions& options = {});

[[nodiscard]] BakePipelineResult prepare_and_bake(
    aether_drender::Context& context,
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const aether_drender::ProjectionView> views,
    const BakePipelineOptions& options = {});

} // namespace aether_mesh
