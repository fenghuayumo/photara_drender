// Optional GPL/commercial CGAL mesh tooling. Not part of the MIT asdiff_render library.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "asdiff_mesh/types.hpp"
#include "asdiff_render/texture_baker.hpp"
#include "asdiff_render/types.hpp"

namespace asdiff_mesh {

struct PrepareOptions {
    DecimateOptions decimate;
    RemeshOptions remesh;
    bool use_instant_remesh = false;
    asdiff_render::UvAtlasOptions atlas;
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
    asdiff_render::TextureBakeOptions bake;
};

struct BakePipelineResult {
    PreparedBakeMesh mesh;
    asdiff_render::TextureBakeOutput texture;
};

[[nodiscard]] std::vector<float> compute_vertex_normals(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices);

[[nodiscard]] PreparedBakeMesh prepare_for_baking(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const PrepareOptions& options = {});

[[nodiscard]] asdiff_render::TextureBakeOutput bake_texture_atlas(
    asdiff_render::TextureBaker& baker,
    const PreparedBakeMesh& mesh,
    std::span<const asdiff_render::ProjectionView> views,
    const asdiff_render::TextureBakeOptions& options = {});

[[nodiscard]] BakePipelineResult prepare_and_bake(
    asdiff_render::Context& context,
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const asdiff_render::ProjectionView> views,
    const BakePipelineOptions& options = {});

} // namespace asdiff_mesh
