// This optional library uses CGAL Surface Mesh Simplification, licensed separately under
// GPL-3.0-or-later or a commercial CGAL license. It is not linked into asdiff_render.

#include "asdiff_mesh/pipeline.hpp"

#include "asdiff_mesh/mesh_ops.hpp"
#include "asdiff_render/uv_atlas.hpp"

#include <cmath>
#include <stdexcept>

namespace asdiff_mesh {

std::vector<float> compute_vertex_normals(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices) {
    if (positions.empty() || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    if (triangle_indices.empty() || triangle_indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    std::vector<float> normals(positions.size(), 0.0F);
    const auto face_count = triangle_indices.size() / 3;
    for (std::size_t face = 0; face < face_count; ++face) {
        const auto i0 = triangle_indices[face * 3];
        const auto i1 = triangle_indices[face * 3 + 1];
        const auto i2 = triangle_indices[face * 3 + 2];
        const float ax = positions[i1 * 3] - positions[i0 * 3];
        const float ay = positions[i1 * 3 + 1] - positions[i0 * 3 + 1];
        const float az = positions[i1 * 3 + 2] - positions[i0 * 3 + 2];
        const float bx = positions[i2 * 3] - positions[i0 * 3];
        const float by = positions[i2 * 3 + 1] - positions[i0 * 3 + 1];
        const float bz = positions[i2 * 3 + 2] - positions[i0 * 3 + 2];
        const float nx = ay * bz - az * by;
        const float ny = az * bx - ax * bz;
        const float nz = ax * by - ay * bx;
        for (const auto index : {i0, i1, i2}) {
            normals[index * 3] += nx;
            normals[index * 3 + 1] += ny;
            normals[index * 3 + 2] += nz;
        }
    }
    for (std::size_t vertex = 0; vertex < normals.size() / 3; ++vertex) {
        const float x = normals[vertex * 3];
        const float y = normals[vertex * 3 + 1];
        const float z = normals[vertex * 3 + 2];
        const float length = std::sqrt(x * x + y * y + z * z);
        const float inv = length > 1e-20F ? 1.0F / length : 0.0F;
        normals[vertex * 3] = x * inv;
        normals[vertex * 3 + 1] = y * inv;
        normals[vertex * 3 + 2] = z * inv;
    }
    return normals;
}

PreparedBakeMesh prepare_for_baking(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const PrepareOptions& options) {
    PreparedBakeMesh prepared;
    if (options.use_instant_remesh) {
        prepared.remeshed = remesh_field_aligned(positions, triangle_indices, options.remesh);
    } else {
        prepared.remeshed.positions.assign(positions.begin(), positions.end());
        prepared.remeshed.indices.assign(triangle_indices.begin(), triangle_indices.end());
    }

    prepared.manifold = repair_and_decimate(
        prepared.remeshed.positions, prepared.remeshed.indices, options.decimate);

    const auto source_normals =
        compute_vertex_normals(prepared.manifold.positions, prepared.manifold.indices);
    const auto unwrapped = asdiff_render::unwrap_uv(
        prepared.manifold.positions, prepared.manifold.indices, options.atlas);

    prepared.positions = unwrapped.positions;
    prepared.uv = unwrapped.uv;
    prepared.indices = unwrapped.indices;
    prepared.vertex_remap = unwrapped.vertex_remap;
    prepared.face_chart_ids = unwrapped.face_chart_ids;
    prepared.chart_count = unwrapped.chart_count;
    prepared.partition_count = unwrapped.partition_count;
    prepared.max_stretch = unwrapped.max_stretch;
    prepared.normals.resize(unwrapped.vertex_remap.size() * 3);
    for (std::size_t vertex = 0; vertex < unwrapped.vertex_remap.size(); ++vertex) {
        const auto source = unwrapped.vertex_remap[vertex];
        prepared.normals[vertex * 3] = source_normals[source * 3];
        prepared.normals[vertex * 3 + 1] = source_normals[source * 3 + 1];
        prepared.normals[vertex * 3 + 2] = source_normals[source * 3 + 2];
    }
    return prepared;
}

asdiff_render::TextureBakeOutput bake_texture_atlas(
    asdiff_render::TextureBaker& baker,
    const PreparedBakeMesh& mesh,
    std::span<const asdiff_render::ProjectionView> views,
    const asdiff_render::TextureBakeOptions& options) {
    return baker.bake(mesh.positions, mesh.normals, mesh.uv, mesh.indices, views, options);
}

BakePipelineResult prepare_and_bake(
    asdiff_render::Context& context,
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const asdiff_render::ProjectionView> views,
    const BakePipelineOptions& options) {
    BakePipelineResult result;
    result.mesh = prepare_for_baking(positions, triangle_indices, options.prepare);
    asdiff_render::TextureBaker baker(context);
    result.texture = bake_texture_atlas(baker, result.mesh, views, options.bake);
    return result;
}

} // namespace asdiff_mesh
