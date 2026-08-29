// Optional GPL/commercial CGAL mesh tooling. Not part of the MIT aether_drender library.

#pragma once

#include <span>

#include "aether_mesh/types.hpp"

namespace aether_mesh {

[[nodiscard]] bool has_instant_meshes_backend() noexcept;

// Field-aligned remeshing. Output is always triangulated for downstream CGAL/UVAtlas.
[[nodiscard]] TriangleMesh remesh_field_aligned(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const RemeshOptions& options = {});

[[nodiscard]] TriangleMesh repair_and_decimate(
    std::span<const float> positions,
    std::span<const std::uint32_t> face_indices,
    std::uint32_t corners_per_face,
    const DecimateOptions& options = {});

[[nodiscard]] TriangleMesh repair_and_decimate(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const DecimateOptions& options = {});

[[nodiscard]] MeshStatistics inspect_triangle_mesh(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    bool check_self_intersections = false);

} // namespace aether_mesh
