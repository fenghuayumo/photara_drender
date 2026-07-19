// This optional library uses CGAL Surface Mesh Simplification, licensed separately under
// GPL-3.0-or-later or a commercial CGAL license. It is not linked into asdiff_render.

#include "asdiff_mesh/mesh_ops.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/boost/graph/helpers.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

namespace asdiff_mesh {
namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Edge = boost::graph_traits<Mesh>::edge_descriptor;

void validate_mesh_input(
    std::span<const float> positions,
    std::span<const std::uint32_t> face_indices,
    std::uint32_t corners_per_face) {
    if (positions.empty() || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    if (corners_per_face < 3) {
        throw std::invalid_argument("corners_per_face must be at least three");
    }
    if (face_indices.empty() || face_indices.size() % corners_per_face != 0) {
        throw std::invalid_argument("face_indices must have shape [face_count, corners_per_face]");
    }
    const auto vertex_count = positions.size() / 3;
    for (const auto index : face_indices) {
        if (index >= vertex_count) {
            throw std::out_of_range("face_indices contains an invalid vertex index");
        }
    }
}

MeshStatistics inspect_mesh(const Mesh& mesh, bool check_self_intersections) {
    MeshStatistics statistics;
    statistics.vertex_count = num_vertices(mesh);
    statistics.edge_count = num_edges(mesh);
    statistics.face_count = num_faces(mesh);
    for (const Edge edge : edges(mesh)) {
        if (CGAL::is_border(edge, mesh)) {
            ++statistics.border_edge_count;
        }
    }
    statistics.valid_polygon_mesh = CGAL::is_valid_polygon_mesh(mesh);
    statistics.triangle_mesh = CGAL::is_triangle_mesh(mesh);
    statistics.closed = CGAL::is_closed(mesh);
    statistics.self_intersecting = check_self_intersections && PMP::does_self_intersect(mesh);
    return statistics;
}

TriangleMesh mesh_to_triangle_mesh(const Mesh& mesh) {
    TriangleMesh output;
    std::unordered_map<Mesh::Vertex_index, std::uint32_t> vertex_map;
    vertex_map.reserve(mesh.number_of_vertices());
    output.positions.reserve(mesh.number_of_vertices() * 3);
    std::uint32_t next_index = 0;
    for (const auto vertex : mesh.vertices()) {
        const Point& point = mesh.point(vertex);
        vertex_map.emplace(vertex, next_index++);
        output.positions.push_back(static_cast<float>(point.x()));
        output.positions.push_back(static_cast<float>(point.y()));
        output.positions.push_back(static_cast<float>(point.z()));
    }
    output.indices.reserve(mesh.number_of_faces() * 3);
    for (const auto face : mesh.faces()) {
        std::array<std::uint32_t, 3> triangle{};
        std::uint32_t corner = 0;
        for (const auto vertex : vertices_around_face(mesh.halfedge(face), mesh)) {
            if (corner >= 3) {
                throw std::runtime_error("repair_and_decimate produced a non-triangle face");
            }
            triangle[corner++] = vertex_map.at(vertex);
        }
        if (corner != 3) {
            throw std::runtime_error("repair_and_decimate produced a non-triangle face");
        }
        output.indices.insert(output.indices.end(), triangle.begin(), triangle.end());
    }
    return output;
}

} // namespace

TriangleMesh repair_and_decimate(
    std::span<const float> positions,
    std::span<const std::uint32_t> face_indices,
    std::uint32_t corners_per_face,
    const DecimateOptions& options) {
    validate_mesh_input(positions, face_indices, corners_per_face);
    if (options.target_face_count < 4) {
        throw std::invalid_argument("target_face_count must be greater than three");
    }

    const auto vertex_count = positions.size() / 3;
    const auto face_count = face_indices.size() / corners_per_face;
    std::vector<Point> soup_points;
    soup_points.reserve(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        soup_points.emplace_back(
            positions[vertex * 3],
            positions[vertex * 3 + 1],
            positions[vertex * 3 + 2]);
    }
    std::vector<std::vector<std::size_t>> soup_polygons;
    soup_polygons.reserve(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        std::vector<std::size_t> polygon(corners_per_face);
        for (std::uint32_t corner = 0; corner < corners_per_face; ++corner) {
            polygon[corner] = face_indices[face * corners_per_face + corner];
        }
        soup_polygons.push_back(std::move(polygon));
    }

    PMP::repair_polygon_soup(soup_points, soup_polygons);
    PMP::orient_polygon_soup(soup_points, soup_polygons);
    Mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(soup_points, soup_polygons, mesh);
    if (mesh.is_empty()) {
        throw std::runtime_error("polygon soup conversion produced an empty mesh");
    }
    if (!PMP::triangulate_faces(mesh)) {
        throw std::runtime_error("CGAL failed to triangulate the input mesh");
    }
    PMP::remove_degenerate_faces(mesh);
    PMP::stitch_borders(mesh);
    PMP::duplicate_non_manifold_vertices(mesh);
    PMP::remove_isolated_vertices(mesh);
    mesh.collect_garbage();

    auto repaired = inspect_mesh(mesh, false);
    if (!repaired.valid_polygon_mesh || !repaired.triangle_mesh) {
        throw std::runtime_error("repair did not produce a valid manifold triangle mesh");
    }

    auto constrained_edges = mesh.add_property_map<Edge, bool>("e:is_constrained", false).first;
    for (const Edge edge : edges(mesh)) {
        constrained_edges[edge] = CGAL::is_border(edge, mesh);
    }
    if (num_faces(mesh) > options.target_face_count) {
        const SMS::Face_count_stop_predicate<Mesh> stop(options.target_face_count);
        SMS::edge_collapse(
            mesh,
            stop,
            CGAL::parameters::get_cost(SMS::LindstromTurk_cost<Mesh>())
                .get_placement(SMS::LindstromTurk_placement<Mesh>())
                .edge_is_constrained_map(constrained_edges));
    }
    PMP::remove_degenerate_faces(mesh);
    PMP::remove_isolated_vertices(mesh);
    mesh.collect_garbage();

    const auto output_statistics = inspect_mesh(mesh, options.check_self_intersections);
    if (!output_statistics.valid_polygon_mesh || !output_statistics.triangle_mesh ||
        output_statistics.self_intersecting) {
        throw std::runtime_error("output mesh failed manifold or self-intersection validation");
    }
    return mesh_to_triangle_mesh(mesh);
}

TriangleMesh repair_and_decimate(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const DecimateOptions& options) {
    return repair_and_decimate(positions, triangle_indices, 3, options);
}

MeshStatistics inspect_triangle_mesh(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    bool check_self_intersections) {
    validate_mesh_input(positions, triangle_indices, 3);
    const auto repaired = repair_and_decimate(
        positions,
        triangle_indices,
        DecimateOptions{
            .target_face_count = std::max<std::size_t>(4, triangle_indices.size() / 3),
            .check_self_intersections = check_self_intersections,
        });
    MeshStatistics statistics;
    statistics.vertex_count = repaired.positions.size() / 3;
    statistics.face_count = repaired.indices.size() / 3;
    statistics.valid_polygon_mesh = true;
    statistics.triangle_mesh = true;
    statistics.self_intersecting = false;
    return statistics;
}

#if !ASDIFF_HAS_INSTANT_MESHES
bool has_instant_meshes_backend() noexcept {
    return false;
}

TriangleMesh remesh_field_aligned(
    std::span<const float>,
    std::span<const std::uint32_t>,
    const RemeshOptions&) {
    throw std::runtime_error(
        "Instant Meshes support was not built; configure with ASDIFF_ENABLE_INSTANT_MESHES=ON");
}
#endif

} // namespace asdiff_mesh
