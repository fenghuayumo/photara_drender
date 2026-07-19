// This optional tool uses CGAL Surface Mesh Simplification, licensed separately under
// GPL-3.0-or-later or a commercial CGAL license. It is not linked into asdiff_render.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
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

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Edge = boost::graph_traits<Mesh>::edge_descriptor;

class NullStreamBuffer final : public std::streambuf {
protected:
    int overflow(int value) override { return traits_type::not_eof(value); }
};

class ScopedStreamBuffer final {
public:
    ScopedStreamBuffer(std::ostream& stream, std::streambuf& replacement)
        : stream_(stream), previous_(stream.rdbuf(&replacement)) {}
    ~ScopedStreamBuffer() { stream_.rdbuf(previous_); }

    ScopedStreamBuffer(const ScopedStreamBuffer&) = delete;
    ScopedStreamBuffer& operator=(const ScopedStreamBuffer&) = delete;

private:
    std::ostream& stream_;
    std::streambuf* previous_;
};

struct MeshStatistics {
    std::size_t vertices = 0;
    std::size_t edges = 0;
    std::size_t faces = 0;
    std::size_t border_edges = 0;
    bool valid_polygon_mesh = false;
    bool triangle_mesh = false;
    bool closed = false;
    bool self_intersecting = false;
};

MeshStatistics inspect_mesh(const Mesh& mesh, bool check_self_intersections) {
    MeshStatistics statistics;
    statistics.vertices = num_vertices(mesh);
    statistics.edges = num_edges(mesh);
    statistics.faces = num_faces(mesh);
    for (const Edge edge : edges(mesh)) {
        if (CGAL::is_border(edge, mesh)) {
            ++statistics.border_edges;
        }
    }
    statistics.valid_polygon_mesh = CGAL::is_valid_polygon_mesh(mesh);
    statistics.triangle_mesh = CGAL::is_triangle_mesh(mesh);
    statistics.closed = CGAL::is_closed(mesh);
    statistics.self_intersecting = check_self_intersections && PMP::does_self_intersect(mesh);
    return statistics;
}

void print_statistics(std::string_view label, const MeshStatistics& statistics) {
    std::cout << label
              << " vertices=" << statistics.vertices
              << " edges=" << statistics.edges
              << " faces=" << statistics.faces
              << " border_edges=" << statistics.border_edges
              << " valid=" << statistics.valid_polygon_mesh
              << " triangles=" << statistics.triangle_mesh
              << " closed=" << statistics.closed
              << " self_intersecting=" << statistics.self_intersecting << '\n';
}

std::size_t parse_count(const char* value) {
    std::size_t result = 0;
    const std::string_view text(value);
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result < 4) {
        throw std::invalid_argument("target_face_count must be an integer greater than three");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "usage: asdiff_mesh_preprocessor <input> <output> <target_face_count> [--check-self-intersections]\n";
            return 2;
        }
        const std::filesystem::path input_path(argv[1]);
        const std::filesystem::path output_path(argv[2]);
        const std::size_t target_face_count = parse_count(argv[3]);
        const bool check_self_intersections = argc == 5 && std::string_view(argv[4]) == "--check-self-intersections";
        if (argc == 5 && !check_self_intersections) {
            throw std::invalid_argument("unknown option; expected --check-self-intersections");
        }

        std::vector<Point> soup_points;
        std::vector<std::vector<std::size_t>> soup_polygons;
        if (!CGAL::IO::read_polygon_soup(input_path.string(), soup_points, soup_polygons) || soup_polygons.empty()) {
            throw std::runtime_error("failed to read input polygon soup");
        }
        const std::size_t original_soup_point_count = soup_points.size();
        PMP::repair_polygon_soup(soup_points, soup_polygons);
        const bool soup_oriented_without_duplication = PMP::orient_polygon_soup(soup_points, soup_polygons);
        std::cout << "soup points=" << soup_points.size()
                  << " polygons=" << soup_polygons.size()
                  << " duplicated_for_manifold=" << (soup_points.size() - original_soup_point_count)
                  << " oriented_without_duplication=" << soup_oriented_without_duplication << '\n';
        Mesh mesh;
        PMP::polygon_soup_to_polygon_mesh(soup_points, soup_polygons, mesh);
        if (mesh.is_empty()) {
            throw std::runtime_error("polygon soup conversion produced an empty mesh");
        }
        print_statistics("input", inspect_mesh(mesh, false));

        if (!PMP::triangulate_faces(mesh)) {
            throw std::runtime_error("CGAL failed to triangulate the input mesh");
        }
        const bool removed_all_degenerate_faces = PMP::remove_degenerate_faces(mesh);
        const std::size_t stitched_border_pairs = PMP::stitch_borders(mesh);
        const std::size_t duplicated_non_manifold_vertices = PMP::duplicate_non_manifold_vertices(mesh);
        const std::size_t removed_isolated_vertices = PMP::remove_isolated_vertices(mesh);
        mesh.collect_garbage();
        std::cout << "repair removed_all_degenerate_faces=" << removed_all_degenerate_faces
                  << " stitched_border_pairs=" << stitched_border_pairs
                  << " duplicated_non_manifold_vertices=" << duplicated_non_manifold_vertices
                  << " removed_isolated_vertices=" << removed_isolated_vertices << '\n';
        const auto repaired_statistics = inspect_mesh(mesh, false);
        print_statistics("repaired", repaired_statistics);
        if (!repaired_statistics.valid_polygon_mesh || !repaired_statistics.triangle_mesh) {
            throw std::runtime_error("repair did not produce a valid manifold triangle mesh");
        }

        auto constrained_edges = mesh.add_property_map<Edge, bool>("e:is_constrained", false).first;
        for (const Edge edge : edges(mesh)) {
            constrained_edges[edge] = CGAL::is_border(edge, mesh);
        }
        if (num_faces(mesh) > target_face_count) {
            const SMS::Face_count_stop_predicate<Mesh> stop(target_face_count);
            const std::size_t removed_edges = SMS::edge_collapse(
                mesh,
                stop,
                CGAL::parameters::get_cost(SMS::LindstromTurk_cost<Mesh>())
                    .get_placement(SMS::LindstromTurk_placement<Mesh>())
                    .edge_is_constrained_map(constrained_edges));
            std::cout << "decimation removed_edges=" << removed_edges << '\n';
        }
        PMP::remove_degenerate_faces(mesh);
        PMP::remove_isolated_vertices(mesh);
        mesh.collect_garbage();

        const auto output_statistics = inspect_mesh(mesh, check_self_intersections);
        print_statistics("output", output_statistics);
        if (!output_statistics.valid_polygon_mesh || !output_statistics.triangle_mesh ||
            output_statistics.self_intersecting) {
            throw std::runtime_error("output mesh failed manifold or self-intersection validation");
        }
        // CGAL 6.2's PLY graph writer prints one diagnostic per vertex. Suppress only
        // the writer call; preprocessing statistics above remain machine-readable.
        NullStreamBuffer null_stream_buffer;
        bool wrote_mesh = false;
        {
            ScopedStreamBuffer redirect(std::cout, null_stream_buffer);
            wrote_mesh = CGAL::IO::write_polygon_mesh(
                output_path.string(), mesh,
                CGAL::parameters::stream_precision(17).use_binary_mode(true));
        }
        if (!wrote_mesh) {
            throw std::runtime_error("failed to write output polygon mesh");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mesh preprocessing failed: " << error.what() << '\n';
        return 1;
    }
}
