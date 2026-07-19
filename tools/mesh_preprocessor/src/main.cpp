// Optional CLI wrapper around the in-memory asdiff_mesh API.
// Uses CGAL (GPL-3.0-or-later/commercial) and is not linked into asdiff_render.

#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <vector>

#include "asdiff_mesh/mesh_ops.hpp"

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;

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
            std::cerr << "usage: asdiff_mesh_preprocessor <input> <output> <target_face_count> "
                         "[--check-self-intersections]\n";
            return 2;
        }
        const std::filesystem::path input_path(argv[1]);
        const std::filesystem::path output_path(argv[2]);
        const std::size_t target_face_count = parse_count(argv[3]);
        const bool check_self_intersections =
            argc == 5 && std::string_view(argv[4]) == "--check-self-intersections";
        if (argc == 5 && !check_self_intersections) {
            throw std::invalid_argument("unknown option; expected --check-self-intersections");
        }

        std::vector<Point> soup_points;
        std::vector<std::vector<std::size_t>> soup_polygons;
        if (!CGAL::IO::read_polygon_soup(input_path.string(), soup_points, soup_polygons) ||
            soup_polygons.empty()) {
            throw std::runtime_error("failed to read input polygon soup");
        }

        std::vector<float> positions(soup_points.size() * 3);
        for (std::size_t vertex = 0; vertex < soup_points.size(); ++vertex) {
            positions[vertex * 3] = static_cast<float>(soup_points[vertex].x());
            positions[vertex * 3 + 1] = static_cast<float>(soup_points[vertex].y());
            positions[vertex * 3 + 2] = static_cast<float>(soup_points[vertex].z());
        }
        std::uint32_t corners = 0;
        for (const auto& polygon : soup_polygons) {
            if (polygon.size() < 3) {
                throw std::runtime_error("input contains a polygon with fewer than three vertices");
            }
            if (corners == 0) {
                corners = static_cast<std::uint32_t>(polygon.size());
            } else if (polygon.size() != corners) {
                // Mixed polygons: triangulate via CGAL by forcing triangle soup through fan for CLI only.
                corners = 0;
                break;
            }
        }

        asdiff_mesh::TriangleMesh result;
        if (corners >= 3) {
            std::vector<std::uint32_t> face_indices;
            face_indices.reserve(soup_polygons.size() * corners);
            for (const auto& polygon : soup_polygons) {
                for (const auto index : polygon) {
                    face_indices.push_back(static_cast<std::uint32_t>(index));
                }
            }
            result = asdiff_mesh::repair_and_decimate(
                positions,
                face_indices,
                corners,
                asdiff_mesh::DecimateOptions{target_face_count, check_self_intersections});
        } else {
            // Fan-triangulate mixed polygons in memory before repair/decimate.
            std::vector<std::uint32_t> triangle_indices;
            for (const auto& polygon : soup_polygons) {
                for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
                    triangle_indices.push_back(static_cast<std::uint32_t>(polygon[0]));
                    triangle_indices.push_back(static_cast<std::uint32_t>(polygon[corner]));
                    triangle_indices.push_back(static_cast<std::uint32_t>(polygon[corner + 1]));
                }
            }
            result = asdiff_mesh::repair_and_decimate(
                positions,
                triangle_indices,
                asdiff_mesh::DecimateOptions{target_face_count, check_self_intersections});
        }

        Mesh mesh;
        std::vector<Mesh::Vertex_index> vertices;
        vertices.reserve(result.positions.size() / 3);
        for (std::size_t vertex = 0; vertex < result.positions.size() / 3; ++vertex) {
            vertices.push_back(mesh.add_vertex(Point(
                result.positions[vertex * 3],
                result.positions[vertex * 3 + 1],
                result.positions[vertex * 3 + 2])));
        }
        for (std::size_t face = 0; face < result.indices.size() / 3; ++face) {
            mesh.add_face(
                vertices[result.indices[face * 3]],
                vertices[result.indices[face * 3 + 1]],
                vertices[result.indices[face * 3 + 2]]);
        }

        std::cout << "output vertices=" << result.positions.size() / 3
                  << " faces=" << result.indices.size() / 3 << '\n';

        NullStreamBuffer null_stream_buffer;
        bool wrote_mesh = false;
        {
            ScopedStreamBuffer redirect(std::cout, null_stream_buffer);
            wrote_mesh = CGAL::IO::write_polygon_mesh(
                output_path.string(),
                mesh,
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
