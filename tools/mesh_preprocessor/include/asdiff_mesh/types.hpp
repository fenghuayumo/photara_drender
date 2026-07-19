// Optional GPL/commercial CGAL mesh tooling. Not part of the MIT asdiff_render library.

#pragma once

#include <cstdint>
#include <vector>

namespace asdiff_mesh {

struct TriangleMesh {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
};

struct MeshStatistics {
    std::size_t vertex_count = 0;
    std::size_t edge_count = 0;
    std::size_t face_count = 0;
    std::size_t border_edge_count = 0;
    bool valid_polygon_mesh = false;
    bool triangle_mesh = false;
    bool closed = false;
    bool self_intersecting = false;
};

struct DecimateOptions {
    std::size_t target_face_count = 1'000'000;
    bool check_self_intersections = false;
};

struct RemeshOptions {
    int vertex_count = -1;
    int face_count = -1;
    float scale = -1.0F;
    int rosy = 4;
    int posy = 4;
    float crease_angle = 0.0F;
    bool align_to_boundaries = true;
    bool extrinsic = true;
    int smooth_iterations = 2;
    int knn_points = 1000;
    bool deterministic = true;
};

} // namespace asdiff_mesh
