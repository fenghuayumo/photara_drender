// Instant Meshes field-aligned remeshing backend (BSD), fetched via FetchContent.
// Kept outside the MIT photara_drender library; linked only into optional photara_mesh.

#include "photara_mesh/mesh_ops.hpp"

#include <stdexcept>

#if !PHOTARA_HAS_INSTANT_MESHES

namespace photara_mesh {

bool has_instant_meshes_backend() noexcept {
    return false;
}

TriangleMesh remesh_field_aligned(
    std::span<const float>,
    std::span<const std::uint32_t>,
    const RemeshOptions&) {
    throw std::runtime_error(
        "Instant Meshes support was not built; configure with PHOTARA_ENABLE_INSTANT_MESHES=ON");
}

} // namespace photara_mesh

#else

#include <bvh.h>
#include <common.h>
#include <dedge.h>
#include <extract.h>
#include <field.h>
#include <hierarchy.h>
#include <meshstats.h>
#include <normal.h>
#include <subdivide.h>

#include <tbb/task_scheduler_init.h>

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace photara_mesh {
namespace {

class NullStreamBuffer final : public std::streambuf {
protected:
    int overflow(int value) override { return traits_type::not_eof(value); }
};

class ScopedStreamRedirect final {
public:
    ScopedStreamRedirect(std::ostream& stream, std::streambuf& replacement)
        : stream_(stream), previous_(stream.rdbuf(&replacement)) {}
    ~ScopedStreamRedirect() { stream_.rdbuf(previous_); }

    ScopedStreamRedirect(const ScopedStreamRedirect&) = delete;
    ScopedStreamRedirect& operator=(const ScopedStreamRedirect&) = delete;

private:
    std::ostream& stream_;
    std::streambuf* previous_;
};

void validate_triangle_input(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices) {
    if (positions.empty() || positions.size() % 3 != 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
    if (triangle_indices.empty() || triangle_indices.size() % 3 != 0) {
        throw std::invalid_argument("triangle_indices must have shape [triangle_count, 3]");
    }
    const auto vertex_count = positions.size() / 3;
    for (const auto index : triangle_indices) {
        if (index >= vertex_count) {
            throw std::out_of_range("triangle_indices contains an invalid vertex index");
        }
    }
}

TriangleMesh triangulate_extracted_mesh(const MatrixXf& vertices, const MatrixXu& faces) {
    TriangleMesh output;
    output.positions.resize(static_cast<std::size_t>(vertices.cols()) * 3);
    for (int vertex = 0; vertex < vertices.cols(); ++vertex) {
        output.positions[static_cast<std::size_t>(vertex) * 3] = vertices(0, vertex);
        output.positions[static_cast<std::size_t>(vertex) * 3 + 1] = vertices(1, vertex);
        output.positions[static_cast<std::size_t>(vertex) * 3 + 2] = vertices(2, vertex);
    }

    const int corners = faces.rows();
    if (corners != 3 && corners != 4) {
        throw std::runtime_error("Instant Meshes returned an unsupported face size");
    }
    output.indices.reserve(static_cast<std::size_t>(faces.cols()) * (corners == 4 ? 6 : 3));
    for (int face = 0; face < faces.cols(); ++face) {
        const auto i0 = faces(0, face);
        const auto i1 = faces(1, face);
        const auto i2 = faces(2, face);
        output.indices.push_back(i0);
        output.indices.push_back(i1);
        output.indices.push_back(i2);
        // Instant Meshes represents an irregular triangle in a quad-dominant
        // result by repeating its final corner. Do not emit the corresponding
        // degenerate second triangle.
        if (corners == 4 && faces(3, face) != i2) {
            const auto i3 = faces(3, face);
            output.indices.push_back(i0);
            output.indices.push_back(i2);
            output.indices.push_back(i3);
        }
    }
    return output;
}

} // namespace

bool has_instant_meshes_backend() noexcept {
    return true;
}

TriangleMesh remesh_field_aligned(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const RemeshOptions& options) {
    validate_triangle_input(positions, triangle_indices);
    if (options.rosy != 2 && options.rosy != 4 && options.rosy != 6) {
        throw std::invalid_argument("rosy must be 2, 4, or 6");
    }
    if (options.posy != 3 && options.posy != 4 && options.posy != 6) {
        throw std::invalid_argument("posy must be 3, 4, or 6");
    }

    NullStreamBuffer null_buffer;
    ScopedStreamRedirect redirect_cout(std::cout, null_buffer);
    ScopedStreamRedirect redirect_cerr(std::cerr, null_buffer);

    tbb::task_scheduler_init scheduler;

    const int vertex_count_in = static_cast<int>(positions.size() / 3);
    const int face_count_in = static_cast<int>(triangle_indices.size() / 3);
    MatrixXu F(3, face_count_in);
    MatrixXf V(3, vertex_count_in);
    MatrixXf N;
    VectorXf A;
    std::set<uint32_t> crease_in;
    std::set<uint32_t> crease_out;
    BVH* bvh = nullptr;
    AdjacencyMatrix adj = nullptr;

    for (int vertex = 0; vertex < vertex_count_in; ++vertex) {
        V(0, vertex) = positions[static_cast<std::size_t>(vertex) * 3];
        V(1, vertex) = positions[static_cast<std::size_t>(vertex) * 3 + 1];
        V(2, vertex) = positions[static_cast<std::size_t>(vertex) * 3 + 2];
    }
    for (int face = 0; face < face_count_in; ++face) {
        F(0, face) = triangle_indices[static_cast<std::size_t>(face) * 3];
        F(1, face) = triangle_indices[static_cast<std::size_t>(face) * 3 + 1];
        F(2, face) = triangle_indices[static_cast<std::size_t>(face) * 3 + 2];
    }

    MeshStats stats = compute_mesh_stats(F, V, options.deterministic);
    int vertex_count = options.vertex_count;
    int face_count = options.face_count;
    Float scale = options.scale;
    const int posy = options.posy == 3 ? 6 : options.posy;

    if (scale < 0 && vertex_count < 0 && face_count < 0) {
        vertex_count = std::max(4, vertex_count_in / 16);
    }
    if (scale > 0) {
        const Float face_area =
            posy == 4 ? (scale * scale) : (std::sqrt(3.f) / 4.f * scale * scale);
        face_count = static_cast<int>(stats.mSurfaceArea / face_area);
        vertex_count = posy == 4 ? face_count : (face_count / 2);
    } else if (face_count > 0) {
        const Float face_area = stats.mSurfaceArea / face_count;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
        scale = posy == 4 ? std::sqrt(face_area)
                          : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    } else if (vertex_count > 0) {
        face_count = posy == 4 ? vertex_count : (vertex_count * 2);
        const Float face_area = stats.mSurfaceArea / face_count;
        scale = posy == 4 ? std::sqrt(face_area)
                          : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    }

    MultiResolutionHierarchy mRes;
    VectorXu V2E;
    VectorXu E2E;
    VectorXb boundary;
    VectorXb nonManifold;
    if (stats.mMaximumEdgeLength * 2 > scale ||
        stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);
        subdivide(
            F,
            V,
            V2E,
            E2E,
            boundary,
            nonManifold,
            std::min(scale / 2, static_cast<Float>(stats.mAverageEdgeLength * 2)),
            options.deterministic);
    }
    build_dedge(F, V, V2E, E2E, boundary, nonManifold);
    adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);
    if (options.crease_angle >= 0) {
        generate_crease_normals(
            F, V, V2E, E2E, boundary, nonManifold, options.crease_angle, N, crease_in);
    } else {
        generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);
    }
    compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);
    mRes.setE2E(std::move(E2E));
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);
    mRes.build(options.deterministic);
    mRes.resetSolution();

    if (options.align_to_boundaries) {
        mRes.clearConstraints();
        for (uint32_t edge = 0; edge < 3 * static_cast<uint32_t>(mRes.F().cols()); ++edge) {
            if (mRes.E2E()[edge] == INVALID) {
                const uint32_t i0 = mRes.F()(edge % 3, edge / 3);
                const uint32_t i1 = mRes.F()((edge + 1) % 3, edge / 3);
                Vector3f p0 = mRes.V().col(i0);
                Vector3f p1 = mRes.V().col(i1);
                Vector3f edge_dir = p1 - p0;
                if (edge_dir.squaredNorm() > 0) {
                    edge_dir.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge_dir;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] = mRes.COw()[i1] = 1.0f;
                }
            }
        }
        mRes.propagateConstraints(options.rosy, posy);
    }

    if (options.smooth_iterations > 0) {
        bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
        bvh->build();
    }

    Optimizer optimizer(mRes, false);
    optimizer.setRoSy(options.rosy);
    optimizer.setPoSy(posy);
    optimizer.setExtrinsic(options.extrinsic);
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    optimizer.wait();

    std::map<uint32_t, uint32_t> singularities;
    compute_orientation_singularities(mRes, singularities, options.extrinsic, options.rosy);

    optimizer.optimizePositions(-1);
    optimizer.notify();
    optimizer.wait();
    optimizer.shutdown();

    MatrixXf O_extr;
    MatrixXf N_extr;
    MatrixXf Nf_extr;
    std::vector<std::vector<TaggedLink>> adj_extr;
    extract_graph(
        mRes,
        options.extrinsic,
        options.rosy,
        posy,
        adj_extr,
        O_extr,
        N_extr,
        crease_in,
        crease_out,
        options.deterministic);

    MatrixXu F_extr;
    extract_faces(
        adj_extr,
        O_extr,
        N_extr,
        Nf_extr,
        F_extr,
        posy,
        mRes.scale(),
        crease_out,
        true,
        false,
        bvh,
        options.smooth_iterations);
    delete bvh;

    if (O_extr.cols() == 0 || F_extr.cols() == 0) {
        throw std::runtime_error("Instant Meshes produced an empty mesh");
    }
    return triangulate_extracted_mesh(O_extr, F_extr);
}

} // namespace photara_mesh

#endif
