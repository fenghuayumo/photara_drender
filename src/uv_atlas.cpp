#include "asdiff_render/uv_atlas.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
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
    if (options.width == 0 || options.height == 0 || options.parallel_partitions == 0 ||
        options.gutter < 0.0F ||
        options.max_stretch < 0.0F || options.max_stretch > 1.0F) {
        throw std::invalid_argument("UV atlas dimensions, gutter, or max_stretch are invalid");
    }
    if (options.parallel_partitions > 1 && options.max_chart_count != 0) {
        throw std::invalid_argument("max_chart_count is not supported with parallel UVAtlas partitions");
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

struct EdgeUse {
    std::uint32_t face;
    std::uint32_t edge;
    bool matched = false;
};

std::vector<std::uint32_t> build_adjacency(std::span<const std::uint32_t> indices) {
    constexpr std::uint32_t UNUSED = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> adjacency(indices.size(), UNUSED);
    std::unordered_map<EdgeKey, EdgeUse, EdgeHash> edge_uses;
    edge_uses.reserve(indices.size() / 2);
    const auto face_count = static_cast<std::uint32_t>(indices.size() / 3);
    for (std::uint32_t face = 0; face < face_count; ++face) {
        for (std::uint32_t edge = 0; edge < 3; ++edge) {
            const auto a = indices[face * 3 + edge];
            const auto b = indices[face * 3 + (edge + 1) % 3];
            if (a == b) {
                throw std::invalid_argument("triangle mesh contains a degenerate edge");
            }
            const EdgeKey key{std::min(a, b), std::max(a, b)};
            const auto iterator = edge_uses.find(key);
            if (iterator == edge_uses.end()) {
                edge_uses.emplace(key, EdgeUse{face, edge});
            } else {
                if (iterator->second.matched) {
                    throw std::invalid_argument("triangle mesh contains a non-manifold edge");
                }
                adjacency[face * 3 + edge] = iterator->second.face;
                adjacency[iterator->second.face * 3 + iterator->second.edge] = face;
                iterator->second.matched = true;
            }
        }
    }
    return adjacency;
}

using FacePartition = std::vector<std::uint32_t>;

std::array<double, 3> face_centroid(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    std::uint32_t face) {
    std::array<double, 3> centroid{};
    for (std::uint32_t corner = 0; corner < 3; ++corner) {
        const auto vertex = indices[face * 3 + corner];
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            centroid[axis] += static_cast<double>(positions[vertex * 3 + axis]) / 3.0;
        }
    }
    return centroid;
}

std::pair<FacePartition, FacePartition> split_partition_pca(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    const FacePartition& faces) {
    std::array<double, 3> mean{};
    for (const auto face : faces) {
        const auto point = face_centroid(positions, indices, face);
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            mean[axis] += point[axis];
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(faces.size());
    }

    std::array<std::array<double, 3>, 3> covariance{};
    for (const auto face : faces) {
        const auto point = face_centroid(positions, indices, face);
        std::array<double, 3> delta{};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            delta[axis] = point[axis] - mean[axis];
        }
        for (std::uint32_t row = 0; row < 3; ++row) {
            for (std::uint32_t column = 0; column < 3; ++column) {
                covariance[row][column] += delta[row] * delta[column];
            }
        }
    }

    std::uint32_t initial_axis = 0;
    if (covariance[1][1] > covariance[initial_axis][initial_axis]) {
        initial_axis = 1;
    }
    if (covariance[2][2] > covariance[initial_axis][initial_axis]) {
        initial_axis = 2;
    }
    std::array<double, 3> axis{};
    axis[initial_axis] = 1.0;
    for (std::uint32_t iteration = 0; iteration < 16; ++iteration) {
        std::array<double, 3> next{};
        for (std::uint32_t row = 0; row < 3; ++row) {
            for (std::uint32_t column = 0; column < 3; ++column) {
                next[row] += covariance[row][column] * axis[column];
            }
        }
        const double length = std::sqrt(next[0] * next[0] + next[1] * next[1] + next[2] * next[2]);
        if (length <= std::numeric_limits<double>::epsilon()) {
            break;
        }
        for (std::uint32_t component = 0; component < 3; ++component) {
            axis[component] = next[component] / length;
        }
    }

    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    std::vector<double> projections;
    projections.reserve(faces.size());
    for (const auto face : faces) {
        const auto point = face_centroid(positions, indices, face);
        double projection = 0.0;
        for (std::uint32_t component = 0; component < 3; ++component) {
            projection += axis[component] * (point[component] - mean[component]);
        }
        projections.push_back(projection);
        minimum = std::min(minimum, projection);
        maximum = std::max(maximum, projection);
    }

    const double center = 0.5 * (minimum + maximum);
    FacePartition first;
    FacePartition second;
    first.reserve(faces.size() / 2);
    second.reserve(faces.size() / 2);
    for (std::size_t index = 0; index < faces.size(); ++index) {
        (projections[index] < center ? first : second).push_back(faces[index]);
    }
    if (first.empty() || second.empty()) {
        const auto middle = faces.begin() + static_cast<std::ptrdiff_t>(faces.size() / 2);
        first.assign(faces.begin(), middle);
        second.assign(middle, faces.end());
    }
    return {std::move(first), std::move(second)};
}

std::vector<FacePartition> partition_faces_pca(
    std::span<const float> positions,
    std::span<const std::uint32_t> indices,
    std::uint32_t requested_partitions) {
    FacePartition all_faces(indices.size() / 3);
    std::iota(all_faces.begin(), all_faces.end(), 0U);
    if (requested_partitions <= 1 || all_faces.size() <= 1) {
        return {std::move(all_faces)};
    }

    const std::size_t vertex_count = positions.size() / 3;
    const std::size_t max_faces = std::max<std::size_t>(
        1, (vertex_count - 1) / (static_cast<std::size_t>(requested_partitions) - 1));
    std::vector<FacePartition> pending;
    std::vector<FacePartition> result;
    pending.push_back(std::move(all_faces));
    while (!pending.empty()) {
        std::vector<FacePartition> next;
        for (auto& partition : pending) {
            if (partition.size() <= max_faces) {
                result.push_back(std::move(partition));
            } else {
                auto [first, second] = split_partition_pca(positions, indices, partition);
                next.push_back(std::move(first));
                next.push_back(std::move(second));
            }
        }
        pending = std::move(next);
    }
    return result;
}

std::vector<std::uint32_t> decode_indices(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("UVAtlas returned an unexpected index format");
    }
    std::vector<std::uint32_t> indices(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(indices.data(), bytes.data(), bytes.size());
    return indices;
}

std::vector<std::uint8_t> encode_indices(const std::vector<std::uint32_t>& indices) {
    std::vector<std::uint8_t> bytes(indices.size() * sizeof(std::uint32_t));
    std::memcpy(bytes.data(), indices.data(), bytes.size());
    return bytes;
}

DirectX::UVATLAS atlas_flags(const std::optional<bool>& quality, bool force_fast) {
    if (quality.has_value()) {
        return *quality ? DirectX::UVATLAS_GEODESIC_QUALITY : DirectX::UVATLAS_GEODESIC_FAST;
    }
    return force_fast ? DirectX::UVATLAS_GEODESIC_FAST : DirectX::UVATLAS_DEFAULT;
}

struct PartitionOutput {
    std::vector<DirectX::UVAtlasVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> adjacency;
    std::vector<std::uint32_t> original_faces;
    std::vector<std::uint32_t> face_charts;
    std::vector<std::uint32_t> vertex_remap;
    float max_stretch = 0.0F;
    std::size_t chart_count = 0;
};

PartitionOutput unwrap_partition(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const FacePartition& faces,
    const UvAtlasOptions& options,
    bool force_fast) {
    constexpr std::uint32_t UNUSED = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> global_to_local(positions.size() / 3, UNUSED);
    std::vector<std::uint32_t> local_to_global;
    std::vector<std::uint32_t> local_indices;
    local_indices.reserve(faces.size() * 3);
    for (const auto face : faces) {
        for (std::uint32_t corner = 0; corner < 3; ++corner) {
            const auto global_vertex = triangle_indices[face * 3 + corner];
            auto& local_vertex = global_to_local[global_vertex];
            if (local_vertex == UNUSED) {
                local_vertex = static_cast<std::uint32_t>(local_to_global.size());
                local_to_global.push_back(global_vertex);
            }
            local_indices.push_back(local_vertex);
        }
    }

    std::vector<DirectX::XMFLOAT3> local_positions(local_to_global.size());
    for (std::size_t local_vertex = 0; local_vertex < local_to_global.size(); ++local_vertex) {
        const auto global_vertex = local_to_global[local_vertex];
        local_positions[local_vertex] = {
            positions[global_vertex * 3],
            positions[global_vertex * 3 + 1],
            positions[global_vertex * 3 + 2],
        };
    }

    PartitionOutput output;
    output.original_faces = faces;
    const auto adjacency = build_adjacency(local_indices);
    std::vector<std::uint8_t> index_bytes;
    std::vector<std::uint32_t> local_vertex_remap;
    const HRESULT result = DirectX::UVAtlasPartition(
        local_positions.data(),
        local_positions.size(),
        local_indices.data(),
        DXGI_FORMAT_R32_UINT,
        faces.size(),
        0,
        options.max_stretch,
        adjacency.data(),
        nullptr,
        nullptr,
        {},
        DirectX::UVATLAS_DEFAULT_CALLBACK_FREQUENCY,
        atlas_flags(options.quality, force_fast),
        output.vertices,
        index_bytes,
        &output.face_charts,
        &local_vertex_remap,
        output.adjacency,
        &output.max_stretch,
        &output.chart_count);
    if (FAILED(result)) {
        throw std::runtime_error("UVAtlasPartition failed with HRESULT " + std::to_string(result));
    }
    output.indices = decode_indices(index_bytes);
    if (output.indices.size() != faces.size() * 3 ||
        output.adjacency.size() != faces.size() * 3 ||
        output.face_charts.size() != faces.size() ||
        local_vertex_remap.size() != output.vertices.size()) {
        throw std::runtime_error("UVAtlasPartition returned inconsistent output sizes");
    }
    output.vertex_remap.resize(local_vertex_remap.size());
    for (std::size_t vertex = 0; vertex < local_vertex_remap.size(); ++vertex) {
        output.vertex_remap[vertex] = local_to_global.at(local_vertex_remap[vertex]);
    }
    return output;
}

UvAtlasOutput unwrap_uv_parallel(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices,
    const UvAtlasOptions& options) {
    auto partitions = partition_faces_pca(positions, triangle_indices, options.parallel_partitions);
    std::vector<PartitionOutput> partition_outputs(partitions.size());
    const std::size_t default_workers = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(
        partitions.size(), options.worker_count == 0 ? default_workers : options.worker_count);
    std::atomic_size_t next_partition = 0;
    std::exception_ptr worker_error;
    std::mutex error_mutex;
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            try {
                while (true) {
                    const auto partition = next_partition.fetch_add(1);
                    if (partition >= partitions.size()) {
                        break;
                    }
                    partition_outputs[partition] =
                        unwrap_partition(positions, triangle_indices, partitions[partition], options, true);
                }
            } catch (...) {
                std::scoped_lock lock(error_mutex);
                if (!worker_error) {
                    worker_error = std::current_exception();
                }
                next_partition.store(partitions.size());
            }
        });
    }
    workers.clear();
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    std::vector<DirectX::UVAtlasVertex> combined_vertices;
    std::vector<std::uint32_t> combined_indices;
    std::vector<std::uint32_t> combined_adjacency;
    std::vector<std::uint32_t> original_faces;
    std::vector<std::uint32_t> grouped_face_charts;
    std::vector<std::uint32_t> combined_vertex_remap;
    float maximum_stretch = 0.0F;
    std::uint32_t chart_offset = 0;
    constexpr std::uint32_t UNUSED = std::numeric_limits<std::uint32_t>::max();
    for (auto& partition : partition_outputs) {
        if (combined_vertices.size() >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) -
                    partition.vertices.size() ||
            combined_indices.size() / 3 >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) -
                    partition.indices.size() / 3 ||
            partition.chart_count >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() - chart_offset)) {
            throw std::overflow_error("parallel UVAtlas output exceeds 32-bit index limits");
        }
        const auto vertex_offset = static_cast<std::uint32_t>(combined_vertices.size());
        const auto face_offset = static_cast<std::uint32_t>(combined_indices.size() / 3);
        for (auto& index : partition.indices) {
            index += vertex_offset;
        }
        for (auto& adjacent_face : partition.adjacency) {
            if (adjacent_face != UNUSED) {
                adjacent_face += face_offset;
            }
        }
        for (auto& chart : partition.face_charts) {
            chart += chart_offset;
        }
        combined_vertices.insert(
            combined_vertices.end(), partition.vertices.begin(), partition.vertices.end());
        combined_indices.insert(combined_indices.end(), partition.indices.begin(), partition.indices.end());
        combined_adjacency.insert(
            combined_adjacency.end(), partition.adjacency.begin(), partition.adjacency.end());
        original_faces.insert(
            original_faces.end(), partition.original_faces.begin(), partition.original_faces.end());
        grouped_face_charts.insert(
            grouped_face_charts.end(), partition.face_charts.begin(), partition.face_charts.end());
        combined_vertex_remap.insert(
            combined_vertex_remap.end(), partition.vertex_remap.begin(), partition.vertex_remap.end());
        maximum_stretch = std::max(maximum_stretch, partition.max_stretch);
        chart_offset += static_cast<std::uint32_t>(partition.chart_count);
    }

    auto combined_index_bytes = encode_indices(combined_indices);
    const HRESULT pack_result = DirectX::UVAtlasPack(
        combined_vertices,
        combined_index_bytes,
        DXGI_FORMAT_R32_UINT,
        options.width,
        options.height,
        options.gutter,
        combined_adjacency,
        {},
        DirectX::UVATLAS_DEFAULT_CALLBACK_FREQUENCY);
    if (FAILED(pack_result)) {
        throw std::runtime_error("UVAtlasPack failed with HRESULT " + std::to_string(pack_result));
    }
    combined_indices = decode_indices(combined_index_bytes);

    UvAtlasOutput output;
    output.positions.resize(combined_vertices.size() * 3);
    output.uv.resize(combined_vertices.size() * 2);
    if (combined_vertex_remap.size() != combined_vertices.size()) {
        throw std::runtime_error("parallel UVAtlas returned an inconsistent vertex remap");
    }
    for (std::size_t vertex = 0; vertex < combined_vertices.size(); ++vertex) {
        const auto source_vertex = combined_vertex_remap[vertex];
        output.positions[vertex * 3] = positions[source_vertex * 3];
        output.positions[vertex * 3 + 1] = positions[source_vertex * 3 + 1];
        output.positions[vertex * 3 + 2] = positions[source_vertex * 3 + 2];
        output.uv[vertex * 2] = combined_vertices[vertex].uv.x;
        output.uv[vertex * 2 + 1] = combined_vertices[vertex].uv.y;
    }
    output.indices.resize(triangle_indices.size());
    output.face_chart_ids.resize(triangle_indices.size() / 3);
    for (std::size_t grouped_face = 0; grouped_face < original_faces.size(); ++grouped_face) {
        const auto original_face = original_faces[grouped_face];
        for (std::uint32_t corner = 0; corner < 3; ++corner) {
            output.indices[original_face * 3 + corner] = combined_indices[grouped_face * 3 + corner];
        }
        output.face_chart_ids[original_face] = grouped_face_charts[grouped_face];
    }
    output.vertex_remap = std::move(combined_vertex_remap);
    output.chart_count = chart_offset;
    output.partition_count = static_cast<std::uint32_t>(partitions.size());
    output.max_stretch = maximum_stretch;
    return output;
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
    if (options.parallel_partitions > 1) {
        return unwrap_uv_parallel(positions, triangle_indices, options);
    }
    std::vector<DirectX::XMFLOAT3> directx_positions(positions.size() / 3);
    std::memcpy(directx_positions.data(), positions.data(), positions.size_bytes());
    const auto adjacency = build_adjacency(triangle_indices);
    std::vector<DirectX::UVAtlasVertex> output_vertices;
    std::vector<std::uint8_t> output_index_bytes;
    std::vector<std::uint32_t> face_partitions;
    std::vector<std::uint32_t> vertex_remap;
    float output_stretch = 0.0F;
    std::size_t output_chart_count = 0;
    const auto flags = atlas_flags(options.quality, false);
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
        flags,
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
    if (output_index_bytes.size() != triangle_indices.size() * sizeof(std::uint32_t) ||
        face_partitions.size() != triangle_indices.size() / 3 ||
        vertex_remap.size() != output_vertices.size()) {
        throw std::runtime_error("UVAtlasCreate returned inconsistent output sizes");
    }

    UvAtlasOutput output;
    output.positions.resize(output_vertices.size() * 3);
    output.uv.resize(output_vertices.size() * 2);
    for (std::size_t vertex = 0; vertex < output_vertices.size(); ++vertex) {
        const auto& source = output_vertices[vertex];
        const auto source_vertex = vertex_remap[vertex];
        if (source_vertex >= positions.size() / 3) {
            throw std::runtime_error("UVAtlasCreate returned an invalid vertex remap");
        }
        output.positions[vertex * 3 + 0] = positions[source_vertex * 3];
        output.positions[vertex * 3 + 1] = positions[source_vertex * 3 + 1];
        output.positions[vertex * 3 + 2] = positions[source_vertex * 3 + 2];
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
