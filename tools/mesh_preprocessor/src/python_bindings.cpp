#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "photara_mesh/mesh_ops.hpp"
#include "photara_mesh/pipeline.hpp"

namespace py = nanobind;

namespace {

using FloatInput = py::ndarray<const float, py::numpy, py::c_contig>;
using U32Input = py::ndarray<const std::uint32_t, py::numpy, py::c_contig>;
using FloatOutput = py::ndarray<float, py::numpy>;
using U32Output = py::ndarray<std::uint32_t, py::numpy>;

struct BufferInfo {
    std::size_t ndim;
    std::vector<py::ssize_t> shape;
};

template <typename... Args>
BufferInfo request(const py::ndarray<Args...>& array) {
    BufferInfo info{array.ndim(), {}};
    info.shape.reserve(info.ndim);
    for (std::size_t i = 0; i < info.ndim; ++i) {
        info.shape.push_back(static_cast<py::ssize_t>(array.shape(i)));
    }
    return info;
}

template <typename T>
std::span<const T> as_span(const py::ndarray<const T, py::numpy, py::c_contig>& array) {
    return {array.data(), static_cast<std::size_t>(array.size())};
}

template <typename T, typename Array>
Array vector_to_array(std::vector<T>&& values, const std::vector<py::ssize_t>& shape) {
    auto* data = new std::vector<T>(std::move(values));
    py::capsule owner(data, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    std::vector<std::size_t> dimensions;
    dimensions.reserve(shape.size());
    for (const auto dimension : shape) {
        dimensions.push_back(static_cast<std::size_t>(dimension));
    }
    return Array(data->data(), dimensions.size(), dimensions.data(), owner);
}

FloatOutput vector_to_array(std::vector<float>&& values, const std::vector<py::ssize_t>& shape) {
    return vector_to_array<float, FloatOutput>(std::move(values), shape);
}

U32Output vector_to_u32_array(
    std::vector<std::uint32_t>&& values,
    const std::vector<py::ssize_t>& shape) {
    return vector_to_array<std::uint32_t, U32Output>(std::move(values), shape);
}

void validate_positions(const BufferInfo& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
}

void validate_indices(const BufferInfo& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("indices must have shape [triangle_count, 3]");
    }
}

photara_mesh::DecimateOptions make_decimate_options(std::size_t target_face_count, bool check_self) {
    photara_mesh::DecimateOptions options;
    options.target_face_count = target_face_count;
    options.check_self_intersections = check_self;
    return options;
}

photara_drender::UvAtlasOptions make_atlas_options(
    const std::pair<std::uint32_t, std::uint32_t>& resolution,
    float gutter,
    float max_stretch,
    const std::optional<bool>& quality,
    std::uint32_t parallel_partitions,
    std::uint32_t worker_count) {
    photara_drender::UvAtlasOptions options;
    options.height = resolution.first;
    options.width = resolution.second;
    options.gutter = gutter;
    options.max_stretch = max_stretch;
    options.quality = quality;
    options.parallel_partitions = parallel_partitions;
    options.worker_count = worker_count;
    return options;
}

} // namespace

NB_MODULE(_photara_mesh, module) {
    module.doc() = "Optional in-memory CGAL mesh preparation and baking pipeline (GPL/commercial)";

    module.def("has_instant_meshes_backend", &photara_mesh::has_instant_meshes_backend);

    module.def(
        "remesh_field_aligned",
        [](const FloatInput& positions,
           const U32Input& indices,
           int vertex_count,
           int face_count,
           float scale,
           int rosy,
           int posy,
           float crease_angle,
           bool align_to_boundaries,
           bool extrinsic,
           int smooth_iterations,
           bool deterministic) {
            validate_positions(request(positions));
            validate_indices(request(indices));
            photara_mesh::RemeshOptions options;
            options.vertex_count = vertex_count;
            options.face_count = face_count;
            options.scale = scale;
            options.rosy = rosy;
            options.posy = posy;
            options.crease_angle = crease_angle;
            options.align_to_boundaries = align_to_boundaries;
            options.extrinsic = extrinsic;
            options.smooth_iterations = smooth_iterations;
            options.deterministic = deterministic;
            auto mesh = photara_mesh::remesh_field_aligned(as_span(positions), as_span(indices), options);
            return py::make_tuple(
                vector_to_array(std::move(mesh.positions),
                                {static_cast<py::ssize_t>(mesh.positions.size() / 3), 3}),
                vector_to_u32_array(std::move(mesh.indices),
                                    {static_cast<py::ssize_t>(mesh.indices.size() / 3), 3}));
        },
        py::arg("positions"),
        py::arg("indices"),
        py::arg("vertex_count") = -1,
        py::arg("face_count") = -1,
        py::arg("scale") = -1.0F,
        py::arg("rosy") = 4,
        py::arg("posy") = 4,
        py::arg("crease_angle") = 0.0F,
        py::arg("align_to_boundaries") = true,
        py::arg("extrinsic") = true,
        py::arg("smooth_iterations") = 2,
        py::arg("deterministic") = true);

    module.def(
        "repair_and_decimate",
        [](const FloatInput& positions,
           const U32Input& indices,
           std::size_t target_face_count,
           bool check_self_intersections) {
            validate_positions(request(positions));
            validate_indices(request(indices));
            auto mesh = photara_mesh::repair_and_decimate(
                as_span(positions),
                as_span(indices),
                make_decimate_options(target_face_count, check_self_intersections));
            return py::make_tuple(
                vector_to_array(std::move(mesh.positions),
                                {static_cast<py::ssize_t>(mesh.positions.size() / 3), 3}),
                vector_to_u32_array(std::move(mesh.indices),
                                    {static_cast<py::ssize_t>(mesh.indices.size() / 3), 3}));
        },
        py::arg("positions"),
        py::arg("indices"),
        py::arg("target_face_count") = 1'000'000,
        py::arg("check_self_intersections") = false);

    module.def(
        "prepare_for_baking",
        [](const FloatInput& positions,
           const U32Input& indices,
           std::size_t target_face_count,
           bool use_instant_remesh,
           const std::pair<std::uint32_t, std::uint32_t>& resolution,
           float gutter,
           float max_stretch,
           const std::optional<bool>& quality,
           std::uint32_t parallel_partitions,
           std::uint32_t worker_count) {
            validate_positions(request(positions));
            validate_indices(request(indices));
            photara_mesh::PrepareOptions options;
            options.decimate = make_decimate_options(target_face_count, false);
            options.use_instant_remesh = use_instant_remesh;
            options.atlas = make_atlas_options(
                resolution, gutter, max_stretch, quality, parallel_partitions, worker_count);
            auto prepared = photara_mesh::prepare_for_baking(as_span(positions), as_span(indices), options);
            return py::make_tuple(
                vector_to_array(std::move(prepared.positions),
                                {static_cast<py::ssize_t>(prepared.positions.size() / 3), 3}),
                vector_to_array(std::move(prepared.normals),
                                {static_cast<py::ssize_t>(prepared.normals.size() / 3), 3}),
                vector_to_array(std::move(prepared.uv),
                                {static_cast<py::ssize_t>(prepared.uv.size() / 2), 2}),
                vector_to_u32_array(std::move(prepared.indices),
                                    {static_cast<py::ssize_t>(prepared.indices.size() / 3), 3}),
                vector_to_u32_array(std::move(prepared.vertex_remap),
                                    {static_cast<py::ssize_t>(prepared.vertex_remap.size())}),
                vector_to_u32_array(std::move(prepared.face_chart_ids),
                                    {static_cast<py::ssize_t>(prepared.face_chart_ids.size())}),
                prepared.chart_count,
                prepared.max_stretch,
                prepared.partition_count,
                static_cast<std::uint32_t>(prepared.remeshed.positions.size() / 3),
                static_cast<std::uint32_t>(prepared.remeshed.indices.size() / 3),
                static_cast<std::uint32_t>(prepared.manifold.positions.size() / 3),
                static_cast<std::uint32_t>(prepared.manifold.indices.size() / 3));
        },
        py::arg("positions"),
        py::arg("indices"),
        py::arg("target_face_count") = 1'000'000,
        py::arg("use_instant_remesh") = false,
        py::arg("resolution") = std::pair{4096U, 4096U},
        py::arg("gutter") = 1.0F,
        py::arg("max_stretch") = 1.0F / 6.0F,
        py::arg("quality") = py::none(),
        py::arg("parallel_partitions") = 0,
        py::arg("worker_count") = 0);
}
