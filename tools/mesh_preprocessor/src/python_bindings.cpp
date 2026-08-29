#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aether_mesh/mesh_ops.hpp"
#include "aether_mesh/pipeline.hpp"

namespace py = pybind11;

namespace {

template <typename T>
std::span<const T> as_span(const py::array_t<T, py::array::c_style | py::array::forcecast>& array) {
    return {array.data(), static_cast<std::size_t>(array.size())};
}

py::array_t<float> vector_to_array(std::vector<float> values, const std::vector<py::ssize_t>& shape) {
    py::array_t<float> array(shape);
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

py::array_t<std::uint32_t> vector_to_u32_array(
    std::vector<std::uint32_t> values,
    const std::vector<py::ssize_t>& shape) {
    py::array_t<std::uint32_t> array(shape);
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(std::uint32_t));
    }
    return array;
}

void validate_positions(const py::buffer_info& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 3]");
    }
}

void validate_indices(const py::buffer_info& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("indices must have shape [triangle_count, 3]");
    }
}

aether_mesh::DecimateOptions make_decimate_options(std::size_t target_face_count, bool check_self) {
    aether_mesh::DecimateOptions options;
    options.target_face_count = target_face_count;
    options.check_self_intersections = check_self;
    return options;
}

aether_drender::UvAtlasOptions make_atlas_options(
    const std::pair<std::uint32_t, std::uint32_t>& resolution,
    float gutter,
    float max_stretch,
    const std::optional<bool>& quality,
    std::uint32_t parallel_partitions,
    std::uint32_t worker_count) {
    aether_drender::UvAtlasOptions options;
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

PYBIND11_MODULE(_aether_mesh, module) {
    module.doc() = "Optional in-memory CGAL mesh preparation and baking pipeline (GPL/commercial)";

    module.def("has_instant_meshes_backend", &aether_mesh::has_instant_meshes_backend);

    module.def(
        "remesh_field_aligned",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
           const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
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
            validate_positions(positions.request());
            validate_indices(indices.request());
            aether_mesh::RemeshOptions options;
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
            auto mesh = aether_mesh::remesh_field_aligned(as_span(positions), as_span(indices), options);
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
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
           const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
           std::size_t target_face_count,
           bool check_self_intersections) {
            validate_positions(positions.request());
            validate_indices(indices.request());
            auto mesh = aether_mesh::repair_and_decimate(
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
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
           const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
           std::size_t target_face_count,
           bool use_instant_remesh,
           const std::pair<std::uint32_t, std::uint32_t>& resolution,
           float gutter,
           float max_stretch,
           const std::optional<bool>& quality,
           std::uint32_t parallel_partitions,
           std::uint32_t worker_count) {
            validate_positions(positions.request());
            validate_indices(indices.request());
            aether_mesh::PrepareOptions options;
            options.decimate = make_decimate_options(target_face_count, false);
            options.use_instant_remesh = use_instant_remesh;
            options.atlas = make_atlas_options(
                resolution, gutter, max_stretch, quality, parallel_partitions, worker_count);
            auto prepared = aether_mesh::prepare_for_baking(as_span(positions), as_span(indices), options);
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
        py::arg("parallel_partitions") = 4,
        py::arg("worker_count") = 0);
}
