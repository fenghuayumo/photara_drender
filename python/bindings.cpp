#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "asdiff_render/asdiff_render.hpp"

namespace py = pybind11;
using namespace asdiff_render;

namespace {

template <typename T>
std::span<const T> as_span(const py::array_t<T, py::array::c_style | py::array::forcecast>& array) {
    return {array.data(), static_cast<std::size_t>(array.size())};
}

void validate_positions(const py::buffer_info& info) {
    if (info.ndim != 2 || info.shape[1] != 4 || info.shape[0] <= 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 4]");
    }
}

void validate_indices(const py::buffer_info& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("indices must have shape [triangle_count, 3]");
    }
}

void validate_image(const py::buffer_info& info, std::uint32_t width, std::uint32_t height, const char* name) {
    if (info.ndim != 3 || info.shape[0] != height || info.shape[1] != width || info.shape[2] != 4) {
        throw std::invalid_argument(std::string(name) + " must have shape [height, width, 4]");
    }
}

CullMode parse_cull_mode(const std::string& value) {
    if (value == "none") {
        return CullMode::none;
    }
    if (value == "back") {
        return CullMode::back;
    }
    if (value == "front") {
        return CullMode::front;
    }
    throw std::invalid_argument("cull_mode must be 'none', 'back', or 'front'");
}

AddressMode parse_address_mode(const std::string& value) {
    if (value == "clamp") {
        return AddressMode::clamp;
    }
    if (value == "wrap") {
        return AddressMode::wrap;
    }
    if (value == "mirror") {
        return AddressMode::mirror;
    }
    throw std::invalid_argument("address_mode must be 'clamp', 'wrap', or 'mirror'");
}

py::array_t<float> vector_to_array(std::vector<float>&& values, const std::vector<py::ssize_t>& shape) {
    py::array_t<float> array(shape);
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

py::array_t<std::uint32_t> vector_to_u32_array(
    std::vector<std::uint32_t>&& values,
    const std::vector<py::ssize_t>& shape) {
    py::array_t<std::uint32_t> array(shape);
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(std::uint32_t));
    }
    return array;
}

class PythonUvAtlasResult {
public:
    explicit PythonUvAtlasResult(UvAtlasOutput output) : output_(std::move(output)) {}

    py::array_t<float> positions() const {
        auto values = output_.positions;
        return vector_to_array(std::move(values), {static_cast<py::ssize_t>(output_.positions.size() / 3), 3});
    }

    py::array_t<float> uv() const {
        auto values = output_.uv;
        return vector_to_array(std::move(values), {static_cast<py::ssize_t>(output_.uv.size() / 2), 2});
    }

    py::array_t<std::uint32_t> indices() const {
        auto values = output_.indices;
        return vector_to_u32_array(std::move(values), {static_cast<py::ssize_t>(output_.indices.size() / 3), 3});
    }

    py::array_t<std::uint32_t> vertex_remap() const {
        auto values = output_.vertex_remap;
        return vector_to_u32_array(std::move(values), {static_cast<py::ssize_t>(output_.vertex_remap.size())});
    }

    py::array_t<std::uint32_t> face_chart_ids() const {
        auto values = output_.face_chart_ids;
        return vector_to_u32_array(std::move(values), {static_cast<py::ssize_t>(output_.face_chart_ids.size())});
    }

    std::uint32_t chart_count() const noexcept { return output_.chart_count; }
    std::uint32_t partition_count() const noexcept { return output_.partition_count; }
    float max_stretch() const noexcept { return output_.max_stretch; }

private:
    UvAtlasOutput output_;
};

class PythonTextureBaker {
public:
    PythonTextureBaker(std::uint32_t device_index, bool enable_validation)
        : context_(ContextOptions{device_index, enable_validation}), baker_(context_), refiner_(context_) {}

    py::tuple bake(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& normals,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& uv,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const py::list& images,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& world_to_clip,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& camera_positions,
        const py::object& visibility_masks,
        const py::object& viewports,
        const std::pair<std::uint32_t, std::uint32_t>& resolution,
        const std::string& blend_mode,
        const std::string& visibility_mode,
        std::uint32_t pcf_radius,
        bool allow_visibility_fallback) {
        const auto position_info = positions.request();
        if (position_info.ndim != 2 || position_info.shape[1] != 3 || position_info.shape[0] <= 0) {
            throw std::invalid_argument("positions must have shape [vertex_count, 3]");
        }
        const auto normal_info = normals.request();
        const auto uv_info = uv.request();
        if (normal_info.ndim != 2 || normal_info.shape[0] != position_info.shape[0] || normal_info.shape[1] != 3) {
            throw std::invalid_argument("normals must have shape [vertex_count, 3]");
        }
        if (uv_info.ndim != 2 || uv_info.shape[0] != position_info.shape[0] || uv_info.shape[1] != 2) {
            throw std::invalid_argument("uv must have shape [vertex_count, 2]");
        }
        validate_indices(indices.request());
        const auto matrix_info = world_to_clip.request();
        const auto camera_info = camera_positions.request();
        const auto view_count = static_cast<py::ssize_t>(images.size());
        if (view_count <= 0 || matrix_info.ndim != 3 || matrix_info.shape[0] != view_count ||
            matrix_info.shape[1] != 4 || matrix_info.shape[2] != 4) {
            throw std::invalid_argument("world_to_clip must have shape [view_count, 4, 4]");
        }
        if (camera_info.ndim != 2 || camera_info.shape[0] != view_count || camera_info.shape[1] != 3) {
            throw std::invalid_argument("camera_positions must have shape [view_count, 3]");
        }
        py::list mask_list;
        if (!visibility_masks.is_none()) {
            mask_list = py::cast<py::list>(visibility_masks);
            if (mask_list.size() != view_count) {
                throw std::invalid_argument("visibility_masks must contain one entry per view");
            }
        }
        py::list viewport_list;
        if (!viewports.is_none()) {
            viewport_list = py::cast<py::list>(viewports);
            if (viewport_list.size() != view_count) {
                throw std::invalid_argument("viewports must contain one entry per view");
            }
        }

        std::vector<ProjectionView> projection_views(static_cast<std::size_t>(view_count));
        for (py::ssize_t index = 0; index < view_count; ++index) {
            const auto image = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(images[index]);
            const auto image_info = image.request();
            if (image_info.ndim != 3 || image_info.shape[0] <= 0 || image_info.shape[1] <= 0 ||
                (image_info.shape[2] != 3 && image_info.shape[2] != 4)) {
                throw std::invalid_argument("each image must have shape [height, width, 3 or 4]");
            }
            auto& view = projection_views[static_cast<std::size_t>(index)];
            view.height = static_cast<std::uint32_t>(image_info.shape[0]);
            view.width = static_cast<std::uint32_t>(image_info.shape[1]);
            view.channel_count = static_cast<std::uint32_t>(image_info.shape[2]);
            view.image.assign(image.data(), image.data() + image.size());
            std::copy_n(world_to_clip.data() + index * 16, 16, view.world_to_clip.begin());
            std::copy_n(camera_positions.data() + index * 3, 3, view.camera_position.begin());
            if (!visibility_masks.is_none() && !mask_list[index].is_none()) {
                const auto mask = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(mask_list[index]);
                const auto mask_info = mask.request();
                if (mask_info.ndim != 2 || mask_info.shape[0] != image_info.shape[0] ||
                    mask_info.shape[1] != image_info.shape[1]) {
                    throw std::invalid_argument("each visibility mask must have shape [image_height, image_width]");
                }
                view.visibility_mask.assign(mask.data(), mask.data() + mask.size());
            }
            if (!viewports.is_none() && !viewport_list[index].is_none()) {
                const auto value = py::cast<std::array<float, 4>>(viewport_list[index]);
                view.viewport = Viewport{value[0], value[1], value[2], value[3]};
            }
        }
        TextureBakeOptions options;
        options.height = resolution.first;
        options.width = resolution.second;
        options.pcf_radius = pcf_radius;
        options.allow_visibility_fallback = allow_visibility_fallback;
        if (blend_mode == "best_view") {
            options.blend_mode = ProjectionBlendMode::best_view;
        } else if (blend_mode == "weighted_average") {
            options.blend_mode = ProjectionBlendMode::weighted_average;
        } else {
            throw std::invalid_argument("blend_mode must be 'best_view' or 'weighted_average'");
        }
        if (visibility_mode == "shadow_map") {
            options.visibility_mode = VisibilityMode::shadow_map;
        } else if (visibility_mode == "hybrid_ray_query") {
            options.visibility_mode = VisibilityMode::hybrid_ray_query;
        } else if (visibility_mode == "ray_query") {
            options.visibility_mode = VisibilityMode::ray_query;
        } else {
            throw std::invalid_argument(
                "visibility_mode must be 'shadow_map', 'hybrid_ray_query', or 'ray_query'");
        }
        auto output = baker_.bake(
            as_span(positions), as_span(normals), as_span(uv), as_span(indices), projection_views, options);
        auto color = vector_to_array(
            std::move(output.color), {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width), 4});
        auto confidence = vector_to_array(
            std::move(output.confidence), {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width)});
        auto source_view = vector_to_u32_array(
            std::move(output.source_view), {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width)});
        auto valid_mask = vector_to_array(
            std::move(output.valid_mask), {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width)});
        return py::make_tuple(
            std::move(color), std::move(confidence), std::move(source_view), std::move(valid_mask), output.used_ray_query);
    }

    py::tuple refine_texture(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& initial_texture,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& uv,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const py::list& images,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& world_to_clip,
        const py::object& visibility_masks,
        const py::object& viewports,
        const py::object& seam_uv_pairs,
        std::uint32_t steps,
        std::uint32_t batch_size,
        float learning_rate,
        float minimum_learning_rate,
        float photometric_epsilon,
        std::uint32_t seam_polish_steps,
        float seam_learning_rate) {
        const auto texture_info = initial_texture.request();
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] != 3) {
            throw std::invalid_argument("initial_texture must have shape [height, width, 3]");
        }
        const auto position_info = positions.request();
        const auto uv_info = uv.request();
        if (position_info.ndim != 2 || position_info.shape[0] <= 0 || position_info.shape[1] != 3 ||
            uv_info.ndim != 2 || uv_info.shape[0] != position_info.shape[0] || uv_info.shape[1] != 2) {
            throw std::invalid_argument("positions and uv must have shapes [vertex_count, 3] and [vertex_count, 2]");
        }
        validate_indices(indices.request());
        const auto matrix_info = world_to_clip.request();
        const auto view_count = static_cast<py::ssize_t>(images.size());
        if (view_count <= 0 || matrix_info.ndim != 3 || matrix_info.shape[0] != view_count ||
            matrix_info.shape[1] != 4 || matrix_info.shape[2] != 4) {
            throw std::invalid_argument("world_to_clip must have shape [view_count, 4, 4]");
        }
        py::list mask_list;
        if (!visibility_masks.is_none()) {
            mask_list = py::cast<py::list>(visibility_masks);
            if (mask_list.size() != view_count) {
                throw std::invalid_argument("visibility_masks must contain one entry per view");
            }
        }
        py::list viewport_list;
        if (!viewports.is_none()) {
            viewport_list = py::cast<py::list>(viewports);
            if (viewport_list.size() != view_count) {
                throw std::invalid_argument("viewports must contain one entry per view");
            }
        }
        std::vector<ProjectionView> projection_views(static_cast<std::size_t>(view_count));
        for (py::ssize_t index = 0; index < view_count; ++index) {
            const auto image = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(images[index]);
            const auto image_info = image.request();
            if (image_info.ndim != 3 || image_info.shape[0] <= 0 || image_info.shape[1] <= 0 ||
                (image_info.shape[2] != 3 && image_info.shape[2] != 4)) {
                throw std::invalid_argument("each image must have shape [height, width, 3 or 4]");
            }
            auto& view = projection_views[static_cast<std::size_t>(index)];
            view.height = static_cast<std::uint32_t>(image_info.shape[0]);
            view.width = static_cast<std::uint32_t>(image_info.shape[1]);
            view.channel_count = static_cast<std::uint32_t>(image_info.shape[2]);
            view.image.assign(image.data(), image.data() + image.size());
            std::copy_n(world_to_clip.data() + index * 16, 16, view.world_to_clip.begin());
            if (!visibility_masks.is_none() && !mask_list[index].is_none()) {
                const auto mask = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(mask_list[index]);
                const auto mask_info = mask.request();
                if (mask_info.ndim != 2 || mask_info.shape[0] != image_info.shape[0] ||
                    mask_info.shape[1] != image_info.shape[1]) {
                    throw std::invalid_argument("each visibility mask must match its image dimensions");
                }
                view.visibility_mask.assign(mask.data(), mask.data() + mask.size());
            }
            if (!viewports.is_none() && !viewport_list[index].is_none()) {
                const auto value = py::cast<std::array<float, 4>>(viewport_list[index]);
                view.viewport = Viewport{value[0], value[1], value[2], value[3]};
            }
        }
        TextureRefineOptions options;
        options.height = static_cast<std::uint32_t>(texture_info.shape[0]);
        options.width = static_cast<std::uint32_t>(texture_info.shape[1]);
        options.steps = steps;
        options.batch_size = batch_size;
        options.learning_rate = learning_rate;
        options.minimum_learning_rate = minimum_learning_rate;
        options.photometric_epsilon = photometric_epsilon;
        options.seam_polish_steps = seam_polish_steps;
        options.seam_learning_rate = seam_learning_rate;
        py::array_t<float, py::array::c_style | py::array::forcecast> seam_array;
        std::span<const float> seam_span;
        if (!seam_uv_pairs.is_none()) {
            seam_array = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(seam_uv_pairs);
            const auto seam_info = seam_array.request();
            if (seam_info.ndim != 3 || seam_info.shape[1] != 2 || seam_info.shape[2] != 2) {
                throw std::invalid_argument("seam_uv_pairs must have shape [pair_count, 2, 2]");
            }
            seam_span = as_span(seam_array);
        }
        TextureRefineOutput output;
        {
            py::gil_scoped_release release;
            output = refiner_.refine(
                as_span(initial_texture), as_span(positions), as_span(uv), as_span(indices),
                projection_views, seam_span, options);
        }
        auto color = vector_to_array(
            std::move(output.color),
            {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width), 3});
        auto history = vector_to_array(
            std::move(output.loss_history), {static_cast<py::ssize_t>(options.steps)});
        const auto seam_history_size = static_cast<py::ssize_t>(output.seam_loss_history.size());
        auto seam_history = vector_to_array(
            std::move(output.seam_loss_history), {seam_history_size});
        return py::make_tuple(
            std::move(color), std::move(history), std::move(seam_history),
            output.precompute_seconds, output.optimization_seconds);
    }

    [[nodiscard]] const DeviceInfo& device_info() const noexcept { return context_.device_info(); }

private:
    Context context_;
    TextureBaker baker_;
    TextureRefiner refiner_;
};

class PythonRasterizer {
public:
    PythonRasterizer(std::uint32_t device_index, bool enable_validation)
        : context_(ContextOptions{device_index, enable_validation}), rasterizer_(context_) {}

    py::tuple forward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const std::pair<std::uint32_t, std::uint32_t>& resolution,
        const std::string& cull_mode,
        bool output_barycentric_derivatives,
        const std::optional<std::array<float, 4>>& viewport) {
        validate_positions(positions.request());
        validate_indices(indices.request());
        RasterizeOptions options;
        options.height = resolution.first;
        options.width = resolution.second;
        options.cull_mode = parse_cull_mode(cull_mode);
        options.output_barycentric_derivatives = output_barycentric_derivatives;
        if (viewport) {
            options.viewport = Viewport{(*viewport)[0], (*viewport)[1], (*viewport)[2], (*viewport)[3]};
        }
        auto output = rasterizer_.forward(as_span(positions), as_span(indices), options);
        auto raster = vector_to_array(
            std::move(output.raster),
            {static_cast<py::ssize_t>(options.height), static_cast<py::ssize_t>(options.width), 4});
        py::array_t<float> derivatives;
        if (output_barycentric_derivatives) {
            derivatives = vector_to_array(
                std::move(output.barycentric_derivatives),
                {static_cast<py::ssize_t>(options.height), static_cast<py::ssize_t>(options.width), 4});
        } else {
            derivatives = py::array_t<float>({0, 0, 4});
        }
        return py::make_tuple(std::move(raster), std::move(derivatives));
    }

    py::array_t<float> backward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& raster,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& grad_raster,
        const std::optional<std::array<float, 4>>& viewport) {
        const auto position_info = positions.request();
        validate_positions(position_info);
        validate_indices(indices.request());
        const auto raster_info = raster.request();
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [height, width, 4]");
        }
        const auto height = static_cast<std::uint32_t>(raster_info.shape[0]);
        const auto width = static_cast<std::uint32_t>(raster_info.shape[1]);
        validate_image(grad_raster.request(), width, height, "grad_raster");
        RasterizeOutput forward_output;
        forward_output.width = width;
        forward_output.height = height;
        forward_output.raster.assign(raster.data(), raster.data() + raster.size());
        if (viewport) {
            forward_output.viewport = Viewport{(*viewport)[0], (*viewport)[1], (*viewport)[2], (*viewport)[3]};
        }
        auto gradient = rasterizer_.backward(
            as_span(positions),
            as_span(indices),
            forward_output,
            as_span(grad_raster));
        return vector_to_array(
            std::move(gradient),
            {static_cast<py::ssize_t>(position_info.shape[0]), 4});
    }

    py::array_t<float> interpolate_forward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& vertex_attributes,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& raster) {
        const auto attribute_info = vertex_attributes.request();
        if (attribute_info.ndim != 2 || attribute_info.shape[0] <= 0 || attribute_info.shape[1] <= 0) {
            throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
        }
        validate_indices(indices.request());
        const auto raster_info = raster.request();
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [height, width, 4]");
        }
        RasterizeOutput raster_output;
        raster_output.height = static_cast<std::uint32_t>(raster_info.shape[0]);
        raster_output.width = static_cast<std::uint32_t>(raster_info.shape[1]);
        raster_output.raster.assign(raster.data(), raster.data() + raster.size());
        auto output = rasterizer_.interpolate_forward(
            as_span(vertex_attributes),
            static_cast<std::uint32_t>(attribute_info.shape[1]),
            as_span(indices),
            raster_output);
        return vector_to_array(
            std::move(output.values),
            {
                static_cast<py::ssize_t>(output.height),
                static_cast<py::ssize_t>(output.width),
                static_cast<py::ssize_t>(output.attribute_count),
            });
    }

    py::tuple interpolate_backward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& vertex_attributes,
        const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& raster,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& grad_interpolated) {
        const auto attribute_info = vertex_attributes.request();
        if (attribute_info.ndim != 2 || attribute_info.shape[0] <= 0 || attribute_info.shape[1] <= 0) {
            throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
        }
        validate_indices(indices.request());
        const auto raster_info = raster.request();
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [height, width, 4]");
        }
        const auto gradient_info = grad_interpolated.request();
        if (gradient_info.ndim != 3 || gradient_info.shape[0] != raster_info.shape[0] ||
            gradient_info.shape[1] != raster_info.shape[1] || gradient_info.shape[2] != attribute_info.shape[1]) {
            throw std::invalid_argument("grad_interpolated must have shape [height, width, attribute_count]");
        }
        RasterizeOutput raster_output;
        raster_output.height = static_cast<std::uint32_t>(raster_info.shape[0]);
        raster_output.width = static_cast<std::uint32_t>(raster_info.shape[1]);
        raster_output.raster.assign(raster.data(), raster.data() + raster.size());
        auto gradients = rasterizer_.interpolate_backward(
            as_span(vertex_attributes),
            static_cast<std::uint32_t>(attribute_info.shape[1]),
            as_span(indices),
            raster_output,
            as_span(grad_interpolated));
        auto grad_attributes = vector_to_array(
            std::move(gradients.attributes),
            {attribute_info.shape[0], attribute_info.shape[1]});
        auto grad_raster = vector_to_array(
            std::move(gradients.raster),
            {raster_info.shape[0], raster_info.shape[1], 4});
        return py::make_tuple(std::move(grad_attributes), std::move(grad_raster));
    }

    py::array_t<float> texture_forward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& texture,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& uv,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& raster,
        const std::string& address_mode) {
        const auto texture_info = texture.request();
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] <= 0) {
            throw std::invalid_argument("texture must have shape [texture_height, texture_width, channel_count]");
        }
        const auto uv_info = uv.request();
        const auto raster_info = raster.request();
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [image_height, image_width, 4]");
        }
        if (uv_info.ndim != 3 || uv_info.shape[0] != raster_info.shape[0] ||
            uv_info.shape[1] != raster_info.shape[1] || uv_info.shape[2] != 2) {
            throw std::invalid_argument("uv must have shape [image_height, image_width, 2]");
        }
        RasterizeOutput raster_output;
        raster_output.height = static_cast<std::uint32_t>(raster_info.shape[0]);
        raster_output.width = static_cast<std::uint32_t>(raster_info.shape[1]);
        raster_output.raster.assign(raster.data(), raster.data() + raster.size());
        const TextureDesc texture_desc{
            static_cast<std::uint32_t>(texture_info.shape[1]),
            static_cast<std::uint32_t>(texture_info.shape[0]),
            static_cast<std::uint32_t>(texture_info.shape[2]),
            parse_address_mode(address_mode),
        };
        auto output = rasterizer_.texture_forward(as_span(texture), texture_desc, as_span(uv), raster_output);
        return vector_to_array(
            std::move(output.values),
            {
                static_cast<py::ssize_t>(output.height),
                static_cast<py::ssize_t>(output.width),
                static_cast<py::ssize_t>(output.channel_count),
            });
    }

    py::tuple texture_backward(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& texture,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& uv,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& raster,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& grad_sampled,
        const std::string& address_mode,
        bool compute_uv_gradient) {
        const auto texture_info = texture.request();
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] <= 0) {
            throw std::invalid_argument("texture must have shape [texture_height, texture_width, channel_count]");
        }
        const auto uv_info = uv.request();
        const auto raster_info = raster.request();
        const auto gradient_info = grad_sampled.request();
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4 || uv_info.ndim != 3 ||
            uv_info.shape[0] != raster_info.shape[0] || uv_info.shape[1] != raster_info.shape[1] || uv_info.shape[2] != 2) {
            throw std::invalid_argument("uv and raster dimensions are incompatible");
        }
        if (gradient_info.ndim != 3 || gradient_info.shape[0] != raster_info.shape[0] ||
            gradient_info.shape[1] != raster_info.shape[1] || gradient_info.shape[2] != texture_info.shape[2]) {
            throw std::invalid_argument("grad_sampled must have shape [image_height, image_width, channel_count]");
        }
        RasterizeOutput raster_output;
        raster_output.height = static_cast<std::uint32_t>(raster_info.shape[0]);
        raster_output.width = static_cast<std::uint32_t>(raster_info.shape[1]);
        raster_output.raster.assign(raster.data(), raster.data() + raster.size());
        const TextureDesc texture_desc{
            static_cast<std::uint32_t>(texture_info.shape[1]),
            static_cast<std::uint32_t>(texture_info.shape[0]),
            static_cast<std::uint32_t>(texture_info.shape[2]),
            parse_address_mode(address_mode),
        };
        auto gradients = rasterizer_.texture_backward(
            as_span(texture), texture_desc, as_span(uv), raster_output, as_span(grad_sampled), compute_uv_gradient);
        auto grad_texture = vector_to_array(
            std::move(gradients.texture),
            {texture_info.shape[0], texture_info.shape[1], texture_info.shape[2]});
        if (!compute_uv_gradient) {
            return py::make_tuple(std::move(grad_texture), py::none());
        }
        auto grad_uv = vector_to_array(
            std::move(gradients.uv),
            {uv_info.shape[0], uv_info.shape[1], 2});
        return py::make_tuple(std::move(grad_texture), std::move(grad_uv));
    }

    [[nodiscard]] const DeviceInfo& device_info() const noexcept {
        return context_.device_info();
    }

private:
    Context context_;
    Rasterizer rasterizer_;
};

} // namespace

PYBIND11_MODULE(_asdiff_render, module) {
    module.doc() = "Portable differentiable rasterization powered by Vulkan";

    py::class_<DeviceInfo>(module, "DeviceInfo")
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("vendor_id", &DeviceInfo::vendor_id)
        .def_readonly("device_id", &DeviceInfo::device_id)
        .def_readonly("api_version", &DeviceInfo::api_version)
        .def_readonly("supports_ray_query", &DeviceInfo::supports_ray_query)
        .def("__repr__", [](const DeviceInfo& info) {
            return "DeviceInfo(name='" + info.name + "', vendor_id=" + std::to_string(info.vendor_id) + ")";
        });

    py::class_<PythonRasterizer>(module, "Rasterizer")
        .def(py::init<std::uint32_t, bool>(), py::arg("device_index") = 0, py::arg("enable_validation") = false)
        .def_property_readonly("device_info", &PythonRasterizer::device_info, py::return_value_policy::reference_internal)
        .def(
            "forward",
            &PythonRasterizer::forward,
            py::arg("positions"),
            py::arg("indices"),
            py::arg("resolution"),
            py::arg("cull_mode") = "none",
            py::arg("output_barycentric_derivatives") = true,
            py::arg("viewport") = py::none())
        .def(
            "backward",
            &PythonRasterizer::backward,
            py::arg("positions"),
            py::arg("indices"),
            py::arg("raster"),
            py::arg("grad_raster"),
            py::arg("viewport") = py::none())
        .def(
            "interpolate_forward",
            &PythonRasterizer::interpolate_forward,
            py::arg("vertex_attributes"),
            py::arg("indices"),
            py::arg("raster"))
        .def(
            "interpolate_backward",
            &PythonRasterizer::interpolate_backward,
            py::arg("vertex_attributes"),
            py::arg("indices"),
            py::arg("raster"),
            py::arg("grad_interpolated"))
        .def(
            "texture_forward",
            &PythonRasterizer::texture_forward,
            py::arg("texture"),
            py::arg("uv"),
            py::arg("raster"),
            py::arg("address_mode") = "clamp")
        .def(
            "texture_backward",
            &PythonRasterizer::texture_backward,
            py::arg("texture"),
            py::arg("uv"),
            py::arg("raster"),
            py::arg("grad_sampled"),
            py::arg("address_mode") = "clamp",
            py::arg("compute_uv_gradient") = true);

    py::class_<PythonUvAtlasResult>(module, "UvAtlasResult")
        .def_property_readonly("positions", &PythonUvAtlasResult::positions)
        .def_property_readonly("uv", &PythonUvAtlasResult::uv)
        .def_property_readonly("indices", &PythonUvAtlasResult::indices)
        .def_property_readonly("vertex_remap", &PythonUvAtlasResult::vertex_remap)
        .def_property_readonly("face_chart_ids", &PythonUvAtlasResult::face_chart_ids)
        .def_property_readonly("chart_count", &PythonUvAtlasResult::chart_count)
        .def_property_readonly("partition_count", &PythonUvAtlasResult::partition_count)
        .def_property_readonly("max_stretch", &PythonUvAtlasResult::max_stretch);

    module.def("has_uv_atlas_backend", &has_uv_atlas_backend);
    module.def(
        "unwrap_uv",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& positions,
           const py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>& indices,
           const std::pair<std::uint32_t, std::uint32_t>& resolution,
           float gutter,
           float max_stretch,
           std::uint32_t max_chart_count,
           const std::optional<bool>& quality,
           std::uint32_t parallel_partitions,
           std::uint32_t worker_count) {
            const auto info = positions.request();
            if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
                throw std::invalid_argument("positions must have shape [vertex_count, 3]");
            }
            validate_indices(indices.request());
            UvAtlasOptions options;
            options.height = resolution.first;
            options.width = resolution.second;
            options.gutter = gutter;
            options.max_stretch = max_stretch;
            options.max_chart_count = max_chart_count;
            options.quality = quality;
            options.parallel_partitions = parallel_partitions;
            options.worker_count = worker_count;
            return PythonUvAtlasResult(asdiff_render::unwrap_uv(as_span(positions), as_span(indices), options));
        },
        py::arg("positions"), py::arg("indices"), py::arg("resolution") = std::pair{1024U, 1024U},
        py::arg("gutter") = 1.0F, py::arg("max_stretch") = 1.0F / 6.0F,
        py::arg("max_chart_count") = 0, py::arg("quality") = py::none(),
        py::arg("parallel_partitions") = 1, py::arg("worker_count") = 0);

    py::class_<PythonTextureBaker>(module, "TextureBaker")
        .def(py::init<std::uint32_t, bool>(), py::arg("device_index") = 0, py::arg("enable_validation") = false)
        .def_property_readonly("device_info", &PythonTextureBaker::device_info, py::return_value_policy::reference_internal)
        .def(
            "bake",
            &PythonTextureBaker::bake,
            py::arg("positions"), py::arg("normals"), py::arg("uv"), py::arg("indices"),
            py::arg("images"), py::arg("world_to_clip"), py::arg("camera_positions"),
            py::arg("visibility_masks") = py::none(), py::arg("viewports") = py::none(),
            py::arg("resolution") = std::pair{1024U, 1024U},
            py::arg("blend_mode") = "weighted_average", py::arg("visibility_mode") = "shadow_map",
            py::arg("pcf_radius") = 1, py::arg("allow_visibility_fallback") = true)
        .def(
            "refine_texture",
            &PythonTextureBaker::refine_texture,
            py::arg("initial_texture"), py::arg("positions"), py::arg("uv"), py::arg("indices"),
            py::arg("images"), py::arg("world_to_clip"),
            py::arg("visibility_masks") = py::none(), py::arg("viewports") = py::none(),
            py::arg("seam_uv_pairs") = py::none(),
            py::arg("steps") = 1000, py::arg("batch_size") = 4,
            py::arg("learning_rate") = 5e-3F, py::arg("minimum_learning_rate") = 2.5e-4F,
            py::arg("photometric_epsilon") = 1e-3F,
            py::arg("seam_polish_steps") = 30, py::arg("seam_learning_rate") = 1e-3F);

    module.def("enumerate_devices", &Context::enumerate_devices);
}
