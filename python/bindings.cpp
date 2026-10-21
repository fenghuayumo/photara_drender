#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "photara_drender/photara_drender.hpp"

namespace py = nanobind;
using namespace photara_drender;

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

void validate_positions(const BufferInfo& info) {
    if (info.ndim != 2 || info.shape[1] != 4 || info.shape[0] <= 0) {
        throw std::invalid_argument("positions must have shape [vertex_count, 4]");
    }
}

void validate_indices(const BufferInfo& info) {
    if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
        throw std::invalid_argument("indices must have shape [triangle_count, 3]");
    }
}

void validate_image(const BufferInfo& info, std::uint32_t width, std::uint32_t height, const char* name) {
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

class PythonUvAtlasResult {
public:
    explicit PythonUvAtlasResult(UvAtlasOutput output) : output_(std::move(output)) {}

    FloatOutput positions() const {
        auto values = output_.positions;
        return vector_to_array(std::move(values), {static_cast<py::ssize_t>(output_.positions.size() / 3), 3});
    }

    FloatOutput uv() const {
        auto values = output_.uv;
        return vector_to_array(std::move(values), {static_cast<py::ssize_t>(output_.uv.size() / 2), 2});
    }

    U32Output indices() const {
        auto values = output_.indices;
        return vector_to_u32_array(std::move(values), {static_cast<py::ssize_t>(output_.indices.size() / 3), 3});
    }

    U32Output vertex_remap() const {
        auto values = output_.vertex_remap;
        return vector_to_u32_array(std::move(values), {static_cast<py::ssize_t>(output_.vertex_remap.size())});
    }

    U32Output face_chart_ids() const {
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
        const FloatInput& positions,
        const FloatInput& normals,
        const FloatInput& uv,
        const U32Input& indices,
        const py::list& images,
        const FloatInput& world_to_clip,
        const FloatInput& camera_positions,
        const py::object& visibility_masks,
        const py::object& viewports,
        const std::pair<std::uint32_t, std::uint32_t>& resolution,
        const std::string& blend_mode,
        const std::string& visibility_mode,
        std::uint32_t pcf_radius,
        bool allow_visibility_fallback,
        float softmax_scale,
        float mask_floor) {
        const auto position_info = request(positions);
        if (position_info.ndim != 2 || position_info.shape[1] != 3 || position_info.shape[0] <= 0) {
            throw std::invalid_argument("positions must have shape [vertex_count, 3]");
        }
        const auto normal_info = request(normals);
        const auto uv_info = request(uv);
        if (normal_info.ndim != 2 || normal_info.shape[0] != position_info.shape[0] || normal_info.shape[1] != 3) {
            throw std::invalid_argument("normals must have shape [vertex_count, 3]");
        }
        if (uv_info.ndim != 2 || uv_info.shape[0] != position_info.shape[0] || uv_info.shape[1] != 2) {
            throw std::invalid_argument("uv must have shape [vertex_count, 2]");
        }
        validate_indices(request(indices));
        const auto matrix_info = request(world_to_clip);
        const auto camera_info = request(camera_positions);
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
            const auto image = py::cast<FloatInput>(images[index]);
            const auto image_info = request(image);
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
                const auto mask = py::cast<FloatInput>(mask_list[index]);
                const auto mask_info = request(mask);
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
        } else if (blend_mode == "softmax") {
            options.blend_mode = ProjectionBlendMode::softmax;
        } else {
            throw std::invalid_argument(
                "blend_mode must be 'best_view', 'weighted_average', or 'softmax'");
        }
        options.softmax_scale = softmax_scale;
        options.mask_floor = mask_floor;
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
        auto coverage_mask = vector_to_array(
            std::move(output.coverage_mask), {static_cast<py::ssize_t>(output.height), static_cast<py::ssize_t>(output.width)});
        return py::make_tuple(
            std::move(color), std::move(confidence), std::move(source_view), std::move(valid_mask),
            std::move(coverage_mask), output.used_ray_query);
    }

    py::tuple refine_texture(
        const FloatInput& initial_texture,
        const FloatInput& positions,
        const FloatInput& uv,
        const U32Input& indices,
        const py::list& images,
        const FloatInput& world_to_clip,
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
        const auto texture_info = request(initial_texture);
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] != 3) {
            throw std::invalid_argument("initial_texture must have shape [height, width, 3]");
        }
        const auto position_info = request(positions);
        const auto uv_info = request(uv);
        if (position_info.ndim != 2 || position_info.shape[0] <= 0 || position_info.shape[1] != 3 ||
            uv_info.ndim != 2 || uv_info.shape[0] != position_info.shape[0] || uv_info.shape[1] != 2) {
            throw std::invalid_argument("positions and uv must have shapes [vertex_count, 3] and [vertex_count, 2]");
        }
        validate_indices(request(indices));
        const auto matrix_info = request(world_to_clip);
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
            const auto image = py::cast<FloatInput>(images[index]);
            const auto image_info = request(image);
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
                const auto mask = py::cast<FloatInput>(mask_list[index]);
                const auto mask_info = request(mask);
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
        FloatInput seam_array;
        std::span<const float> seam_span;
        if (!seam_uv_pairs.is_none()) {
            seam_array = py::cast<FloatInput>(seam_uv_pairs);
            const auto seam_info = request(seam_array);
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
        const FloatInput& positions,
        const U32Input& indices,
        const std::pair<std::uint32_t, std::uint32_t>& resolution,
        const std::string& cull_mode,
        bool output_barycentric_derivatives,
        const std::optional<std::array<float, 4>>& viewport) {
        validate_positions(request(positions));
        validate_indices(request(indices));
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
        FloatOutput derivatives;
        if (output_barycentric_derivatives) {
            derivatives = vector_to_array(
                std::move(output.barycentric_derivatives),
                {static_cast<py::ssize_t>(options.height), static_cast<py::ssize_t>(options.width), 4});
        } else {
            derivatives = vector_to_array(std::vector<float>{}, {0, 0, 4});
        }
        return py::make_tuple(std::move(raster), std::move(derivatives));
    }

    FloatOutput backward(
        const FloatInput& positions,
        const U32Input& indices,
        const FloatInput& raster,
        const FloatInput& grad_raster,
        const std::optional<std::array<float, 4>>& viewport) {
        const auto position_info = request(positions);
        validate_positions(position_info);
        validate_indices(request(indices));
        const auto raster_info = request(raster);
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [height, width, 4]");
        }
        const auto height = static_cast<std::uint32_t>(raster_info.shape[0]);
        const auto width = static_cast<std::uint32_t>(raster_info.shape[1]);
        validate_image(request(grad_raster), width, height, "grad_raster");
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

    FloatOutput interpolate_forward(
        const FloatInput& vertex_attributes,
        const U32Input& indices,
        const FloatInput& raster) {
        const auto attribute_info = request(vertex_attributes);
        if (attribute_info.ndim != 2 || attribute_info.shape[0] <= 0 || attribute_info.shape[1] <= 0) {
            throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
        }
        validate_indices(request(indices));
        const auto raster_info = request(raster);
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
        const FloatInput& vertex_attributes,
        const U32Input& indices,
        const FloatInput& raster,
        const FloatInput& grad_interpolated) {
        const auto attribute_info = request(vertex_attributes);
        if (attribute_info.ndim != 2 || attribute_info.shape[0] <= 0 || attribute_info.shape[1] <= 0) {
            throw std::invalid_argument("vertex_attributes must have shape [vertex_count, attribute_count]");
        }
        validate_indices(request(indices));
        const auto raster_info = request(raster);
        if (raster_info.ndim != 3 || raster_info.shape[2] != 4) {
            throw std::invalid_argument("raster must have shape [height, width, 4]");
        }
        const auto gradient_info = request(grad_interpolated);
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

    FloatOutput texture_forward(
        const FloatInput& texture,
        const FloatInput& uv,
        const FloatInput& raster,
        const std::string& address_mode) {
        const auto texture_info = request(texture);
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] <= 0) {
            throw std::invalid_argument("texture must have shape [texture_height, texture_width, channel_count]");
        }
        const auto uv_info = request(uv);
        const auto raster_info = request(raster);
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
        const FloatInput& texture,
        const FloatInput& uv,
        const FloatInput& raster,
        const FloatInput& grad_sampled,
        const std::string& address_mode,
        bool compute_uv_gradient) {
        const auto texture_info = request(texture);
        if (texture_info.ndim != 3 || texture_info.shape[0] <= 0 || texture_info.shape[1] <= 0 ||
            texture_info.shape[2] <= 0) {
            throw std::invalid_argument("texture must have shape [texture_height, texture_width, channel_count]");
        }
        const auto uv_info = request(uv);
        const auto raster_info = request(raster);
        const auto gradient_info = request(grad_sampled);
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

NB_MODULE(_photara_drender, module) {
    module.doc() = "Portable differentiable rasterization powered by Vulkan";

    py::class_<DeviceInfo>(module, "DeviceInfo")
        .def_ro("name", &DeviceInfo::name)
        .def_ro("vendor_id", &DeviceInfo::vendor_id)
        .def_ro("device_id", &DeviceInfo::device_id)
        .def_ro("api_version", &DeviceInfo::api_version)
        .def_ro("supports_ray_query", &DeviceInfo::supports_ray_query)
        .def("__repr__", [](const DeviceInfo& info) {
            return "DeviceInfo(name='" + info.name + "', vendor_id=" + std::to_string(info.vendor_id) + ")";
        });

    py::class_<PythonRasterizer>(module, "Rasterizer")
        .def(py::init<std::uint32_t, bool>(), py::arg("device_index") = 0, py::arg("enable_validation") = false)
        .def_prop_ro("device_info", &PythonRasterizer::device_info, py::rv_policy::reference_internal)
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
        .def_prop_ro("positions", &PythonUvAtlasResult::positions, py::rv_policy::move)
        .def_prop_ro("uv", &PythonUvAtlasResult::uv, py::rv_policy::move)
        .def_prop_ro("indices", &PythonUvAtlasResult::indices, py::rv_policy::move)
        .def_prop_ro("vertex_remap", &PythonUvAtlasResult::vertex_remap, py::rv_policy::move)
        .def_prop_ro("face_chart_ids", &PythonUvAtlasResult::face_chart_ids, py::rv_policy::move)
        .def_prop_ro("chart_count", &PythonUvAtlasResult::chart_count)
        .def_prop_ro("partition_count", &PythonUvAtlasResult::partition_count)
        .def_prop_ro("max_stretch", &PythonUvAtlasResult::max_stretch);

    module.def("has_uv_atlas_backend", &has_uv_atlas_backend);
    module.def(
        "unwrap_uv",
        [](const FloatInput& positions,
           const U32Input& indices,
           const std::pair<std::uint32_t, std::uint32_t>& resolution,
           float gutter,
           float max_stretch,
           std::uint32_t max_chart_count,
           const std::optional<bool>& quality,
           std::uint32_t parallel_partitions,
           std::uint32_t worker_count) {
            const auto info = request(positions);
            if (info.ndim != 2 || info.shape[1] != 3 || info.shape[0] <= 0) {
                throw std::invalid_argument("positions must have shape [vertex_count, 3]");
            }
            validate_indices(request(indices));
            UvAtlasOptions options;
            options.height = resolution.first;
            options.width = resolution.second;
            options.gutter = gutter;
            options.max_stretch = max_stretch;
            options.max_chart_count = max_chart_count;
            options.quality = quality;
            options.parallel_partitions = parallel_partitions;
            options.worker_count = worker_count;
            return PythonUvAtlasResult(photara_drender::unwrap_uv(as_span(positions), as_span(indices), options));
        },
        py::arg("positions"), py::arg("indices"), py::arg("resolution") = std::pair{1024U, 1024U},
        py::arg("gutter") = 1.0F, py::arg("max_stretch") = 1.0F / 6.0F,
        py::arg("max_chart_count") = 0, py::arg("quality") = py::none(),
        py::arg("parallel_partitions") = 0, py::arg("worker_count") = 0,
        "Unwrap a mesh into a UV atlas. parallel_partitions=0 picks the partition count "
        "automatically (four PCA partitions on large meshes, one UVAtlasCreate call for "
        "small ones); 1 forces a single chart set and N>1 forces N partitions.");

    py::class_<PythonTextureBaker>(module, "TextureBaker")
        .def(py::init<std::uint32_t, bool>(), py::arg("device_index") = 0, py::arg("enable_validation") = false)
        .def_prop_ro("device_info", &PythonTextureBaker::device_info, py::rv_policy::reference_internal)
        .def(
            "bake",
            &PythonTextureBaker::bake,
            py::arg("positions"), py::arg("normals"), py::arg("uv"), py::arg("indices"),
            py::arg("images"), py::arg("world_to_clip"), py::arg("camera_positions"),
            py::arg("visibility_masks") = py::none(), py::arg("viewports") = py::none(),
            py::arg("resolution") = std::pair{1024U, 1024U},
            py::arg("blend_mode") = "weighted_average", py::arg("visibility_mode") = "ray_query",
            py::arg("pcf_radius") = 1, py::arg("allow_visibility_fallback") = true,
            py::arg("softmax_scale") = 0.0F, py::arg("mask_floor") = 0.0F)
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

    module.def(
        "pad_texture_atlas",
        [](const FloatInput& color,
           const FloatInput& valid_mask,
           std::uint32_t margin,
           const py::object& coverage_mask,
           bool fill_unobserved,
           std::uint32_t maximum_fill_distance) {
            const auto color_info = request(color);
            const auto valid_info = request(valid_mask);
            if (color_info.ndim != 3 || color_info.shape[0] <= 0 || color_info.shape[1] <= 0 ||
                (color_info.shape[2] != 3 && color_info.shape[2] != 4)) {
                throw std::invalid_argument("color must have shape [height, width, 3 or 4]");
            }
            if (valid_info.ndim != 2 || valid_info.shape[0] != color_info.shape[0] ||
                valid_info.shape[1] != color_info.shape[1]) {
                throw std::invalid_argument("valid_mask must match the color dimensions");
            }
            TextureBakeOutput output;
            output.width = static_cast<std::uint32_t>(color_info.shape[1]);
            output.height = static_cast<std::uint32_t>(color_info.shape[0]);
            output.color.assign(color.data(), color.data() + color.size());
            output.valid_mask.assign(valid_mask.data(), valid_mask.data() + valid_mask.size());
            if (!coverage_mask.is_none()) {
                const auto coverage = py::cast<FloatInput>(
                    coverage_mask);
                const auto coverage_info = request(coverage);
                if (coverage_info.ndim != 2 || coverage_info.shape[0] != color_info.shape[0] ||
                    coverage_info.shape[1] != color_info.shape[1]) {
                    throw std::invalid_argument("coverage_mask must match the color dimensions");
                }
                output.coverage_mask.assign(coverage.data(), coverage.data() + coverage.size());
            }
            TexturePaddingOptions options;
            options.margin = margin;
            options.fill_unobserved = fill_unobserved;
            options.maximum_fill_distance = maximum_fill_distance;
            const auto stats = pad_texture_atlas(output, options);
            const std::vector<py::ssize_t> color_shape{
                color_info.shape[0], color_info.shape[1], color_info.shape[2]};
            const std::vector<py::ssize_t> mask_shape{color_info.shape[0], color_info.shape[1]};
            return py::make_tuple(
                vector_to_array(std::move(output.color), color_shape),
                vector_to_array(std::move(output.filled_mask), mask_shape),
                stats.gutter_texels,
                stats.unobserved_texels,
                stats.remaining_unobserved);
        },
        py::arg("color"), py::arg("valid_mask"), py::arg("margin") = 4,
        py::arg("coverage_mask") = py::none(), py::arg("fill_unobserved") = true,
        py::arg("maximum_fill_distance") = 0);
}
