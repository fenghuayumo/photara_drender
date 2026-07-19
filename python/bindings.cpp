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
        const std::string& address_mode) {
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
            as_span(texture), texture_desc, as_span(uv), raster_output, as_span(grad_sampled));
        auto grad_texture = vector_to_array(
            std::move(gradients.texture),
            {texture_info.shape[0], texture_info.shape[1], texture_info.shape[2]});
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
            py::arg("address_mode") = "clamp");

    module.def("enumerate_devices", &Context::enumerate_devices);
}
