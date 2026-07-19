#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace asdiff_render {

enum class CullMode : std::uint32_t {
    none = 0,
    back = 1,
    front = 2,
};

struct ContextOptions {
    std::uint32_t device_index = 0;
    bool enable_validation = false;
};

struct DeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
};

struct RasterizeOptions {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    CullMode cull_mode = CullMode::none;
    bool output_barycentric_derivatives = true;
};

struct RasterizeOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> raster;
    std::vector<float> barycentric_derivatives;
};

struct InterpolateOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t attribute_count = 0;
    std::vector<float> values;
};

struct InterpolateGradients {
    std::vector<float> attributes;
    std::vector<float> raster;
};

} // namespace asdiff_render
