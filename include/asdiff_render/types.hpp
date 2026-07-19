#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace asdiff_render {

enum class CullMode : std::uint32_t {
    none = 0,
    back = 1,
    front = 2,
};

enum class AddressMode : std::uint32_t {
    clamp = 0,
    wrap = 1,
    mirror = 2,
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

struct Viewport {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct RasterizeOptions {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    CullMode cull_mode = CullMode::none;
    bool output_barycentric_derivatives = true;
    std::optional<Viewport> viewport;
};

struct RasterizeOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> raster;
    std::vector<float> barycentric_derivatives;
    Viewport viewport;
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

struct TextureDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channel_count = 0;
    AddressMode address_mode = AddressMode::clamp;
};

struct TextureOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channel_count = 0;
    std::vector<float> values;
};

struct TextureGradients {
    std::vector<float> texture;
    std::vector<float> uv;
};

} // namespace asdiff_render
