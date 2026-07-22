#pragma once

#include <array>
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
    bool supports_ray_query = false;
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

enum class ProjectionBlendMode : std::uint32_t {
    best_view = 0,
    weighted_average = 1,
};

enum class VisibilityMode : std::uint32_t {
    shadow_map = 0,
    hybrid_ray_query = 1,
    ray_query = 2,
};

struct UvAtlasOptions {
    std::uint32_t width = 1024;
    std::uint32_t height = 1024;
    std::uint32_t max_chart_count = 0;
    float max_stretch = 1.0F / 6.0F;
    float gutter = 1.0F;
    std::optional<bool> quality = std::nullopt;
    std::uint32_t parallel_partitions = 1;
    std::uint32_t worker_count = 0;
};

struct UvAtlasOutput {
    std::vector<float> positions;
    std::vector<float> uv;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> vertex_remap;
    std::vector<std::uint32_t> face_chart_ids;
    std::uint32_t chart_count = 0;
    std::uint32_t partition_count = 1;
    float max_stretch = 0.0F;
};

struct ProjectionView {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channel_count = 0;
    std::vector<float> image;
    std::vector<float> visibility_mask;
    std::array<float, 16> world_to_clip{};
    std::array<float, 3> camera_position{};
    std::optional<Viewport> viewport;
    float weight = 1.0F;
    float depth_bias = 1e-3F;
    float min_view_cosine = 0.05F;
};

struct TextureBakeOptions {
    std::uint32_t width = 1024;
    std::uint32_t height = 1024;
    ProjectionBlendMode blend_mode = ProjectionBlendMode::best_view;
    // Exact ray-query visibility is the quality/default path. A shadow map is
    // allocated only when explicitly requested or when fallback is allowed on
    // a device without VK_KHR_ray_query support.
    VisibilityMode visibility_mode = VisibilityMode::ray_query;
    std::uint32_t pcf_radius = 1;
    float ray_origin_bias = 1e-4F;
    bool allow_visibility_fallback = true;
};

struct TextureBakeOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> color;
    std::vector<float> confidence;
    std::vector<std::uint32_t> source_view;
    std::vector<float> valid_mask;
    bool used_ray_query = false;
};

struct TextureRefineOptions {
    std::uint32_t width = 1024;
    std::uint32_t height = 1024;
    std::uint32_t steps = 1000;
    std::uint32_t batch_size = 4;
    float learning_rate = 5e-3F;
    float minimum_learning_rate = 2.5e-4F;
    float photometric_epsilon = 1e-3F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;
    float clamp_min = 0.0F;
    float clamp_max = 1.0F;
    std::uint32_t seam_polish_steps = 30;
    float seam_learning_rate = 1e-3F;
    float seam_epsilon = 1e-4F;
};

struct TextureRefineOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> color;
    std::vector<float> loss_history;
    std::vector<float> seam_loss_history;
    double precompute_seconds = 0.0;
    double optimization_seconds = 0.0;
};

} // namespace asdiff_render
