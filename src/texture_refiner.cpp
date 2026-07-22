#include "asdiff_render/texture_refiner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "asdiff_render/rasterizer.hpp"
#include "context_internal.hpp"

namespace asdiff_render {
namespace {

constexpr std::uint32_t BLOCK_SIZE = 64;

std::uint32_t divide_round_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

VkDescriptorBufferInfo descriptor(const Buffer& buffer) {
    return {buffer.handle, 0, buffer.size};
}

void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
    }
}

std::vector<float> transform_positions(
    std::span<const float> positions,
    const std::array<float, 16>& matrix) {
    std::vector<float> result(positions.size() / 3 * 4);
    for (std::size_t vertex = 0; vertex < positions.size() / 3; ++vertex) {
        const float x = positions[vertex * 3];
        const float y = positions[vertex * 3 + 1];
        const float z = positions[vertex * 3 + 2];
        for (std::uint32_t row = 0; row < 4; ++row) {
            result[vertex * 4 + row] = matrix[row * 4] * x + matrix[row * 4 + 1] * y +
                                       matrix[row * 4 + 2] * z + matrix[row * 4 + 3];
        }
    }
    return result;
}

struct GradientPushConstants {
    std::uint32_t pixel_count;
    std::uint32_t texture_width;
    std::uint32_t texture_height;
    std::uint32_t loss_index;
    float photometric_epsilon;
    float normalization;
};

static_assert(sizeof(GradientPushConstants) == 24);

struct AdamPushConstants {
    std::uint32_t value_count;
    float learning_rate;
    float beta1;
    float beta2;
    float adam_epsilon;
    float bias_correction1;
    float bias_correction2;
    float clamp_min;
    float clamp_max;
};

static_assert(sizeof(AdamPushConstants) == 36);

using SeamPushConstants = GradientPushConstants;

struct CachedView {
    Buffer uv;
    Buffer target;
    Buffer mask;
    std::uint32_t pixel_count = 0;
    double valid_weight = 0.0;
};

VkDescriptorSet allocate_descriptor_set(Context::Impl& context, const ComputePipeline& pipeline) {
    VkDescriptorSetAllocateInfo allocate_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = context.descriptor_pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &pipeline.descriptor_set_layout;
    VkDescriptorSet result = VK_NULL_HANDLE;
    check_vk(vkAllocateDescriptorSets(context.device, &allocate_info, &result), "vkAllocateDescriptorSets");
    return result;
}

void update_descriptor_set(
    Context::Impl& context,
    VkDescriptorSet descriptor_set,
    std::span<const VkDescriptorBufferInfo> buffers) {
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::uint32_t index = 0; index < buffers.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &buffers[index];
    }
    vkUpdateDescriptorSets(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void shader_memory_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &barrier,
        0,
        nullptr,
        0,
        nullptr);
}

void transfer_to_shader_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void shader_to_transfer_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace

class TextureRefiner::Impl {
public:
    Impl(Context::Impl& context, Context& public_context)
        : context_(context), rasterizer_(public_context),
          gradient_pipeline_(context.create_pipeline(
              "atlas_photometric_gradient.hlsl.spv", 6, sizeof(GradientPushConstants))),
          seam_pipeline_(context.create_pipeline(
              "atlas_seam_gradient.hlsl.spv", 4, sizeof(SeamPushConstants))),
          adam_pipeline_(context.create_pipeline(
              "atlas_adam_update.hlsl.spv", 4, sizeof(AdamPushConstants))) {}

    Context::Impl& context_;
    Rasterizer rasterizer_;
    ComputePipeline gradient_pipeline_;
    ComputePipeline seam_pipeline_;
    ComputePipeline adam_pipeline_;
};

TextureRefiner::TextureRefiner(Context& context) : impl_(std::make_unique<Impl>(*context.impl_, context)) {}
TextureRefiner::~TextureRefiner() = default;
TextureRefiner::TextureRefiner(TextureRefiner&&) noexcept = default;
TextureRefiner& TextureRefiner::operator=(TextureRefiner&&) noexcept = default;

TextureRefineOutput TextureRefiner::refine(
    std::span<const float> initial_texture,
    std::span<const float> positions,
    std::span<const float> uv,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const ProjectionView> views,
    std::span<const float> seam_uv_pairs,
    const TextureRefineOptions& options) {
    const auto vertex_count = positions.size() / 3;
    if (positions.empty() || positions.size() % 3 != 0 || uv.size() != vertex_count * 2 ||
        triangle_indices.empty() || triangle_indices.size() % 3 != 0 ||
        std::ranges::any_of(triangle_indices, [vertex_count](std::uint32_t value) { return value >= vertex_count; })) {
        throw std::invalid_argument("texture refinement mesh arrays have inconsistent shapes");
    }
    const auto texel_count = static_cast<std::size_t>(options.width) * options.height;
    const auto texture_value_count = texel_count * 3;
    if (options.width == 0 || options.height == 0 || initial_texture.size() != texture_value_count || views.empty() ||
        options.steps == 0 || options.batch_size == 0 || !std::isfinite(options.learning_rate) ||
        options.learning_rate <= 0.0F || !std::isfinite(options.minimum_learning_rate) ||
        options.minimum_learning_rate < 0.0F || options.minimum_learning_rate > options.learning_rate ||
        !std::isfinite(options.photometric_epsilon) || options.photometric_epsilon <= 0.0F ||
        !(options.adam_beta1 > 0.0F && options.adam_beta1 < 1.0F) ||
        !(options.adam_beta2 > 0.0F && options.adam_beta2 < 1.0F) ||
        !(options.clamp_min < options.clamp_max) || seam_uv_pairs.size() % 4 != 0 ||
        !std::isfinite(options.seam_learning_rate) || options.seam_learning_rate <= 0.0F ||
        !std::isfinite(options.seam_epsilon) || options.seam_epsilon <= 0.0F) {
        throw std::invalid_argument("texture refinement options are invalid");
    }
    if (texture_value_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("texture refinement atlas exceeds the shader element limit");
    }

    const auto precompute_start = std::chrono::steady_clock::now();
    std::vector<CachedView> cached_views;
    cached_views.reserve(views.size());
    for (const auto& view : views) {
        if (view.width == 0 || view.height == 0 || (view.channel_count != 3 && view.channel_count != 4)) {
            throw std::invalid_argument("texture refinement images must contain RGB or RGBA data");
        }
        const auto pixel_count = static_cast<std::size_t>(view.width) * view.height;
        if (view.image.size() != pixel_count * view.channel_count ||
            (!view.visibility_mask.empty() && view.visibility_mask.size() != pixel_count) ||
            pixel_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("texture refinement image or mask has an invalid shape");
        }
        const auto clip_positions = transform_positions(positions, view.world_to_clip);
        RasterizeOptions raster_options;
        raster_options.width = view.width;
        raster_options.height = view.height;
        raster_options.output_barycentric_derivatives = false;
        raster_options.viewport = view.viewport;
        const auto raster = impl_->rasterizer_.forward(clip_positions, triangle_indices, raster_options);
        const auto interpolated_uv = impl_->rasterizer_.interpolate_forward(uv, 2, triangle_indices, raster);

        std::vector<float> target(pixel_count * 3);
        std::vector<float> mask(pixel_count, 0.0F);
        double valid_weight = 0.0;
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            target[pixel * 3] = view.image[pixel * view.channel_count];
            target[pixel * 3 + 1] = view.image[pixel * view.channel_count + 1];
            target[pixel * 3 + 2] = view.image[pixel * view.channel_count + 2];
            if (raster.raster[pixel * 4 + 3] != 0.0F) {
                const float external = view.visibility_mask.empty()
                    ? 1.0F
                    : std::clamp(view.visibility_mask[pixel], 0.0F, 1.0F);
                mask[pixel] = external;
                valid_weight += external;
            }
        }
        CachedView cached;
        cached.pixel_count = static_cast<std::uint32_t>(pixel_count);
        cached.valid_weight = valid_weight;
        cached.uv = impl_->context_.create_buffer(interpolated_uv.values.size() * sizeof(float), 0, true);
        cached.target = impl_->context_.create_buffer(target.size() * sizeof(float), 0, true);
        cached.mask = impl_->context_.create_buffer(mask.size() * sizeof(float), 0, true);
        cached.uv.upload(interpolated_uv.values.data(), interpolated_uv.values.size() * sizeof(float));
        cached.target.upload(target.data(), target.size() * sizeof(float));
        cached.mask.upload(mask.data(), mask.size() * sizeof(float));
        cached_views.push_back(std::move(cached));
    }
    const auto precompute_end = std::chrono::steady_clock::now();

    auto texture_buffer = impl_->context_.create_buffer(initial_texture.size_bytes(), 0, true);
    const auto optimizer_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto gradient_buffer = impl_->context_.create_buffer(initial_texture.size_bytes(), optimizer_usage, true);
    auto first_moment_buffer = impl_->context_.create_buffer(initial_texture.size_bytes(), optimizer_usage, true);
    auto second_moment_buffer = impl_->context_.create_buffer(initial_texture.size_bytes(), optimizer_usage, true);
    const auto seam_pair_count = static_cast<std::uint32_t>(seam_uv_pairs.size() / 4);
    const auto actual_seam_steps = seam_pair_count == 0 ? 0U : options.seam_polish_steps;
    const auto total_loss_count = options.steps + actual_seam_steps;
    auto loss_buffer = impl_->context_.create_buffer(total_loss_count * sizeof(float), 0, true);
    auto seam_buffer = impl_->context_.create_buffer(std::max<std::size_t>(seam_uv_pairs.size_bytes(), 4), 0, true);
    texture_buffer.upload(initial_texture.data(), initial_texture.size_bytes());
    std::memset(gradient_buffer.mapped, 0, static_cast<std::size_t>(gradient_buffer.size));
    std::memset(first_moment_buffer.mapped, 0, static_cast<std::size_t>(first_moment_buffer.size));
    std::memset(second_moment_buffer.mapped, 0, static_cast<std::size_t>(second_moment_buffer.size));
    std::memset(loss_buffer.mapped, 0, static_cast<std::size_t>(loss_buffer.size));
    if (!seam_uv_pairs.empty()) {
        seam_buffer.upload(seam_uv_pairs.data(), seam_uv_pairs.size_bytes());
    }

    const std::scoped_lock lock(impl_->context_.dispatch_mutex);
    std::vector<VkDescriptorSet> gradient_sets;
    gradient_sets.reserve(cached_views.size());
    for (const auto& view : cached_views) {
        const auto descriptor_set = allocate_descriptor_set(impl_->context_, impl_->gradient_pipeline_);
        const std::array buffers{
            descriptor(texture_buffer), descriptor(view.uv), descriptor(view.target), descriptor(view.mask),
            descriptor(gradient_buffer), descriptor(loss_buffer),
        };
        update_descriptor_set(impl_->context_, descriptor_set, buffers);
        gradient_sets.push_back(descriptor_set);
    }
    const auto adam_set = allocate_descriptor_set(impl_->context_, impl_->adam_pipeline_);
    const std::array adam_buffers{
        descriptor(texture_buffer), descriptor(gradient_buffer), descriptor(first_moment_buffer),
        descriptor(second_moment_buffer),
    };
    update_descriptor_set(impl_->context_, adam_set, adam_buffers);
    VkDescriptorSet seam_set = VK_NULL_HANDLE;
    if (actual_seam_steps != 0) {
        seam_set = allocate_descriptor_set(impl_->context_, impl_->seam_pipeline_);
        const std::array seam_buffers{
            descriptor(texture_buffer), descriptor(seam_buffer), descriptor(gradient_buffer), descriptor(loss_buffer),
        };
        update_descriptor_set(impl_->context_, seam_set, seam_buffers);
    }

    VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = impl_->context_.command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(impl_->context_.device, &command_buffer_info, &command_buffer),
             "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

    const float pi = std::acos(-1.0F);
    for (std::uint32_t step = 0; step < options.steps; ++step) {
        double batch_valid_weight = 0.0;
        for (std::uint32_t batch = 0; batch < options.batch_size; ++batch) {
            const auto view_index = (static_cast<std::size_t>(step) * options.batch_size + batch) % views.size();
            batch_valid_weight += cached_views[view_index].valid_weight;
        }
        const float normalization = batch_valid_weight > 0.0
            ? static_cast<float>(1.0 / (batch_valid_weight * 3.0))
            : 0.0F;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->gradient_pipeline_.handle);
        for (std::uint32_t batch = 0; batch < options.batch_size; ++batch) {
            const auto view_index = (static_cast<std::size_t>(step) * options.batch_size + batch) % views.size();
            const auto& view = cached_views[view_index];
            vkCmdBindDescriptorSets(
                command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->gradient_pipeline_.pipeline_layout,
                0, 1, &gradient_sets[view_index], 0, nullptr);
            const GradientPushConstants push{
                view.pixel_count, options.width, options.height, step,
                options.photometric_epsilon, normalization,
            };
            vkCmdPushConstants(
                command_buffer, impl_->gradient_pipeline_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, divide_round_up(view.pixel_count, BLOCK_SIZE), 1, 1);
        }
        shader_memory_barrier(command_buffer);

        const float phase = static_cast<float>(step) / static_cast<float>(options.steps);
        const float learning_rate = options.minimum_learning_rate +
            0.5F * (options.learning_rate - options.minimum_learning_rate) * (1.0F + std::cos(pi * phase));
        const AdamPushConstants adam_push{
            static_cast<std::uint32_t>(texture_value_count),
            learning_rate,
            options.adam_beta1,
            options.adam_beta2,
            options.adam_epsilon,
            1.0F - std::pow(options.adam_beta1, static_cast<float>(step + 1)),
            1.0F - std::pow(options.adam_beta2, static_cast<float>(step + 1)),
            options.clamp_min,
            options.clamp_max,
        };
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->adam_pipeline_.handle);
        vkCmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->adam_pipeline_.pipeline_layout,
            0, 1, &adam_set, 0, nullptr);
        vkCmdPushConstants(
            command_buffer, impl_->adam_pipeline_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(adam_push), &adam_push);
        vkCmdDispatch(
            command_buffer,
            divide_round_up(static_cast<std::uint32_t>(texture_value_count), BLOCK_SIZE),
            1,
            1);
        shader_memory_barrier(command_buffer);
    }
    if (actual_seam_steps != 0) {
        shader_to_transfer_barrier(command_buffer);
        vkCmdFillBuffer(command_buffer, gradient_buffer.handle, 0, gradient_buffer.size, 0);
        vkCmdFillBuffer(command_buffer, first_moment_buffer.handle, 0, first_moment_buffer.size, 0);
        vkCmdFillBuffer(command_buffer, second_moment_buffer.handle, 0, second_moment_buffer.size, 0);
        transfer_to_shader_barrier(command_buffer);
        for (std::uint32_t seam_step = 0; seam_step < actual_seam_steps; ++seam_step) {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->seam_pipeline_.handle);
            vkCmdBindDescriptorSets(
                command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->seam_pipeline_.pipeline_layout,
                0, 1, &seam_set, 0, nullptr);
            const SeamPushConstants seam_push{
                seam_pair_count, options.width, options.height, options.steps + seam_step,
                options.seam_epsilon, 1.0F / static_cast<float>(seam_pair_count * 3),
            };
            vkCmdPushConstants(
                command_buffer, impl_->seam_pipeline_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(seam_push), &seam_push);
            vkCmdDispatch(command_buffer, divide_round_up(seam_pair_count, BLOCK_SIZE), 1, 1);
            shader_memory_barrier(command_buffer);

            const AdamPushConstants adam_push{
                static_cast<std::uint32_t>(texture_value_count),
                options.seam_learning_rate,
                options.adam_beta1,
                options.adam_beta2,
                options.adam_epsilon,
                1.0F - std::pow(options.adam_beta1, static_cast<float>(seam_step + 1)),
                1.0F - std::pow(options.adam_beta2, static_cast<float>(seam_step + 1)),
                options.clamp_min,
                options.clamp_max,
            };
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->adam_pipeline_.handle);
            vkCmdBindDescriptorSets(
                command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->adam_pipeline_.pipeline_layout,
                0, 1, &adam_set, 0, nullptr);
            vkCmdPushConstants(
                command_buffer, impl_->adam_pipeline_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(adam_push), &adam_push);
            vkCmdDispatch(
                command_buffer,
                divide_round_up(static_cast<std::uint32_t>(texture_value_count), BLOCK_SIZE), 1, 1);
            shader_memory_barrier(command_buffer);
        }
    }
    check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    const auto optimization_start = std::chrono::steady_clock::now();
    check_vk(vkQueueSubmit(impl_->context_.queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
    check_vk(vkQueueWaitIdle(impl_->context_.queue), "vkQueueWaitIdle");
    const auto optimization_end = std::chrono::steady_clock::now();

    TextureRefineOutput output;
    output.width = options.width;
    output.height = options.height;
    output.color.resize(texture_value_count);
    output.loss_history.resize(options.steps);
    output.seam_loss_history.resize(actual_seam_steps);
    output.precompute_seconds = std::chrono::duration<double>(precompute_end - precompute_start).count();
    output.optimization_seconds = std::chrono::duration<double>(optimization_end - optimization_start).count();
    texture_buffer.download(output.color.data(), output.color.size() * sizeof(float));
    loss_buffer.download(output.loss_history.data(), output.loss_history.size() * sizeof(float));
    if (actual_seam_steps != 0) {
        loss_buffer.download(
            output.seam_loss_history.data(), output.seam_loss_history.size() * sizeof(float),
            options.steps * sizeof(float));
    }
    vkFreeCommandBuffers(impl_->context_.device, impl_->context_.command_pool, 1, &command_buffer);
    gradient_sets.push_back(adam_set);
    if (seam_set != VK_NULL_HANDLE) {
        gradient_sets.push_back(seam_set);
    }
    vkFreeDescriptorSets(
        impl_->context_.device,
        impl_->context_.descriptor_pool,
        static_cast<std::uint32_t>(gradient_sets.size()),
        gradient_sets.data());
    return output;
}

} // namespace asdiff_render
