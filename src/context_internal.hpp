#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "photara_drender/context.hpp"

namespace photara_drender {

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;

    Buffer() = default;
    Buffer(
        VkPhysicalDevice physical_device,
        VkDevice logical_device,
        VkDeviceSize byte_size,
        VkBufferUsageFlags usage,
        bool prefer_device_local = false);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(const void* source, std::size_t byte_size, std::size_t offset = 0);
    void download(void* destination, std::size_t byte_size, std::size_t offset = 0) const;
    [[nodiscard]] VkDeviceAddress device_address() const;
};

struct AccelerationStructure {
    VkDevice device = VK_NULL_HANDLE;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR destroy_function = nullptr;
    Buffer storage;

    AccelerationStructure() = default;
    ~AccelerationStructure();
    AccelerationStructure(AccelerationStructure&& other) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&& other) noexcept;
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
};

struct RayQueryScene {
    Buffer vertices;
    Buffer indices;
    AccelerationStructure bottom_level;
    AccelerationStructure top_level;
};

struct ComputePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
    std::uint32_t binding_count = 0;

    ComputePipeline() = default;
    ComputePipeline(
        VkDevice logical_device,
        std::span<const std::byte> spir_v_bytes,
        std::uint32_t storage_buffer_count,
        std::uint32_t push_constant_size);
    ComputePipeline(
        VkDevice logical_device,
        std::span<const std::byte> spir_v_bytes,
        std::span<const VkDescriptorType> descriptor_types,
        std::uint32_t push_constant_size);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
};

class Context::Impl {
public:
    explicit Impl(const ContextOptions& options);
    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    [[nodiscard]] Buffer create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        bool prefer_device_local = false) const;
    [[nodiscard]] ComputePipeline create_pipeline(
        const std::string& shader_name,
        std::uint32_t storage_buffer_count,
        std::uint32_t push_constant_size) const;
    [[nodiscard]] ComputePipeline create_pipeline(
        const std::string& shader_name,
        std::span<const VkDescriptorType> descriptor_types,
        std::uint32_t push_constant_size) const;
    [[nodiscard]] RayQueryScene create_ray_query_scene(
        std::span<const float> positions,
        std::span<const std::uint32_t> triangle_indices) const;
    void dispatch(
        const ComputePipeline& pipeline,
        std::span<const VkDescriptorBufferInfo> buffers,
        const void* push_constants,
        std::uint32_t push_constant_size,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y = 1,
        std::uint32_t group_count_z = 1) const;
    void dispatch_ray_query(
        const ComputePipeline& pipeline,
        const Buffer& positions,
        const Buffer& normals,
        const Buffer& raster,
        const AccelerationStructure& scene,
        const Buffer& visibility,
        const void* push_constants,
        std::uint32_t push_constant_size,
        std::uint32_t group_count_x) const;

    DeviceInfo device_info;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family_index = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    mutable std::mutex dispatch_mutex;
};

} // namespace photara_drender
