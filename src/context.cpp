#include "context_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "interpolate_forward.hlsl.embedded.hpp"
#include "interpolate_grad_attributes.hlsl.embedded.hpp"
#include "interpolate_grad_raster.hlsl.embedded.hpp"
#include "rasterize_backward.hlsl.embedded.hpp"
#include "rasterize_forward.hlsl.embedded.hpp"
#include "reduce_vertex_gradients.hlsl.embedded.hpp"
#include "texture_forward.hlsl.embedded.hpp"
#include "texture_grad_texture.hlsl.embedded.hpp"
#include "texture_grad_uv.hlsl.embedded.hpp"
#include "project_texture.hlsl.embedded.hpp"
#include "project_ray_visibility.hlsl.embedded.hpp"

namespace asdiff_render {
namespace {

bool query_ray_query_support(VkPhysicalDevice physical_device);

void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
    }
}

std::uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type was found");
}

std::vector<std::byte> read_spir_v(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open shader: " + path.string());
    }
    const auto byte_size = input.tellg();
    if (byte_size <= 0 || (byte_size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V file: " + path.string());
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(byte_size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), byte_size);
    return bytes;
}

std::optional<std::filesystem::path> shader_directory_override() {
#ifdef _WIN32
    char* raw_value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&raw_value, &value_size, "ASDIFF_SHADER_DIR") != 0 || raw_value == nullptr) {
        return std::nullopt;
    }
    std::filesystem::path value(raw_value);
    std::free(raw_value);
    return value;
#else
    const char* raw_value = std::getenv("ASDIFF_SHADER_DIR");
    return raw_value == nullptr ? std::nullopt : std::optional<std::filesystem::path>(raw_value);
#endif
}

std::span<const std::byte> embedded_shader(const std::string& shader_name) {
    if (shader_name == "rasterize_forward.hlsl.spv") {
        return std::as_bytes(std::span{rasterize_forward_hlsl_spv});
    }
    if (shader_name == "rasterize_backward.hlsl.spv") {
        return std::as_bytes(std::span{rasterize_backward_hlsl_spv});
    }
    if (shader_name == "reduce_vertex_gradients.hlsl.spv") {
        return std::as_bytes(std::span{reduce_vertex_gradients_hlsl_spv});
    }
    if (shader_name == "interpolate_forward.hlsl.spv") {
        return std::as_bytes(std::span{interpolate_forward_hlsl_spv});
    }
    if (shader_name == "interpolate_grad_attributes.hlsl.spv") {
        return std::as_bytes(std::span{interpolate_grad_attributes_hlsl_spv});
    }
    if (shader_name == "interpolate_grad_raster.hlsl.spv") {
        return std::as_bytes(std::span{interpolate_grad_raster_hlsl_spv});
    }
    if (shader_name == "texture_forward.hlsl.spv") {
        return std::as_bytes(std::span{texture_forward_hlsl_spv});
    }
    if (shader_name == "texture_grad_texture.hlsl.spv") {
        return std::as_bytes(std::span{texture_grad_texture_hlsl_spv});
    }
    if (shader_name == "texture_grad_uv.hlsl.spv") {
        return std::as_bytes(std::span{texture_grad_uv_hlsl_spv});
    }
    if (shader_name == "project_texture.hlsl.spv") {
        return std::as_bytes(std::span{project_texture_hlsl_spv});
    }
    if (shader_name == "project_ray_visibility.hlsl.spv") {
        return std::as_bytes(std::span{project_ray_visibility_hlsl_spv});
    }
    throw std::invalid_argument("Unknown embedded shader: " + shader_name);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && callback_data != nullptr) {
        // A callback must not throw. The application can attach a debugger here.
    }
    return VK_FALSE;
}

DeviceInfo make_device_info(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    return {
        properties.deviceName,
        properties.vendorID,
        properties.deviceID,
        properties.apiVersion,
        query_ray_query_support(physical_device),
    };
}

std::uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if ((properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            return i;
        }
    }
    throw std::runtime_error("The selected Vulkan device has no compute queue");
}

bool has_instance_extension(const char* extension_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
             "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data()),
             "vkEnumerateInstanceExtensionProperties");
    return std::ranges::any_of(properties, [extension_name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, extension_name) == 0;
    });
}

bool has_instance_layer(const char* layer_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr), "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> properties(count);
    check_vk(vkEnumerateInstanceLayerProperties(&count, properties.data()), "vkEnumerateInstanceLayerProperties");
    return std::ranges::any_of(properties, [layer_name](const VkLayerProperties& property) {
        return std::strcmp(property.layerName, layer_name) == 0;
    });
}

bool has_device_extension(VkPhysicalDevice physical_device, const char* extension_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
             "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, properties.data()),
             "vkEnumerateDeviceExtensionProperties");
    return std::ranges::any_of(properties, [extension_name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, extension_name) == 0;
    });
}

bool query_ray_query_support(VkPhysicalDevice physical_device) {
    if (!has_device_extension(physical_device, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
        !has_device_extension(physical_device, VK_KHR_RAY_QUERY_EXTENSION_NAME) ||
        !has_device_extension(physical_device, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
        return false;
    }
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    ray_query.pNext = &acceleration_structure;
    acceleration_structure.pNext = &buffer_address;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &ray_query;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    return ray_query.rayQuery == VK_TRUE && acceleration_structure.accelerationStructure == VK_TRUE &&
           buffer_address.bufferDeviceAddress == VK_TRUE;
}

} // namespace

Buffer::Buffer(
    VkPhysicalDevice physical_device,
    VkDevice logical_device,
    VkDeviceSize byte_size,
    VkBufferUsageFlags usage)
    : device(logical_device), size(std::max<VkDeviceSize>(byte_size, 4)) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &handle), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, handle, &requirements);
    VkMemoryAllocateInfo allocation_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = find_memory_type(
        physical_device,
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateFlagsInfo allocation_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
        allocation_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocation_info.pNext = &allocation_flags;
    }
    check_vk(vkAllocateMemory(device, &allocation_info, nullptr, &memory), "vkAllocateMemory");
    check_vk(vkBindBufferMemory(device, handle, memory, 0), "vkBindBufferMemory");
    check_vk(vkMapMemory(device, memory, 0, size, 0, &mapped), "vkMapMemory");
}

Buffer::~Buffer() {
    if (device != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
        if (mapped != nullptr) {
            vkUnmapMemory(device, memory);
        }
        vkDestroyBuffer(device, handle, nullptr);
        vkFreeMemory(device, memory, nullptr);
    }
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        std::swap(device, other.device);
        std::swap(handle, other.handle);
        std::swap(memory, other.memory);
        std::swap(size, other.size);
        std::swap(mapped, other.mapped);
    }
    return *this;
}

void Buffer::upload(const void* source, std::size_t byte_size, std::size_t offset) {
    if (offset + byte_size > size) {
        throw std::out_of_range("Buffer upload exceeds allocation");
    }
    if (byte_size != 0) {
        std::memcpy(static_cast<std::byte*>(mapped) + offset, source, byte_size);
    }
}

void Buffer::download(void* destination, std::size_t byte_size, std::size_t offset) const {
    if (offset + byte_size > size) {
        throw std::out_of_range("Buffer download exceeds allocation");
    }
    if (byte_size != 0) {
        std::memcpy(destination, static_cast<const std::byte*>(mapped) + offset, byte_size);
    }
}

VkDeviceAddress Buffer::device_address() const {
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = handle;
    return vkGetBufferDeviceAddress(device, &address_info);
}

AccelerationStructure::~AccelerationStructure() {
    if (device != VK_NULL_HANDLE && handle != VK_NULL_HANDLE && destroy_function != nullptr) {
        destroy_function(device, handle, nullptr);
    }
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept {
    *this = std::move(other);
}

AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept {
    if (this != &other) {
        std::swap(device, other.device);
        std::swap(handle, other.handle);
        std::swap(destroy_function, other.destroy_function);
        std::swap(storage, other.storage);
    }
    return *this;
}

ComputePipeline::ComputePipeline(
    VkDevice logical_device,
    std::span<const std::byte> spir_v_bytes,
    std::uint32_t storage_buffer_count,
    std::uint32_t push_constant_size)
    : ComputePipeline() {
    std::vector<VkDescriptorType> descriptor_types(storage_buffer_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    *this = ComputePipeline(logical_device, spir_v_bytes, descriptor_types, push_constant_size);
}

ComputePipeline::ComputePipeline(
    VkDevice logical_device,
    std::span<const std::byte> spir_v_bytes,
    std::span<const VkDescriptorType> descriptor_types,
    std::uint32_t push_constant_size)
    : device(logical_device), binding_count(static_cast<std::uint32_t>(descriptor_types.size())) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(descriptor_types.size());
    for (std::uint32_t i = 0; i < descriptor_types.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = descriptor_types[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout_info.pBindings = bindings.data();
    check_vk(
        vkCreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr, &descriptor_set_layout),
        "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = push_constant_size == 0 ? 0U : 1U;
    pipeline_layout_info.pPushConstantRanges = push_constant_size == 0 ? nullptr : &push_range;
    check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

    if (spir_v_bytes.empty() || spir_v_bytes.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("Invalid embedded SPIR-V bytecode");
    }
    std::vector<std::uint32_t> spir_v(spir_v_bytes.size() / sizeof(std::uint32_t));
    std::memcpy(spir_v.data(), spir_v_bytes.data(), spir_v_bytes.size());
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spir_v.size() * sizeof(std::uint32_t);
    shader_info.pCode = spir_v.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    const VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &handle);
    vkDestroyShaderModule(device, shader_module, nullptr);
    check_vk(result, "vkCreateComputePipelines");
}

ComputePipeline::~ComputePipeline() {
    if (device != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, handle, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept {
    *this = std::move(other);
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        std::swap(device, other.device);
        std::swap(descriptor_set_layout, other.descriptor_set_layout);
        std::swap(pipeline_layout, other.pipeline_layout);
        std::swap(handle, other.handle);
        std::swap(binding_count, other.binding_count);
    }
    return *this;
}

Context::Impl::Impl(const ContextOptions& options) {
    const bool validation_enabled = options.enable_validation || ASDIFF_ENABLE_VALIDATION;
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (validation_enabled) {
        if (!has_instance_layer("VK_LAYER_KHRONOS_validation")) {
            throw std::runtime_error("Vulkan validation was requested but VK_LAYER_KHRONOS_validation is unavailable");
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    const bool portability_enumeration_available =
        has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability_enumeration_available) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "asdiff_render";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.pEngineName = "asdiff_render";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    if (portability_enumeration_available) {
        instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    instance_info.ppEnabledLayerNames = layers.data();
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();
    check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    if (validation_enabled) {
        const auto create_debug_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create_debug_messenger != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT debug_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug_info.pfnUserCallback = debug_callback;
            check_vk(create_debug_messenger(instance, &debug_info, nullptr, &debug_messenger), "debug messenger creation");
        }
    }

    std::uint32_t device_count = 0;
    check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan devices were found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check_vk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");
    if (options.device_index >= device_count) {
        throw std::out_of_range("Vulkan device_index is out of range");
    }
    physical_device = devices[options.device_index];
    device_info = make_device_info(physical_device);
    queue_family_index = find_compute_queue_family(physical_device);

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_info;
    std::vector<const char*> device_extensions;
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (has_device_extension(physical_device, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    if (device_info.supports_ray_query) {
        device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        ray_query.rayQuery = VK_TRUE;
        ray_query.pNext = &acceleration_structure;
        acceleration_structure.accelerationStructure = VK_TRUE;
        acceleration_structure.pNext = &buffer_address;
        buffer_address.bufferDeviceAddress = VK_TRUE;
        device_create_info.pNext = &ray_query;
    }
    device_create_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
    device_create_info.ppEnabledExtensionNames = device_extensions.data();
    check_vk(vkCreateDevice(physical_device, &device_create_info, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, queue_family_index, 0, &queue);

    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = queue_family_index;
    check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

    std::vector<VkDescriptorPoolSize> pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
    };
    if (device_info.supports_ray_query) {
        pool_sizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 512});
    }
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 512;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");
}

Context::Impl::~Impl() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (debug_messenger != VK_NULL_HANDLE) {
        const auto destroy_debug_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_debug_messenger != nullptr) {
            destroy_debug_messenger(instance, debug_messenger, nullptr);
        }
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

Buffer Context::Impl::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage) const {
    return Buffer(physical_device, device, size, usage);
}

ComputePipeline Context::Impl::create_pipeline(
    const std::string& shader_name,
    std::uint32_t storage_buffer_count,
    std::uint32_t push_constant_size) const {
    if (const auto override_directory = shader_directory_override()) {
        const auto bytes = read_spir_v(*override_directory / shader_name);
        return ComputePipeline(device, bytes, storage_buffer_count, push_constant_size);
    }
    return ComputePipeline(device, embedded_shader(shader_name), storage_buffer_count, push_constant_size);
}

ComputePipeline Context::Impl::create_pipeline(
    const std::string& shader_name,
    std::span<const VkDescriptorType> descriptor_types,
    std::uint32_t push_constant_size) const {
    if (const auto override_directory = shader_directory_override()) {
        const auto bytes = read_spir_v(*override_directory / shader_name);
        return ComputePipeline(device, bytes, descriptor_types, push_constant_size);
    }
    return ComputePipeline(device, embedded_shader(shader_name), descriptor_types, push_constant_size);
}

RayQueryScene Context::Impl::create_ray_query_scene(
    std::span<const float> positions,
    std::span<const std::uint32_t> triangle_indices) const {
    if (!device_info.supports_ray_query) {
        throw std::runtime_error("The selected Vulkan device does not support ray queries");
    }
    if (positions.empty() || positions.size() % 3 != 0 || triangle_indices.empty() || triangle_indices.size() % 3 != 0) {
        throw std::invalid_argument("Ray-query geometry must contain float3 positions and triangle indices");
    }
    const std::scoped_lock lock(dispatch_mutex);
    const auto create_acceleration_structure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    const auto destroy_acceleration_structure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    const auto get_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    const auto command_build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    const auto get_acceleration_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    if (create_acceleration_structure == nullptr || destroy_acceleration_structure == nullptr ||
        get_build_sizes == nullptr || command_build == nullptr || get_acceleration_address == nullptr) {
        throw std::runtime_error("Required Vulkan acceleration-structure entry points are unavailable");
    }

    RayQueryScene scene;
    const auto input_usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    scene.vertices = create_buffer(positions.size_bytes(), input_usage);
    scene.indices = create_buffer(triangle_indices.size_bytes(), input_usage);
    scene.vertices.upload(positions.data(), positions.size_bytes());
    scene.indices.upload(triangle_indices.data(), triangle_indices.size_bytes());

    auto build_structure = [&](VkAccelerationStructureTypeKHR type,
                               const VkAccelerationStructureGeometryKHR& geometry,
                               std::uint32_t primitive_count) {
        VkAccelerationStructureBuildGeometryInfoKHR build_info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        build_info.type = type;
        build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.geometryCount = 1;
        build_info.pGeometries = &geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        get_build_sizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_count, &sizes);

        AccelerationStructure structure;
        structure.device = device;
        structure.destroy_function = destroy_acceleration_structure;
        structure.storage = create_buffer(
            sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR create_info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create_info.buffer = structure.storage.handle;
        create_info.size = sizes.accelerationStructureSize;
        create_info.type = type;
        check_vk(create_acceleration_structure(device, &create_info, nullptr, &structure.handle),
                 "vkCreateAccelerationStructureKHR");

        auto scratch = create_buffer(
            sizes.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        build_info.dstAccelerationStructure = structure.handle;
        build_info.scratchData.deviceAddress = scratch.device_address();
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = primitive_count;
        const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};

        VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_buffer_info.commandPool = command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
        command_build(command_buffer, 1, &build_info, ranges);
        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        check_vk(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
        check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
        return structure;
    };

    VkAccelerationStructureGeometryTrianglesDataKHR triangle_data{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangle_data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangle_data.vertexData.deviceAddress = scene.vertices.device_address();
    triangle_data.vertexStride = sizeof(float) * 3;
    triangle_data.maxVertex = static_cast<std::uint32_t>(positions.size() / 3 - 1);
    triangle_data.indexType = VK_INDEX_TYPE_UINT32;
    triangle_data.indexData.deviceAddress = scene.indices.device_address();
    VkAccelerationStructureGeometryKHR bottom_geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    bottom_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    bottom_geometry.geometry.triangles = triangle_data;
    bottom_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    scene.bottom_level = build_structure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        bottom_geometry,
        static_cast<std::uint32_t>(triangle_indices.size() / 3));

    VkAccelerationStructureDeviceAddressInfoKHR address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure = scene.bottom_level.handle;
    VkAccelerationStructureInstanceKHR geometry_instance{};
    geometry_instance.transform.matrix[0][0] = 1.0F;
    geometry_instance.transform.matrix[1][1] = 1.0F;
    geometry_instance.transform.matrix[2][2] = 1.0F;
    geometry_instance.mask = 0xff;
    geometry_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    geometry_instance.accelerationStructureReference = get_acceleration_address(device, &address_info);
    auto instance_buffer = create_buffer(sizeof(geometry_instance), input_usage);
    instance_buffer.upload(&geometry_instance, sizeof(geometry_instance));
    VkAccelerationStructureGeometryInstancesDataKHR instance_data{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instance_data.arrayOfPointers = VK_FALSE;
    instance_data.data.deviceAddress = instance_buffer.device_address();
    VkAccelerationStructureGeometryKHR top_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_geometry.geometry.instances = instance_data;
    top_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    scene.top_level = build_structure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_geometry, 1);
    return scene;
}

void Context::Impl::dispatch(
    const ComputePipeline& pipeline,
    std::span<const VkDescriptorBufferInfo> buffers,
    const void* push_constants,
    std::uint32_t push_constant_size,
    std::uint32_t group_count_x,
    std::uint32_t group_count_y,
    std::uint32_t group_count_z) const {
    const std::scoped_lock lock(dispatch_mutex);
    if (buffers.size() != pipeline.binding_count) {
        throw std::invalid_argument("Descriptor buffer count does not match pipeline layout");
    }
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &pipeline.descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check_vk(vkAllocateDescriptorSets(device, &set_info, &descriptor_set), "vkAllocateDescriptorSets");

    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    if (push_constant_size != 0) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            push_constant_size,
            push_constants);
    }
    vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);
    check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    check_vk(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
    check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    vkFreeDescriptorSets(device, descriptor_pool, 1, &descriptor_set);
}

void Context::Impl::dispatch_ray_query(
    const ComputePipeline& pipeline,
    const Buffer& positions,
    const Buffer& normals,
    const Buffer& raster,
    const AccelerationStructure& scene,
    const Buffer& visibility,
    const void* push_constants,
    std::uint32_t push_constant_size,
    std::uint32_t group_count_x) const {
    const std::scoped_lock lock(dispatch_mutex);
    if (!device_info.supports_ray_query || pipeline.binding_count != 5 || scene.handle == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid ray-query dispatch state");
    }
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &pipeline.descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check_vk(vkAllocateDescriptorSets(device, &set_info, &descriptor_set), "vkAllocateDescriptorSets");

    const std::array buffer_infos{
        VkDescriptorBufferInfo{positions.handle, 0, positions.size},
        VkDescriptorBufferInfo{normals.handle, 0, normals.size},
        VkDescriptorBufferInfo{raster.handle, 0, raster.size},
        VkDescriptorBufferInfo{visibility.handle, 0, visibility.size},
    };
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (std::uint32_t binding = 0; binding < 3; ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = descriptor_set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &buffer_infos[binding];
    }
    VkWriteDescriptorSetAccelerationStructureKHR acceleration_write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    acceleration_write.accelerationStructureCount = 1;
    acceleration_write.pAccelerationStructures = &scene.handle;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].pNext = &acceleration_write;
    writes[3].dstSet = descriptor_set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = descriptor_set;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &buffer_infos[3];
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout,
        0, 1, &descriptor_set, 0, nullptr);
    if (push_constant_size != 0) {
        vkCmdPushConstants(
            command_buffer, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, push_constant_size, push_constants);
    }
    vkCmdDispatch(command_buffer, group_count_x, 1, 1);
    check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    check_vk(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
    check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    vkFreeDescriptorSets(device, descriptor_pool, 1, &descriptor_set);
}

Context::Context(const ContextOptions& options) : impl_(std::make_unique<Impl>(options)) {}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

const DeviceInfo& Context::device_info() const noexcept {
    return impl_->device_info;
}

std::vector<DeviceInfo> Context::enumerate_devices() {
    VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "asdiff_render_device_enumeration";
    application_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &application_info;
    std::vector<const char*> extensions;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    }
#endif
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    VkInstance temporary_instance = VK_NULL_HANDLE;
    check_vk(vkCreateInstance(&create_info, nullptr, &temporary_instance), "vkCreateInstance");
    std::uint32_t count = 0;
    check_vk(vkEnumeratePhysicalDevices(temporary_instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physical_devices(count);
    check_vk(vkEnumeratePhysicalDevices(temporary_instance, &count, physical_devices.data()), "vkEnumeratePhysicalDevices");
    std::vector<DeviceInfo> result;
    result.reserve(count);
    for (const auto physical_device : physical_devices) {
        result.push_back(make_device_info(physical_device));
    }
    vkDestroyInstance(temporary_instance, nullptr);
    return result;
}

} // namespace asdiff_render
