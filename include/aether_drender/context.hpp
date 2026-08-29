#pragma once

#include <memory>
#include <vector>

#include "aether_drender/types.hpp"

namespace aether_drender {

class Context {
public:
    class Impl;

    explicit Context(const ContextOptions& options = {});
    ~Context();

    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] const DeviceInfo& device_info() const noexcept;
    [[nodiscard]] static std::vector<DeviceInfo> enumerate_devices();

private:
    std::unique_ptr<Impl> impl_;

    friend class Rasterizer;
    friend class TextureBaker;
    friend class TextureRefiner;
};

} // namespace aether_drender
