#include "photara_drender/uv_atlas.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    try {
        if (!photara_drender::has_uv_atlas_backend()) {
            std::cout << "SKIP: UVAtlas unavailable\n";
            return 0;
        }
        constexpr std::uint32_t faces = 50'001;
        std::vector<float> positions;
        std::vector<std::uint32_t> indices;
        positions.reserve(faces * 9);
        for (std::uint32_t face = 0; face < faces; ++face) {
            const float x = static_cast<float>(face % 224);
            const float y = static_cast<float>(face / 224);
            positions.insert(positions.end(), {x, y, 0.F, x + .4F, y, 0.F, x, y + .4F, 0.F});
            indices.insert(indices.end(), {face * 3, face * 3 + 1, face * 3 + 2});
        }
        photara_drender::UvAtlasOptions options;
        options.width = options.height = 4096;
        options.gutter = 2.F;
        options.parallel_partitions = 16;
        const auto output = photara_drender::unwrap_uv(positions, indices, options);
        if (output.indices.size() != indices.size() || output.chart_count != faces)
            throw std::runtime_error("Partition packing dropped faces or charts");
        for (std::size_t corner = 0; corner < indices.size(); ++corner) {
            const auto vertex = output.indices[corner];
            if (output.vertex_remap[vertex] != indices[corner])
                throw std::runtime_error("Partition packing changed face geometry or ordering");
            for (std::size_t axis = 0; axis < 3; ++axis)
                if (output.positions[vertex * 3 + axis] != positions[indices[corner] * 3 + axis])
                    throw std::runtime_error("Partition packing moved a vertex");
        }
        for (const float value : output.uv)
            if (!std::isfinite(value) || value < 0.F || value > 1.F)
                throw std::runtime_error("Partition UV lies outside the atlas");

        // Rasterize all chart interiors to detect overlaps between packed tiles
        // and within each native UVAtlas packing result.
        constexpr auto empty = std::numeric_limits<std::uint32_t>::max();
        std::vector<std::uint32_t> owner(options.width * options.height, empty);
        for (std::uint32_t face = 0; face < faces; ++face) {
            double x[3], y[3];
            for (int corner = 0; corner < 3; ++corner) {
                const auto vertex = output.indices[face * 3 + corner];
                x[corner] = output.uv[vertex * 2] * options.width;
                y[corner] = output.uv[vertex * 2 + 1] * options.height;
            }
            const double denominator = (y[1] - y[2]) * (x[0] - x[2]) +
                (x[2] - x[1]) * (y[0] - y[2]);
            if (std::abs(denominator) < 1e-8)
                throw std::runtime_error("Packing collapsed a UV triangle");
            const int xmin = std::max(0, static_cast<int>(std::floor(*std::min_element(x, x + 3))));
            const int xmax = std::min(4095, static_cast<int>(std::ceil(*std::max_element(x, x + 3))));
            const int ymin = std::max(0, static_cast<int>(std::floor(*std::min_element(y, y + 3))));
            const int ymax = std::min(4095, static_cast<int>(std::ceil(*std::max_element(y, y + 3))));
            for (int py = ymin; py <= ymax; ++py) {
                for (int px = xmin; px <= xmax; ++px) {
                    const double a = ((y[1] - y[2]) * (px + .5 - x[2]) +
                        (x[2] - x[1]) * (py + .5 - y[2])) / denominator;
                    const double b = ((y[2] - y[0]) * (px + .5 - x[2]) +
                        (x[0] - x[2]) * (py + .5 - y[2])) / denominator;
                    if (a < 0 || b < 0 || a + b > 1) continue;
                    auto& pixel = owner[static_cast<std::size_t>(py) * options.width + px];
                    if (pixel != empty && pixel != face)
                        throw std::runtime_error("Packed UV charts overlap");
                    pixel = face;
                }
            }
        }
        std::cout << "Parallel UV packing preserved 50001 faces and vertices, with no chart overlap\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
