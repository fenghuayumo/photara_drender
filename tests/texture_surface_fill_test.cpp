#include "photara_drender/texture_padding.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    using namespace photara_drender;
    TextureBakeOutput atlas;
    atlas.width = 7;
    atlas.height = 1;
    atlas.color.assign(7 * 4, 0.0F);
    atlas.valid_mask = {1, 0, 1, 0, 0, 0, 0};
    atlas.coverage_mask = {1, 1, 1, 1, 1, 0, 0};
    // Faces 0 and 1 share a mesh edge but are separated in UV space by blue
    // face 2. Face 3 is an entirely unseen disconnected component.
    atlas.face_ids = {0, 0, 2, 1, 3, UINT32_MAX, UINT32_MAX};
    atlas.color[0] = 0.8F;
    atlas.color[3] = 1.0F;
    atlas.color[2 * 4 + 2] = 0.6F;
    atlas.color[2 * 4 + 3] = 1.0F;
    const auto measured = atlas.color;
    const auto valid = atlas.valid_mask;
    const std::uint32_t topology[] = {0, 1, 2, 2, 1, 3, 4, 5, 6, 7, 8, 9};
    const auto stats = fill_texture_on_surface(atlas, topology);
    require(stats.measured_faces == 2 && stats.propagated_faces == 1,
        "face adjacency did not cross UV seam correctly");
    require(stats.filled_texels == 2 && stats.remaining_texels == 1,
        "unseen component must remain unfilled");
    require(atlas.color[4] == 0.8F && atlas.color[3 * 4] == 0.8F &&
        atlas.color[3 * 4 + 2] == 0.0F, "colour leaked from unrelated blue island");
    for (int pixel : {0, 2})
        for (int channel = 0; channel < 4; ++channel)
            require(atlas.color[pixel * 4 + channel] == measured[pixel * 4 + channel],
                "measured colour changed");
    require(atlas.valid_mask == valid, "fill changed projection validity");
    TexturePaddingOptions padding;
    padding.margin = 2;
    padding.fill_unobserved = false;
    const auto padded = pad_texture_atlas(atlas, padding);
    require(padded.unobserved_texels == 2 && padded.remaining_unobserved == 1,
        "padding lost surface fill accounting");
    require(atlas.color[5 * 4] == 0.8F && atlas.color[4 * 4] == 0.0F,
        "gutter must use extrapolated colour without filling unrelated component");
    require(atlas.valid_mask == valid && atlas.filled_mask[3] == 1.0F,
        "padding discarded fill provenance");
    const auto repeated = pad_texture_atlas(atlas, padding);
    require(repeated.gutter_texels == padded.gutter_texels &&
        repeated.unobserved_texels == padded.unobserved_texels,
        "repeated padding grew the margin or changed fill classification");
    auto invalid = atlas;
    invalid.face_ids[1] = 100;
    bool rejected = false;
    try { (void)fill_texture_on_surface(invalid, topology); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid raster face ID accepted");
    auto empty = atlas;
    empty.valid_mask.assign(7, 0.0F);
    empty.filled_mask.clear();
    const auto unseen = fill_texture_on_surface(empty, topology);
    require(unseen.filled_texels == 0 && unseen.remaining_texels == 5,
        "unobserved mesh should not invent a source");
    std::cout << "Surface fill and gutter regression passed\n";
}
