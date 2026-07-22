# Differentiable texture-atlas optimization

## Supported optimization path

The current end-to-end path is:

1. Transform mesh vertices into clip space for a calibrated view.
2. Rasterize into the view or a viewport inside a merged ROI.
3. Interpolate vertex UV coordinates for visible pixels.
4. Sample the atlas with differentiable bilinear filtering.
5. Compare the render with the projected photograph through a visibility mask.
6. Backpropagate into atlas texels, vertex UV coordinates, or clip-space positions.

The production COLMAP path uses the native C++ `TextureRefiner`: view rasters, masks, atlas gradients, Adam state, and seam
polish all remain in persistent Vulkan buffers. `optimize_texture_atlas()` is the optional PyTorch research adapter for
experiments that also optimize UV coordinates, camera parameters, exposure, or geometry.

## AIHoloImager integration

The existing `DiffOptimizer.Render()` texture branch maps directly to:

```python
from asdiff_render import Rasterizer, render_textured_mesh

renderer = Rasterizer(device_index=0)
result = render_textured_mesh(
    pos_clip,
    indices,
    vertex_uv,
    atlas,
    (image_height, image_width),
    rasterizer=renderer,
    viewport=(viewport_x, viewport_y, viewport_width, viewport_height),
)
image = result.image
valid_mask = result.valid_mask
```

Important differences from the current wrapper:

- Resolution uses `(height, width)` order.
- Viewport uses lower-left `(x, y, width, height)` coordinates. Convert a top-left ROI before calling if the image loader
  retains top-left origin.
- UV coordinates are normalized. Texel centers are `(x + 0.5) / width` and `(y + 0.5) / height`.
- Atlas content and UV inputs must be contiguous float32 tensors.
- Alpha, foreground masks, depth consistency, and camera-facing weights should be multiplied into `valid_mask`.

The temporary early return that skips `FitTexture()` can be replaced incrementally: first run level-zero sampling with
silhouette pixels excluded, then enable mipmapping and antialiasing after those modules are added.

## Recommended objective

A practical multi-view objective is:

`L = L_photo + lambda_tv * L_chart_tv + lambda_seam * L_seam + lambda_prior * L_initial`

- `L_photo`: masked Charbonnier or Huber loss, optionally with per-view exposure/color calibration.
- `L_chart_tv`: total variation only between valid texels belonging to the same xatlas chart.
- `L_seam`: color agreement for UV pairs representing the same geometric seam.
- `L_initial`: confidence-weighted preservation of high-quality projected texels.

For view weighting, combine alpha/foreground validity, depth consistency, `max(dot(normal, view_dir), 0)`, distance,
photo sharpness, and occlusion confidence. Normalize each loss by its valid weight sum instead of the full image area.

## Data needed from xatlas

For seam-aware optimization, retain more than the final UV coordinates:

- original mesh vertex or corner index for every atlas vertex;
- chart index for every atlas triangle/texel;
- pairs of duplicated UV vertices that map to the same original geometric vertex/edge;
- an atlas-valid mask including chart padding.

These values allow construction of `chart_ids`, `atlas_valid_mask`, and `seam_uv_pairs` accepted by the high-level helpers.

## Remaining quality work

- Mipmapped trilinear filtering and gradients are required for stable minified views.
- Silhouette antialiasing is required before including boundary pixels in photometric losses.
- A UV-layout optimizer additionally needs signed-area, overlap, stretch, texel-density, boundary, and chart-packing
  constraints; optimizing UV coordinates with only photometric loss can fold or overlap charts.
- Zero-copy Vulkan/PyTorch interop and device-local resource pooling are required for large production datasets.
