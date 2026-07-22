# UVAtlas texture baking and projection

## Pipeline

The baking pipeline has four stages:

1. `unwrap_mesh_uv()` partitions and packs the input mesh with Microsoft UVAtlas. Duplicated vertices are exposed through
   `vertex_remap`; per-face chart IDs are preserved for seam-aware optimization.
2. The Vulkan rasterizer renders UV coordinates into atlas space and interpolates world-space position and normal for every
   covered texel.
3. Each calibrated view is rasterized into a camera-space depth buffer. The projection HLSL kernel performs bilinear photo
   sampling, PCF depth testing, optional photo-mask testing, and view-angle confidence evaluation.
4. In `hybrid_ray_query` mode, a Vulkan BLAS/TLAS is built for the mesh. Inline ray queries from atlas surface points to the
   camera reject occlusion that survives the finite-resolution shadow map. Devices without ray-query support can fall back
   to the shadow path.

`best_view` keeps the highest-confidence photograph per texel. `weighted_average` accumulates all visible samples and is
usually a better initialization for subsequent differentiable optimization. `source_view` always records the strongest
individual contributing view.

## Coordinate contract

- `positions`: world-space float32 `[vertex_count, 3]`.
- `world_to_clip`: row-major float32 `[view_count, 4, 4]`, evaluated as `clip = matrix * [position, 1]`.
- Clip-space `x/y/z` are divided by positive `w`; Vulkan-style depth matrices using `z` in `[0, 1]` are recommended.
- Images, masks, viewports, and atlas UV use a lower-left origin.
- Images contain linear RGB or RGBA float32 data. Convert sRGB photographs to linear space before confidence blending.
- `viewport` is lower-left `(x, y, width, height)`. A full image uses `(0, 0, image_width, image_height)`.
- `camera_position` is world-space and controls both normal-facing confidence and ray direction.

For a top-left ROI `(x, y, width, height)` in an image of height `full_height`, use:

```python
vulkan_viewport = (x, full_height - y - height, width, height)
image = np.flipud(image_from_opencv).copy()
mask = np.flipud(mask_from_opencv).copy()
```

## AIHoloImager integration

The implementation replaces the existing texture reconstruction sequence directly:

- `Mesh::UnwrapUv()` becomes `unwrap_mesh_uv()`; retain `vertex_remap` when rebuilding normals and other attributes.
- `FlattenVs/FlattenPs` becomes atlas-space Vulkan compute rasterization plus attribute interpolation.
- `GenShadowMap()` becomes the camera raster generated internally by `TextureBaker`.
- `ProjectTextureCs` becomes `project_texture.hlsl`, with float accumulation, masks, PCF, and optional ray visibility.
- The projected result becomes the initial parameter for `optimize_texture_atlas()` instead of being the final texture.

Pass `projection.proj_mtx * projection.view_mtx * model_mtx` as `world_to_clip` when input positions are still in model
space. If positions are already transformed into world space, omit `model_mtx`. Convert GLM column-major storage to the
documented row-major NumPy representation instead of copying its raw bytes blindly.

Recommended masks combine foreground alpha, camera ROI, depth confidence, and manually excluded regions. UVAtlas chart IDs
can be rasterized into `chart_ids` for the differentiable optimizer so total variation does not blur across chart borders.

## Vulkan capability behavior

`DeviceInfo.supports_ray_query` reports the combined availability of buffer device address, acceleration structures, and
ray query. `hybrid_ray_query` uses all three when available. Set `allow_visibility_fallback=False` to require the ray path;
otherwise unsupported devices use PCF shadow maps and return `used_ray_query=False`.

The core renderer and shadow-map projection remain Vulkan 1.2 portable. Microsoft UVAtlas is an optional CPU build-time
backend: Windows can fetch the pinned release automatically; Linux builds can provide the `uvatlas`, DirectXMath, and
DirectX-Headers CMake packages, or disable UVAtlas and pass a pre-unwrapped mesh.

## Seam-safe output and differentiable refinement

Projection-valid texels do not cover the empty gutter around every UV chart. Use `padding=8` in
`project_texture_atlas()` (or `pad_texture_atlas()` on an existing bake) before filtered rendering or model export. The
projection-valid mask remains unchanged, while RGB and alpha guard texels are extended into the gutter.

`examples/refine_colmap_texture.py` refines an initial bake by rendering it back into the calibrated photographs. It
combines the foreground masks with a photometric loss, chart-aware total variation, dense corresponding samples along UV
seams, an initial-atlas prior, and a final seam-only polish. An optional raster cache avoids repeating mesh rasterization
and UV interpolation when a camera is revisited, but it is disabled by default because 4K atlas staging and optimizer
updates dominate this version's runtime while cached per-view buffers consume substantial host memory.

The default refinement profile performs one photograph step per calibrated view and then 30 inexpensive seam-only steps.
For a stronger two-pass profile, use `--steps 152 --tv-weight 1e-5 --prior-weight 0.05`; per-step seam loss is normally left
at zero because the final seam polish avoids downloading a second full-atlas gradient during every photograph step.
