# UVAtlas texture baking and projection

## Pipeline

The baking pipeline has four stages:

1. `unwrap_mesh_uv()` partitions and packs the input mesh with Microsoft UVAtlas. Duplicated vertices are exposed through
   `vertex_remap`; per-face chart IDs are preserved for seam-aware optimization.
2. The Vulkan rasterizer renders UV coordinates into atlas space and interpolates world-space position and normal for every
   covered texel.
3. In the default `ray_query` quality mode, a Vulkan BLAS/TLAS is built for the mesh. Inline ray queries from atlas surface
   points to each camera are the authoritative visibility test; no shadow map is generated or sampled.
4. `shadow_map` remains the portable fallback. The legacy `hybrid_ray_query` mode combines both tests for compatibility,
   but its finite-resolution PCF test can reject samples that an exact ray query considers visible.

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
ray query. `ray_query` uses all three without generating a shadow map. Set `allow_visibility_fallback=False` to require the
ray path; otherwise unsupported devices use PCF shadow maps and return `used_ray_query=False`.

The core renderer and shadow-map projection remain Vulkan 1.2 portable. Microsoft UVAtlas is an optional CPU build-time
backend: Windows can fetch the pinned release automatically; Linux builds can provide the `uvatlas`, DirectXMath, and
DirectX-Headers CMake packages, or disable UVAtlas and pass a pre-unwrapped mesh.

## Seam-safe output and differentiable refinement

Projection-valid texels do not cover the empty gutter around every UV chart. Use `padding=8` in
`project_texture_atlas()` (or `pad_texture_atlas()` on an existing bake) before filtered rendering or model export. The
projection-valid mask remains unchanged, while RGB and alpha guard texels are extended into the gutter.

`examples/refine_colmap_texture.py` uses the native C++ `TextureRefiner`. Camera rasters, interpolated UVs, photographs,
masks, the 4K texture, gradients, and both Adam moments are allocated once in persistent Vulkan buffers (preferring a
host-visible device-local memory type). The complete optimization command stream is recorded and submitted once; texture
and loss histories are downloaded only after the queue finishes. PyTorch is not involved in this path.

The default high-quality profile performs 1000 mini-batch photometric Adam steps with a cosine learning-rate schedule,
then resets the Adam state and performs native seam-only polish. Pixel-to-atlas gradients use portable uint
compare/exchange to atomically accumulate IEEE float values, avoiding a dependency on optional float-atomic extensions.
The older `optimize_texture_atlas()` PyTorch adapter remains available for research code needing its general autograd API.
