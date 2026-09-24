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

`softmax` is the recommended quality mode. Every sample is weighted by its pixel footprint on the surface,
`|n.v| / d^2` (the inverse of how much world space one image pixel covers there), and views are combined with a
scene-relative exponential softmax (`softmax_scale`, automatically `20 * min squared vertex-to-camera distance` unless the
caller overrides it). The view that resolves a texel best therefore dominates instead of being averaged with grazing,
distant, or lower-resolution samples, which removes ghosting and grazing-angle aliasing from the atlas. The per-view
`weight` multiplies the exponential sharpness in this mode. The accumulation is a numerically stable online softmax, so
the linear-average host normalization still applies.

Object masks are a rendering aid, not a statement that a pixel is unobserved: a mask hides the capture volume behind
the subject. `mask_floor` (> 0) therefore keeps masked pixels as weak samples instead of hard vetoes, so the surface an
object rests on keeps its real texture while masked-in pixels still dominate wherever they exist. Use `mask_floor = 0`
for strict mask semantics, and `1` to ignore masks during the bake.

## Coordinate contract

- `positions`: world-space float32 `[vertex_count, 3]`.
- `world_to_clip`: row-major float32 `[view_count, 4, 4]`, evaluated as `clip = matrix * [position, 1]`.
- Clip-space `x/y/z` are divided by positive `w`; Vulkan-style depth matrices using `z` in `[0, 1]` are recommended.
- Images, masks, viewports, and atlas UV use a lower-left origin.
- Images are float RGB or RGBA in the caller's color space, usually display-referred sRGB so the bake matches the photographs. The baker does not convert between sRGB and linear. Pass scene-linear values only when the whole projection and refinement should stay linear, and encode sRGB yourself on export.
- `viewport` is lower-left `(x, y, width, height)`. A full image uses `(0, 0, image_width, image_height)`.
- `camera_position` is world-space and controls both normal-facing confidence and ray direction.

For a top-left ROI `(x, y, width, height)` in an image of height `full_height`, use:

```python
vulkan_viewport = (x, full_height - y - height, width, height)
image = np.flipud(image_from_opencv).copy()
mask = np.flipud(mask_from_opencv).copy()
```

## Input conventions

`TextureBaker` builds BLAS/TLAS and uses inline ray queries; camera shadow rasters are created only for an explicit
`shadow_map` request or an allowed unsupported-device fallback. A projected bake is an initialization, not the final
texture: the native C++ `TextureRefiner` refines it in Vulkan before export.

Pass `projection.proj_mtx * projection.view_mtx * model_mtx` as `world_to_clip` when input positions are still in model
space. If positions are already transformed into world space, omit `model_mtx`. Callers whose matrices are stored
column-major must transpose them explicitly into the documented row-major NumPy layout instead of reinterpreting the raw
bytes.

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

The high-level COLMAP examples use `delight` as their default texture source. Intrinsic-delighted photographs must be
precomputed in `<images_path>_delighted` or supplied with `--delighted-images-path`; use `--texture-source rgb` for a
deliberate original-RGB bake. The selected source is written into the bake archive, and refinement rejects a different
source so an albedo initialization can never be optimized against lit RGB photographs by accident.

Projection-valid texels do not cover the empty gutter around every UV chart. Use `padding=8` in
`project_texture_atlas()` (or `pad_texture_atlas()` on an existing bake) before filtered rendering or model export. The
projection-valid mask remains unchanged, while RGB and alpha guard texels are extended into the gutter.

`pad_texture_atlas()` is implemented natively (`photara_drender::pad_texture_atlas`) with a Danielsson 8SSEDT nearest
source field, so it also fills rasterized chart texels that never received a sample: masked-out regions, texels rejected
by the minimum facing cosine, and surfaces occluded in every view. Those texels copy the nearest valid texel instead of
staying black, which matters for object scans whose masks hide the surrounding capture volume. Filled texels are reported
in `filled_mask` (never in `valid_mask`), so downstream tools can still distinguish measured from extrapolated texels.
Set `fill_unobserved=false` for a conservative guard-band-only fill.

`examples/bake_colmap.py` keeps the initial ray-query projection in memory, runs 1000 native refinement steps by default,
and exports the PNG/GLB only after seam polish. Use `--no-optimize` to explicitly export an unrefined bake, or
`--save-initial-bake` to request an intermediate checkpoint. `examples/refine_colmap_texture.py` remains the resumable
entry point for an existing bake archive.

Both entry points use the native C++ `TextureRefiner`. Camera rasters, interpolated UVs, photographs,
masks, the 4K texture, gradients, and both Adam moments are allocated once in persistent Vulkan buffers (preferring a
host-visible device-local memory type). The complete optimization command stream is recorded and submitted once; texture
and loss histories are downloaded only after the queue finishes. PyTorch is not involved in this path.

The default high-quality profile performs 1000 mini-batch photometric Adam steps with a cosine learning-rate schedule,
then resets the Adam state and performs native seam-only polish. Pixel-to-atlas gradients use portable uint
compare/exchange to atomically accumulate IEEE float values, avoiding a dependency on optional float-atomic extensions.
The COLMAP refinement example requires foreground masks. Its photometric normalization includes only rasterized pixels
inside the eroded foreground mask, and its diagnostic panels/MAE/RMSE use the exact same mask and crop to the foreground.
The older `optimize_texture_atlas()` PyTorch adapter remains available for research code needing its general autograd API.
