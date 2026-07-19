# Architecture

## Public layers

- `asdiff_render::Context` owns the Vulkan instance, selected compute device, queue, command pool, and descriptor pool.
- `asdiff_render::Rasterizer` owns immutable compute pipelines and exposes rasterization and interpolation forward/backward APIs.
- `_asdiff_render` maps contiguous NumPy arrays to the C++ API with shape and range validation.
- `asdiff_render.torch` composes the native calls into PyTorch autograd functions without linking to a particular PyTorch ABI.

All project-defined functions, variables, files, and namespaces use `snake_case`. Types follow Rust convention and use
`PascalCase`; constants use `UPPER_SNAKE_CASE`.

## Rasterization pipeline

1. CPU tile binning projects triangle bounds and builds compact per-tile triangle lists.
2. One HLSL compute invocation evaluates one pixel against only its tile list.
3. Coverage is tested in screen space; barycentrics are evaluated in homogeneous clip space.
4. The nearest depth wins, with triangle order as the deterministic tie-break.
5. The output is `[u, v, z/w, triangle_id + 1]`; the optional derivative output is
   `[du/dx, du/dy, dv/dx, dv/dy]`.

The discrete visibility decision is intentionally not differentiated. The backward pipeline differentiates the continuous
`u/v` values for the selected primitive. One invocation accumulates each triangle's three corner gradients over its screen
bounding box. A second pipeline reduces corner gradients through a CPU-built vertex adjacency table. This avoids requiring
`VK_EXT_shader_atomic_float` and produces deterministic sums on Vulkan 1.2 implementations.

## Attribute interpolation

The interpolation forward pass evaluates `u * a0 + v * a1 + (1-u-v) * a2`. Backward produces both vertex-attribute
gradients and raster gradients, so PyTorch can chain an image loss through interpolation into clip-space positions.

## Synchronization and portability

The baseline uses host-visible coherent storage buffers and synchronous queue submission. Dispatch is serialized per
context, making a shared `Rasterizer` safe for concurrent Python callers. Runtime shaders are HLSL compiled by DXC to
Vulkan 1.2 SPIR-V and embedded into the native library; wheel copies are also retained for inspection and development-time
override. Portability enumeration and portability-subset extensions are enabled when present, allowing MoltenVK devices to
participate.

## Deliberate current limits

- Inputs are float32 clip positions `[V, 4]`, uint32 triangle indices `[T, 3]`, and a single image/layer.
- Triangles crossing the near plane are skipped because homogeneous clipping is not implemented yet.
- Gradients through barycentric pixel derivatives (a second-order path), texture filtering, and silhouette antialiasing are
  future modules.
- The portable PyTorch path stages through host memory. A production throughput backend should add external-memory and
  external-semaphore tensor interop while retaining these APIs.
- Device-local buffer pooling, asynchronous command batches, pipeline cache persistence, and GPU tile binning are the next
  performance milestones.
