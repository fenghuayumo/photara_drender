# photara_drender

`photara_drender` is a cross-platform differentiable rendering and texture baking library
implemented from scratch using C++20, Vulkan 1.2 compute, and HLSL.
It requires neither CUDA nor an OpenGL context and provides C++, NumPy, and PyTorch interfaces.

Version `0.4.0` includes:

- Differentiable triangle rasterization with tile binning, perspective-correct barycentric coordinates, and pixel derivatives.
- Vertex attribute interpolation and backpropagation.
- Differentiable bilinear texture sampling with atlas texel and UV gradients.
- Microsoft UVAtlas unwrapping with vertex remapping, face chart IDs, and stretch information.
- Position and normal mapping into atlas space.
- Texture projection from multiple cameras, PCF shadow-map visibility, and confidence-weighted blending.
- Precise occlusion checks using `VK_KHR_ray_query` with BLAS/TLAS, with a shadow-map fallback on unsupported devices.
- Differentiable texture atlas optimization across multiple views, chart-aware total variation, and seam consistency loss.

Project-defined functions, variables, files, and namespaces use `snake_case`; types use
`PascalCase`; constants use `UPPER_SNAKE_CASE`. All shaders are written in HLSL and compiled
into Vulkan SPIR-V with DXC.

## Building

Building requires CMake 3.24+, a C++20 compiler, Vulkan SDK 1.2+, and DXC. UV unwrapping builds
the `third_party/UVAtlas` submodule by default
(<https://github.com/fenghuayumo/UVAtlas>). Clone the repository with its submodules:

```powershell
git clone --recurse-submodules https://github.com/fenghuayumo/photara_drender.git
cmake -S . -B build -DPHOTARA_BUILD_PYTHON_BINDINGS=ON -DPHOTARA_BUILD_NATIVE_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

For an existing checkout, initialize UVAtlas with
`git submodule update --init third_party/UVAtlas`. Setting `PHOTARA_USE_VENDORED_UVATLAS=OFF`
uses an installed package instead, or downloads Microsoft's `oct2025` source release on Windows.

Build options:

- `PHOTARA_ENABLE_UVATLAS=OFF`: build only the rendering and projection modules.
- `PHOTARA_USE_VENDORED_UVATLAS=OFF`: use an installed package or `PHOTARA_FETCH_UVATLAS` instead of the submodule.
- `PHOTARA_FETCH_UVATLAS=OFF`: require an installed `uvatlas` CMake package.
- `PHOTARA_UVATLAS_USE_OPENMP=ON`: enable upstream UVAtlas OpenMP chart parameterization.
- `PHOTARA_BUILD_MESH_TOOLS=ON`: build the CGAL manifold repair and mesh decimation tools.
- `PHOTARA_ENABLE_VALIDATION=ON`: enable Vulkan validation layers.

Build a Python wheel with `python -m pip wheel .`.
Windows wheels include the VC++ Release runtime and Vulkan loader DLLs in the Python package
directory by default and register the DLL search path on import. A separate VC++ Redistributable
or Vulkan SDK installation is not required to use the wheel. The target machine still needs
a Vulkan-capable graphics driver and a Python version compatible with the wheel.
Set `PHOTARA_VULKAN_RUNTIME_DLL` to supply a loader at build time, or set
`PHOTARA_BUNDLE_WINDOWS_RUNTIME=OFF` to disable runtime bundling.

## UV Unwrapping and Texture Projection

```python
import numpy as np
from photara_drender import project_texture_atlas, unwrap_mesh_uv

mesh = unwrap_mesh_uv(
    positions,
    indices,
    normals,
    resolution=(2048, 2048),
    gutter=1.0,
)

baked = project_texture_atlas(
    mesh,
    images,
    world_to_clip,       # [view_count, 4, 4], row-major
    camera_positions,    # [view_count, 3]
    visibility_masks=masks,
    resolution=(2048, 2048),
    visibility_mode="ray_query",
    blend_mode="weighted_average",
)

atlas_rgba = baked.color
confidence = baked.confidence
source_view = baked.source_view
```

Input images and output atlases are float32 arrays with shape `[height, width, channels]`;
row 0 corresponds to the top of the image. When using `load_colmap_projection()`, pass
OpenCV/PIL images directly without flipping them vertically. See the
[texture baking guide](docs/texture_baking.md) for matrix and viewport conventions.

## Complete Pipeline for Scanned Meshes

`prepare_mesh_for_baking()` and C++ `photara_mesh::prepare_for_baking()` perform CGAL manifold
repair, boundary-preserving decimation, and parallel UVAtlas unwrapping in memory, without
intermediate PLY files or subprocesses. Quality presets target `high=1,000,000` triangles
(the default), `medium=500,000`, and `low=100,000`. Instant Meshes can be enabled as needed.

```python
prepared = prepare_mesh_for_baking(
    "mesh.ply",
    options=MeshPreparationOptions(quality="high"),
)
projection = load_colmap_projection(
    "sparse/0",
    "images",
    mesh_positions=prepared.mesh.positions,
)
masks = load_projection_masks("masks", projection.image_names)
baked = project_texture_atlas(
    prepared.mesh,
    projection.images,
    projection.world_to_clip,
    projection.camera_positions,
    visibility_masks=masks,
    visibility_mode="ray_query",
)
```

See `examples/prepare_mesh.py` for the preprocessing CLI and `examples/bake_colmap.py` for
the baking CLI. The [mesh preprocessing guide](docs/mesh_preprocessing.md) describes topology
guarantees and build instructions.

## Differentiable Atlas Optimization

After baking with ray queries in memory, the production pipeline immediately runs C++
`TextureRefiner` on Vulkan. By default, it performs 1,000 masked photometric Adam optimization
steps followed by seam polishing, then exports only the optimized model. Use `--no-optimize`
to export the unoptimized texture instead.

The COLMAP example uses `images_delighted` by default. Set `--texture-source rgb` to select
the original RGB images explicitly. Baking and optimization validate that they use the same
image source. Foreground masks are applied to both the optimization loss and the final masked
difference image. PyTorch `optimize_texture_atlas()` remains available as a research interface.
See [differentiable texture atlas optimization](docs/texture_atlas.md) for details.

## Current Limitations

- Texture projection is a discrete initialization step; shadow and ray visibility are not differentiated. The resulting atlas can be passed to the existing differentiable optimizers.
- The rasterizer does not yet implement homogeneous clipping across the near clipping plane.
- Differentiable mipmapped/trilinear sampling and silhouette antialiasing remain planned quality improvements.
- Buffers are currently synchronous and host-visible. Higher throughput for large scenes still requires device-local resource pools, asynchronous batching, and zero-copy PyTorch integration.
