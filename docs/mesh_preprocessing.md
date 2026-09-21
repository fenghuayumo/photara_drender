# Mesh preprocessing and COLMAP baking

The production ordering is entirely in-memory when `PHOTARA_BUILD_MESH_TOOLS=ON`:

1. Instant Meshes optionally remeshes the input in memory via FetchContent'd Instant Meshes
   (`PHOTARA_ENABLE_INSTANT_MESHES=ON`, default when mesh tools are enabled).
2. CGAL repairs and orients the source or remeshed polygon soup in memory, duplicates
   combinatorially non-manifold vertices, removes degeneracies, stitches compatible
   borders, and removes isolated vertices.
3. CGAL Surface Mesh Simplification performs Lindstrom–Turk edge collapse while
   constraining boundary edges.
4. Microsoft UVAtlas unwraps the validated triangle manifold.
5. `TextureBaker` projects calibrated photographs into atlas space.

C++ API (GPL/commercial optional library `photara::mesh`):

```cpp
#include "photara_mesh/mesh_ops.hpp"
#include "photara_mesh/pipeline.hpp"

auto manifold = photara_mesh::repair_and_decimate(positions, indices, {.target_face_count = 1'000'000});
auto prepared = photara_mesh::prepare_for_baking(positions, indices, prepare_options);
auto baked = photara_mesh::prepare_and_bake(context, positions, indices, views, pipeline_options);
```

`photara_mesh` links CGAL and therefore stays outside the MIT `photara_drender` library.
The CLI `photara_mesh_preprocessor` is only a thin file I/O wrapper around the same
memory API.

The CGAL tool guarantees a valid topological triangle manifold. A mesh may intentionally remain open, so “manifold”
does not imply watertight. Geometric self-intersection is a separate property; pass `--check-self-intersections` to the
standalone tool when a strict rejection policy is required.

On Windows with the local vcpkg CGAL installation:

```powershell
cmake -S . -B build_cgal `
  -DCMAKE_TOOLCHAIN_FILE=D:/ProgramTool/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DPHOTARA_BUILD_MESH_TOOLS=ON `
  -DPHOTARA_ENABLE_INSTANT_MESHES=ON `
  -DPHOTARA_UVATLAS_USE_OPENMP=OFF
cmake --build build_cgal --config Release
```

The Python API locates the installed `photara_drender/tools/photara_mesh_preprocessor.exe`, an explicit
`cgal_executable`, or `PHOTARA_MESH_PREPROCESSOR`.

UVAtlas oct2025 contains an internal OpenMP region in chart parameterization. It is disabled by default because it
oversubscribes the CPU when several PCA partitions are charted concurrently. Enable `PHOTARA_UVATLAS_USE_OPENMP` only
for builds that use `parallel_partitions=1`.

The preparation defaults mirror Open3D's UVAtlas parameters: `gutter=1`, `max_stretch=1/6`, and
`quality=None`, which passes `UVATLAS_DEFAULT` and lets UVAtlas select its geodesic mode. Pipelines that require the
pre-0.4 behavior can explicitly set `atlas_gutter=4`, `atlas_max_stretch=0.3`, and `atlas_quality=False`.

Mesh density is selected with `MeshPreparationOptions.quality`: `high` (the default) targets 1,000,000 triangles,
`medium` targets 500,000, and `low` targets 100,000. `target_triangle_count` overrides the selected preset. These are
requested targets: smaller inputs are not subdivided, while meshes with many constrained boundary edges may remain
above the target. Instant Meshes is disabled by default to preserve scan detail; set `use_instant_remesh=True` when
regular retopology is preferred.

`atlas_parallel_partitions` enables the Open3D-style path: face centroids are recursively PCA-partitioned, each
partition is charted concurrently with `UVAtlasPartition`, and all charts are packed together once with
`UVAtlasPack`. Mesh preparation defaults to four partitions; direct `unwrap_uv()` calls default to one. Parallel
partitioning is faster on large meshes but may introduce additional island boundaries.

COLMAP support currently accepts undistorted `PINHOLE` and `SIMPLE_PINHOLE` text models. Other camera models must first
be undistorted by COLMAP. `load_colmap_projection()` converts COLMAP world-to-camera poses into row-major clip matrices,
computes camera centers, derives near/far ranges from the mesh, and optionally linearizes sRGB photographs before blend.

CGAL Surface Mesh Simplification is GPL-3.0-or-later/commercial. The CGAL executable is an optional, separate target and
is not linked into the MIT renderer library. This separation also leaves a clean replacement boundary for another
decimation backend.
