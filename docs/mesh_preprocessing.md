# Mesh preprocessing and COLMAP baking

The production ordering is:

1. Instant Meshes creates a lower-density, boundary-aligned quad layout.
2. The quads are triangulated for interchange.
3. CGAL repairs and orients the polygon soup, duplicates combinatorially non-manifold vertices, removes degeneracies,
   stitches compatible borders, and removes isolated vertices.
4. CGAL Surface Mesh Simplification performs Lindstrom–Turk edge collapse while constraining boundary edges.
5. Microsoft UVAtlas unwraps the validated triangle manifold.
6. COLMAP cameras and photographs feed Vulkan shadow-map plus inline ray-query texture projection.

The CGAL tool guarantees a valid topological triangle manifold. A mesh may intentionally remain open, so “manifold”
does not imply watertight. Geometric self-intersection is a separate property; pass `--check-self-intersections` to the
standalone tool when a strict rejection policy is required.

On Windows with the local vcpkg CGAL installation:

```powershell
cmake -S . -B build_cgal `
  -DCMAKE_TOOLCHAIN_FILE=D:/ProgramTool/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DASDIFF_BUILD_MESH_TOOLS=ON `
  -DASDIFF_UVATLAS_USE_OPENMP=ON
cmake --build build_cgal --config Release
```

The Python API locates the installed `asdiff_render/tools/asdiff_mesh_preprocessor.exe`, an explicit
`cgal_executable`, or `ASDIFF_MESH_PREPROCESSOR`.

UVAtlas oct2025 contains one OpenMP region in chart parameterization. OpenMP is enabled by default in this project,
but initial connected-chart splitting and packing remain serial upstream work. Decimating before UVAtlas therefore has
a larger performance effect than increasing thread count alone.

COLMAP support currently accepts undistorted `PINHOLE` and `SIMPLE_PINHOLE` text models. Other camera models must first
be undistorted by COLMAP. `load_colmap_projection()` converts COLMAP world-to-camera poses into row-major clip matrices,
computes camera centers, derives near/far ranges from the mesh, and optionally linearizes sRGB photographs before blend.

CGAL Surface Mesh Simplification is GPL-3.0-or-later/commercial. The CGAL executable is an optional, separate target and
is not linked into the MIT renderer library. This separation also leaves a clean replacement boundary for another
decimation backend.
