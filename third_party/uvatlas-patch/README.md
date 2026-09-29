# Local UVAtlas patch

`uvatlas-perf.patch` carries the local Microsoft UVAtlas changes that make atlas generation
usable on large scan meshes. It is applied to the checkout that `third_party/uvbench` (and
`photara_drender` built with `-DPHOTARA_USE_VENDORED_UVATLAS=ON`) links against; the pinned
`oct2025` release used by a default build does not contain them.

Base: `microsoft/UVAtlas`, tag `oct2025` (verified — the patch applies cleanly to a pristine
`oct2025` tree and the library builds from the result with `UVATLAS_ENABLE_TIMING=OFF`).

## Applying

```powershell
git clone https://github.com/microsoft/UVAtlas third_party/UVAtlas
git -C third_party/UVAtlas checkout oct2025
git -C third_party/UVAtlas apply ..\uvatlas-patch\uvatlas-perf.patch
```

Then build the benchmark harness, or the product against the patched checkout:

```powershell
# benchmark harness (timing traces on)
cmake -S third_party/uvbench -B third_party/uvbench/build `
      -G "Visual Studio 17 2022" -A x64 -DUVATLAS_ENABLE_TIMING=ON
cmake --build third_party/uvbench/build --config Release --parallel

# product library + Python module linked against the patched checkout
cmake -S . -B build_uvatlas -G "Visual Studio 17 2022" -A x64 `
      -DPHOTARA_USE_VENDORED_UVATLAS=ON
cmake --build build_uvatlas --config Release --parallel
```

## Contents

Performance work: spatial-hash segment overlap tests (`segmentoverlap.h`), a single labeled
Dijkstra for the closest boundary-loop pair, hash-map edge splitting in the root chart, and
phase timing hooks (`uvatlas_timing.h`). On a 1.76M face scan mesh this turned a 2498 s
single-call unwrap into 153 s with the partitioned product path.

Two correctness fixes are also included, because both large meshes hit them:

- `CalMinPathBetweenBoundaries` now labels every reached vertex with the boundary loop its
  Dijkstra tree grew from. Comparing `pdwVertBoundaryID` alone also accepted degenerate
  "boundary vertex + interior neighbour" pairs whose joined path starts and ends on the same
  loop, so `CutChartAlongPath` cut nothing while the caller decremented the loop count and
  `CheckAndCutMultipleBoundaries` never terminated.
- `RootChart::ReorderVertices` stops a fan walk that revisits a corner the walk already
  claimed, instead of spinning forever on non-manifold or duplicate-face input. Such input
  now fails fast with `HRESULT_E_INVALID_DATA` from `BuildFullConnection`, which is the
  documented behaviour for non-manifold meshes.

## Profiling (optional)

The patch adds a `UVATLAS_ENABLE_TIMING` CMake option. With
`-DUVATLAS_ENABLE_TIMING=ON` the library emits `[uvatlas]` phase traces, which
`third_party/uvbench/summarize.py` aggregates. It is off by default and costs nothing when
disabled.
