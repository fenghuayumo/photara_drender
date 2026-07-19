"""In-memory Instant Meshes (optional), CGAL decimation, and UVAtlas preparation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

import numpy as np

from .baking import UnwrappedMesh, unwrap_mesh_uv

try:
    from ._asdiff_mesh import has_instant_meshes_backend as _has_instant_meshes_backend
    from ._asdiff_mesh import prepare_for_baking as _prepare_for_baking_cpp
    from ._asdiff_mesh import remesh_field_aligned as _remesh_field_aligned_cpp
    from ._asdiff_mesh import repair_and_decimate as _repair_and_decimate_cpp
except ImportError:
    _has_instant_meshes_backend = None
    _prepare_for_baking_cpp = None
    _remesh_field_aligned_cpp = None
    _repair_and_decimate_cpp = None


def has_instant_meshes_backend() -> bool:
    return bool(_has_instant_meshes_backend and _has_instant_meshes_backend())


def remesh_field_aligned(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    vertex_count: int = -1,
    face_count: int = -1,
    scale: float = -1.0,
    rosy: int = 4,
    posy: int = 4,
    crease_angle: float = 0.0,
    align_to_boundaries: bool = True,
    extrinsic: bool = True,
    smooth_iterations: int = 2,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Field-aligned remesh via Instant Meshes; output is always triangles."""

    if _remesh_field_aligned_cpp is None or not has_instant_meshes_backend():
        raise RuntimeError(
            "Instant Meshes remeshing requires ASDIFF_BUILD_MESH_TOOLS=ON and "
            "ASDIFF_ENABLE_INSTANT_MESHES=ON"
        )
    return _remesh_field_aligned_cpp(
        np.ascontiguousarray(positions, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.uint32),
        vertex_count,
        face_count,
        scale,
        rosy,
        posy,
        crease_angle,
        align_to_boundaries,
        extrinsic,
        smooth_iterations,
        deterministic,
    )


MeshQuality = Literal["high", "medium", "low"]
MESH_QUALITY_TRIANGLE_COUNTS: dict[str, int] = {
    "high": 1_000_000,
    "medium": 500_000,
    "low": 100_000,
}


@dataclass(frozen=True)
class MeshPreparationOptions:
    """Mesh preparation settings with Open3D-compatible UVAtlas defaults.

    ``atlas_quality=None`` passes ``UVATLAS_DEFAULT`` so UVAtlas can select
    quality or fast partitioning from the mesh size. Use ``False`` or ``True``
    to force fast or quality mode respectively.
    """

    quality: MeshQuality = "high"
    target_triangle_count: Optional[int] = None
    use_instant_remesh: bool = False
    instant_vertex_count: Optional[int] = None
    rosy: int = 4
    posy: int = 4
    align_to_boundaries: bool = True
    extrinsic: bool = True
    smooth_iterations: int = 2
    deterministic: bool = True
    atlas_resolution: tuple[int, int] = (4096, 4096)
    atlas_gutter: float = 1.0
    atlas_max_stretch: float = 1.0 / 6.0
    atlas_quality: Optional[bool] = None
    atlas_parallel_partitions: int = 4
    atlas_worker_count: int = 0


def _resolve_target_triangle_count(options: MeshPreparationOptions) -> int:
    if options.quality not in MESH_QUALITY_TRIANGLE_COUNTS:
        choices = ", ".join(MESH_QUALITY_TRIANGLE_COUNTS)
        raise ValueError(f"quality must be one of: {choices}")
    target = (
        MESH_QUALITY_TRIANGLE_COUNTS[options.quality]
        if options.target_triangle_count is None
        else options.target_triangle_count
    )
    if target < 4:
        raise ValueError("target_triangle_count must be greater than three")
    return target


class MeshPreparationResult(NamedTuple):
    mesh: UnwrappedMesh
    source_vertex_count: int
    source_face_count: int
    remeshed_vertex_count: int
    remeshed_face_count: int
    manifold_vertex_count: int
    manifold_face_count: int
    remesh_seconds: float
    cgal_seconds: float
    uv_atlas_seconds: float


def has_mesh_ops_backend() -> bool:
    return _prepare_for_baking_cpp is not None


def repair_and_decimate_mesh(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    target_face_count: int = 1_000_000,
    check_self_intersections: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair and optionally decimate a triangle mesh entirely in memory."""

    if _repair_and_decimate_cpp is None:
        raise RuntimeError(
            "in-memory mesh ops require ASDIFF_BUILD_MESH_TOOLS=ON and a CGAL-enabled build"
        )
    return _repair_and_decimate_cpp(
        np.ascontiguousarray(positions, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.uint32),
        target_face_count,
        check_self_intersections,
    )


def prepare_mesh_arrays_for_baking(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    options: MeshPreparationOptions = MeshPreparationOptions(),
) -> MeshPreparationResult:
    """Prepare mesh arrays in memory: optional remesh, CGAL decimate, UVAtlas."""

    target_triangle_count = _resolve_target_triangle_count(options)
    source_positions = np.ascontiguousarray(positions, dtype=np.float32)
    source_indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError("positions must have shape [vertex_count, 3]")
    if source_indices.ndim != 2 or source_indices.shape[1] != 3:
        raise ValueError("indices must have shape [triangle_count, 3]")

    if options.use_instant_remesh and not has_instant_meshes_backend():
        raise RuntimeError(
            "Instant Meshes remeshing requires ASDIFF_BUILD_MESH_TOOLS=ON and "
            "ASDIFF_ENABLE_INSTANT_MESHES=ON"
        )

    if _prepare_for_baking_cpp is None:
        raise RuntimeError(
            "in-memory mesh ops require ASDIFF_BUILD_MESH_TOOLS=ON and a CGAL-enabled build"
        )

    # When Instant Meshes is requested, prefer the dedicated C++ remesh API so
    # vertex/face targets from MeshPreparationOptions are honored before CGAL.
    if options.use_instant_remesh:
        if _remesh_field_aligned_cpp is None or _repair_and_decimate_cpp is None:
            raise RuntimeError("Instant Meshes C++ backend is unavailable")
        instant_vertex_count = (
            max(4, (target_triangle_count + 7) // 8)
            if options.instant_vertex_count is None
            else options.instant_vertex_count
        )
        begin = time.perf_counter()
        remeshed_positions, remeshed_faces = _remesh_field_aligned_cpp(
            source_positions,
            source_indices,
            instant_vertex_count,
            -1,
            -1.0,
            options.rosy,
            options.posy,
            0.0,
            options.align_to_boundaries,
            options.extrinsic,
            options.smooth_iterations,
            options.deterministic,
        )
        remesh_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        manifold_positions, manifold_indices = _repair_and_decimate_cpp(
            remeshed_positions, remeshed_faces, target_triangle_count, False
        )
        cgal_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        unwrapped = unwrap_mesh_uv(
            manifold_positions,
            manifold_indices,
            resolution=options.atlas_resolution,
            gutter=options.atlas_gutter,
            max_stretch=options.atlas_max_stretch,
            quality=options.atlas_quality,
            parallel_partitions=options.atlas_parallel_partitions,
            worker_count=options.atlas_worker_count,
        )
        uv_atlas_seconds = time.perf_counter() - begin
        return MeshPreparationResult(
            unwrapped,
            int(source_positions.shape[0]),
            int(source_indices.shape[0]),
            int(remeshed_positions.shape[0]),
            int(remeshed_faces.shape[0]),
            int(manifold_positions.shape[0]),
            int(manifold_indices.shape[0]),
            remesh_seconds,
            cgal_seconds,
            uv_atlas_seconds,
        )

    begin = time.perf_counter()
    (
        positions_out,
        normals_out,
        uv_out,
        indices_out,
        vertex_remap,
        face_chart_ids,
        chart_count,
        max_stretch,
        partition_count,
        remeshed_vertex_count,
        remeshed_face_count,
        manifold_vertex_count,
        manifold_face_count,
    ) = _prepare_for_baking_cpp(
        source_positions,
        source_indices,
        target_triangle_count,
        False,
        options.atlas_resolution,
        options.atlas_gutter,
        options.atlas_max_stretch,
        options.atlas_quality,
        options.atlas_parallel_partitions,
        options.atlas_worker_count,
    )
    pipeline_seconds = time.perf_counter() - begin
    unwrapped = UnwrappedMesh(
        positions_out,
        normals_out,
        uv_out,
        indices_out,
        vertex_remap,
        face_chart_ids,
        int(chart_count),
        float(max_stretch),
        int(partition_count),
    )
    return MeshPreparationResult(
        unwrapped,
        int(source_positions.shape[0]),
        int(source_indices.shape[0]),
        int(remeshed_vertex_count),
        int(remeshed_face_count),
        int(manifold_vertex_count),
        int(manifold_face_count),
        0.0,
        pipeline_seconds,
        0.0,
    )


def prepare_mesh_for_baking(
    source_mesh: Union[str, Path, tuple[np.ndarray, np.ndarray]],
    *,
    options: MeshPreparationOptions = MeshPreparationOptions(),
    cgal_executable: Optional[Union[str, Path]] = None,
    intermediate_directory: Optional[Union[str, Path]] = None,
) -> MeshPreparationResult:
    """Prepare a mesh for texture baking entirely in memory when mesh tools are built.

    ``cgal_executable`` and ``intermediate_directory`` are ignored; they remain only for
    source compatibility with older call sites.
    """

    del cgal_executable, intermediate_directory
    if isinstance(source_mesh, (str, Path)):
        try:
            import trimesh
        except ImportError as error:
            raise ImportError("loading mesh files requires trimesh from asdiff-render[baking]") from error
        source = trimesh.load(str(source_mesh), force="mesh", process=False)
        if not isinstance(source, trimesh.Trimesh) or source.faces.size == 0:
            raise ValueError("source_mesh must contain a non-empty triangle mesh")
        positions = np.ascontiguousarray(source.vertices, dtype=np.float32)
        indices = np.ascontiguousarray(source.faces, dtype=np.uint32)
    else:
        positions, indices = source_mesh
    return prepare_mesh_arrays_for_baking(positions, indices, options=options)
