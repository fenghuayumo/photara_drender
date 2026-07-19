"""Instant Meshes, CGAL decimation, and UVAtlas preparation pipeline."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

import numpy as np

from .baking import UnwrappedMesh, unwrap_mesh_uv


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


def _resolve_cgal_executable(executable: Optional[Union[str, Path]]) -> Path:
    candidates = []
    if executable is not None:
        candidates.append(Path(executable))
    environment_path = os.environ.get("ASDIFF_MESH_PREPROCESSOR")
    if environment_path:
        candidates.append(Path(environment_path))
    candidates.append(
        Path(__file__).resolve().parent
        / "tools"
        / ("asdiff_mesh_preprocessor.exe" if os.name == "nt" else "asdiff_mesh_preprocessor")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "CGAL mesh preprocessor was not found; build with ASDIFF_BUILD_MESH_TOOLS=ON "
        "or set ASDIFF_MESH_PREPROCESSOR"
    )


def prepare_mesh_for_baking(
    source_mesh: Union[str, Path],
    *,
    options: MeshPreparationOptions = MeshPreparationOptions(),
    cgal_executable: Optional[Union[str, Path]] = None,
    intermediate_directory: Optional[Union[str, Path]] = None,
) -> MeshPreparationResult:
    """Run topology-safe remeshing and UV unwrapping for texture baking."""

    target_triangle_count = _resolve_target_triangle_count(options)
    if options.instant_vertex_count is not None and options.instant_vertex_count < 4:
        raise ValueError("instant_vertex_count must be greater than three")
    try:
        import trimesh
    except ImportError as error:
        raise ImportError("mesh preprocessing requires trimesh from asdiff-render[baking]") from error
    pynanoinstantmeshes = None
    if options.use_instant_remesh:
        try:
            import pynanoinstantmeshes
        except ImportError as error:
            raise ImportError("Instant Meshes remeshing requires pynanoinstantmeshes") from error

    source = trimesh.load(str(source_mesh), force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or source.faces.size == 0:
        raise ValueError("source_mesh must contain a non-empty triangle mesh")
    source_positions = np.ascontiguousarray(source.vertices, dtype=np.float32)
    source_indices = np.ascontiguousarray(source.faces, dtype=np.uint32)

    owned_temporary_directory = None
    if intermediate_directory is None:
        owned_temporary_directory = tempfile.TemporaryDirectory(prefix="asdiff_mesh_")
        working_directory = Path(owned_temporary_directory.name)
    else:
        working_directory = Path(intermediate_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
    instant_path = working_directory / "instant_remesh.ply"
    source_path = working_directory / "source_mesh.ply"
    manifold_path = working_directory / "manifold_decimated.ply"
    if options.use_instant_remesh:
        instant_vertex_count = (
            max(4, (target_triangle_count + 7) // 8)
            if options.instant_vertex_count is None
            else options.instant_vertex_count
        )
        begin = time.perf_counter()
        assert pynanoinstantmeshes is not None
        remeshed_positions, remeshed_faces = pynanoinstantmeshes.remesh(
            source_positions,
            source_indices,
            vertex_count=instant_vertex_count,
            rosy=options.rosy,
            posy=options.posy,
            align_to_boundaries=options.align_to_boundaries,
            extrinsic=options.extrinsic,
            smooth_iter=options.smooth_iterations,
            deterministic=options.deterministic,
        )
        remesh_seconds = time.perf_counter() - begin
        remeshed = trimesh.Trimesh(vertices=remeshed_positions, faces=remeshed_faces, process=False)
        remeshed.export(instant_path)
        cgal_input_path = instant_path
        remeshed_vertex_count = int(np.asarray(remeshed_positions).shape[0])
        remeshed_face_count = int(remeshed.faces.shape[0])
    else:
        remesh_seconds = 0.0
        source.export(source_path)
        cgal_input_path = source_path
        remeshed_vertex_count = int(source_positions.shape[0])
        remeshed_face_count = int(source_indices.shape[0])

    executable = _resolve_cgal_executable(cgal_executable)
    begin = time.perf_counter()
    process = subprocess.run(
        [str(executable), str(cgal_input_path), str(manifold_path), str(target_triangle_count)],
        check=False,
        capture_output=True,
        text=True,
    )
    cgal_seconds = time.perf_counter() - begin
    if process.returncode != 0:
        raise RuntimeError(
            f"CGAL preprocessing failed with exit code {process.returncode}:\n{process.stdout}\n{process.stderr}"
        )
    manifold = trimesh.load(str(manifold_path), force="mesh", process=False)
    manifold_positions = np.ascontiguousarray(manifold.vertices, dtype=np.float32)
    manifold_indices = np.ascontiguousarray(manifold.faces, dtype=np.uint32)

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
    result = MeshPreparationResult(
        unwrapped,
        int(source_positions.shape[0]),
        int(source_indices.shape[0]),
        remeshed_vertex_count,
        remeshed_face_count,
        int(manifold_positions.shape[0]),
        int(manifold_indices.shape[0]),
        remesh_seconds,
        cgal_seconds,
        uv_atlas_seconds,
    )
    if owned_temporary_directory is not None:
        owned_temporary_directory.cleanup()
    return result
