"""COLMAP text-model loading for calibrated texture projection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


class ColmapCamera(NamedTuple):
    camera_id: int
    model: str
    width: int
    height: int
    parameters: np.ndarray


class ColmapImage(NamedTuple):
    image_id: int
    camera_id: int
    name: str
    rotation: np.ndarray
    translation: np.ndarray


class ColmapProjection(NamedTuple):
    images: Tuple[np.ndarray, ...]
    world_to_clip: np.ndarray
    camera_positions: np.ndarray
    image_names: Tuple[str, ...]


def _data_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip() and not line.startswith("#")]


def _quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_colmap_text_model(sparse_path: Union[str, Path]) -> tuple[Dict[int, ColmapCamera], Tuple[ColmapImage, ...]]:
    """Read COLMAP ``cameras.txt`` and ``images.txt`` without points2D data."""

    sparse_path = Path(sparse_path)
    cameras: Dict[int, ColmapCamera] = {}
    for line in _data_lines(sparse_path / "cameras.txt"):
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = ColmapCamera(
            camera_id,
            fields[1],
            int(fields[2]),
            int(fields[3]),
            np.asarray(fields[4:], dtype=np.float64),
        )

    lines = _data_lines(sparse_path / "images.txt")
    if len(lines) % 2 != 0:
        raise ValueError("COLMAP images.txt must contain one points2D line after each image line")
    colmap_images = []
    for image_line in lines[::2]:
        fields = image_line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"invalid COLMAP image record: {image_line[:120]}")
        quaternion = np.asarray(fields[1:5], dtype=np.float64)
        quaternion_length = np.linalg.norm(quaternion)
        if not np.isfinite(quaternion_length) or quaternion_length <= 0.0:
            raise ValueError(f"image {fields[0]} contains an invalid quaternion")
        quaternion /= quaternion_length
        colmap_images.append(
            ColmapImage(
                int(fields[0]),
                int(fields[8]),
                fields[9],
                _quaternion_to_rotation(quaternion),
                np.asarray(fields[5:8], dtype=np.float64),
            )
        )
    colmap_images.sort(key=lambda value: value.image_id)
    return cameras, tuple(colmap_images)


def _pinhole_parameters(camera: ColmapCamera) -> tuple[float, float, float, float]:
    if camera.model == "PINHOLE" and camera.parameters.size == 4:
        return tuple(float(value) for value in camera.parameters)  # type: ignore[return-value]
    if camera.model == "SIMPLE_PINHOLE" and camera.parameters.size == 3:
        focal, cx, cy = camera.parameters
        return float(focal), float(focal), float(cx), float(cy)
    raise ValueError(
        f"camera {camera.camera_id} uses {camera.model}; undistort it to PINHOLE or SIMPLE_PINHOLE before projection"
    )


def _depth_range(
    image: ColmapImage,
    mesh_positions: Optional[np.ndarray],
    near: Optional[float],
    far: Optional[float],
) -> tuple[float, float]:
    if near is not None and far is not None:
        if not 0.0 < near < far:
            raise ValueError("near and far must satisfy 0 < near < far")
        return float(near), float(far)
    if mesh_positions is None:
        resolved_near = 1e-3 if near is None else float(near)
        resolved_far = 1e3 if far is None else float(far)
        if not 0.0 < resolved_near < resolved_far:
            raise ValueError("near and far must satisfy 0 < near < far")
        return resolved_near, resolved_far
    depths = mesh_positions @ image.rotation[2] + image.translation[2]
    positive_depths = depths[depths > 1e-6]
    if positive_depths.size == 0:
        raise ValueError(f"mesh is entirely behind COLMAP camera for {image.name}")
    resolved_near = max(1e-4, float(positive_depths.min()) * 0.5) if near is None else float(near)
    resolved_far = float(positive_depths.max()) * 1.5 if far is None else float(far)
    if not 0.0 < resolved_near < resolved_far:
        raise ValueError("near and far must satisfy 0 < near < far")
    return resolved_near, resolved_far


def _world_to_clip(
    camera: ColmapCamera,
    image: ColmapImage,
    near: float,
    far: float,
) -> np.ndarray:
    fx, fy, cx, cy = _pinhole_parameters(camera)
    rotation = image.rotation
    translation = image.translation
    x_offset = 2.0 * cx / camera.width - 1.0
    y_offset = 2.0 * cy / camera.height - 1.0
    depth_scale = (far + near) / (far - near)
    depth_offset = -2.0 * far * near / (far - near)
    matrix = np.empty((4, 4), dtype=np.float64)
    matrix[0, :3] = (2.0 * fx / camera.width) * rotation[0] + x_offset * rotation[2]
    matrix[0, 3] = (2.0 * fx / camera.width) * translation[0] + x_offset * translation[2]
    matrix[1, :3] = (2.0 * fy / camera.height) * rotation[1] + y_offset * rotation[2]
    matrix[1, 3] = (2.0 * fy / camera.height) * translation[1] + y_offset * translation[2]
    matrix[2, :3] = depth_scale * rotation[2]
    matrix[2, 3] = depth_scale * translation[2] + depth_offset
    matrix[3, :3] = rotation[2]
    matrix[3, 3] = translation[2]
    return matrix.astype(np.float32)


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_colmap_projection(
    sparse_path: Union[str, Path],
    images_path: Union[str, Path],
    *,
    mesh_positions: Optional[np.ndarray] = None,
    image_names: Optional[Sequence[str]] = None,
    image_stride: int = 1,
    max_images: Optional[int] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
    linearize_srgb: bool = True,
) -> ColmapProjection:
    """Load calibrated photographs and construct this renderer's row-major clip matrices."""

    if image_stride < 1:
        raise ValueError("image_stride must be positive")
    cameras, colmap_images = read_colmap_text_model(sparse_path)
    selected_names = None if image_names is None else set(image_names)
    selected = [image for image in colmap_images if selected_names is None or image.name in selected_names]
    if selected_names is not None:
        missing = selected_names.difference(image.name for image in selected)
        if missing:
            raise FileNotFoundError(f"images are absent from COLMAP model: {sorted(missing)}")
    selected = selected[::image_stride]
    if max_images is not None:
        selected = selected[:max_images]
    if not selected:
        raise ValueError("no COLMAP images matched the selection")

    mesh = None
    if mesh_positions is not None:
        mesh = np.asarray(mesh_positions, dtype=np.float64)
        if mesh.ndim != 2 or mesh.shape[1] != 3:
            raise ValueError("mesh_positions must have shape [vertex_count, 3]")
        if mesh.shape[0] == 0 or not np.isfinite(mesh).all():
            raise ValueError("mesh_positions must be non-empty and finite")

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required to load COLMAP photographs; install asdiff-render[baking]") from error

    images_root = Path(images_path)
    photographs = []
    matrices = []
    positions = []
    names = []
    for image in selected:
        camera = cameras.get(image.camera_id)
        if camera is None:
            raise ValueError(f"image {image.name} references missing camera {image.camera_id}")
        image_path = images_root / image.name
        with Image.open(image_path) as source:
            photograph = np.asarray(source.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        if photograph.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"{image.name} has size {photograph.shape[1]}x{photograph.shape[0]}, expected {camera.width}x{camera.height}"
            )
        if linearize_srgb:
            photograph = _srgb_to_linear(photograph)
        resolved_near, resolved_far = _depth_range(image, mesh, near, far)
        photographs.append(np.ascontiguousarray(photograph))
        matrices.append(_world_to_clip(camera, image, resolved_near, resolved_far))
        positions.append((-image.rotation.T @ image.translation).astype(np.float32))
        names.append(image.name)
    return ColmapProjection(
        tuple(photographs),
        np.ascontiguousarray(matrices, dtype=np.float32),
        np.ascontiguousarray(positions, dtype=np.float32),
        tuple(names),
    )


def load_projection_masks(
    masks_path: Union[str, Path],
    image_names: Sequence[str],
) -> Tuple[np.ndarray, ...]:
    """Load grayscale masks in the same order as a :class:`ColmapProjection`."""

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required to load projection masks; install asdiff-render[baking]") from error
    masks_root = Path(masks_path)
    masks = []
    for name in image_names:
        with Image.open(masks_root / name) as source:
            mask = np.asarray(source.convert("L"), dtype=np.float32) / np.float32(255.0)
        masks.append(np.ascontiguousarray(mask))
    return tuple(masks)
