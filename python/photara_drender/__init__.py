"""Python interface for photara_drender."""

from ._photara_drender import (
    DeviceInfo,
    Rasterizer,
    TextureBaker,
    UvAtlasResult,
    enumerate_devices,
    has_uv_atlas_backend,
    unwrap_uv,
)
from .baking import BakedTexture, UnwrappedMesh, pad_texture_atlas, project_texture_atlas, unwrap_mesh_uv
from .colmap import (
    ColmapProjection,
    load_colmap_projection,
    load_projection_masks,
    read_colmap_text_model,
    resolve_texture_images_path,
)
from .mesh_processing import (
    MESH_QUALITY_TRIANGLE_COUNTS,
    MeshPreparationOptions,
    MeshPreparationResult,
    has_instant_meshes_backend,
    has_mesh_ops_backend,
    prepare_mesh_arrays_for_baking,
    prepare_mesh_for_baking,
    remesh_field_aligned,
    repair_and_decimate_mesh,
)

try:
    from .torch import interpolate, rasterize, texture
    from .atlas import (
        AtlasOptimizationOptions,
        TexturedRender,
        atlas_total_variation,
        masked_charbonnier_loss,
        optimize_texture_atlas,
        render_textured_mesh,
        seam_consistency_loss,
    )
except ImportError:
    AtlasOptimizationOptions = None
    TexturedRender = None
    atlas_total_variation = None
    interpolate = None
    masked_charbonnier_loss = None
    optimize_texture_atlas = None
    rasterize = None
    render_textured_mesh = None
    seam_consistency_loss = None
    texture = None

__all__ = [
    "AtlasOptimizationOptions",
    "BakedTexture",
    "ColmapProjection",
    "DeviceInfo",
    "MESH_QUALITY_TRIANGLE_COUNTS",
    "MeshPreparationOptions",
    "MeshPreparationResult",
    "has_instant_meshes_backend",
    "has_mesh_ops_backend",
    "prepare_mesh_arrays_for_baking",
    "remesh_field_aligned",
    "repair_and_decimate_mesh",
    "Rasterizer",
    "TexturedRender",
    "TextureBaker",
    "UnwrappedMesh",
    "UvAtlasResult",
    "atlas_total_variation",
    "enumerate_devices",
    "has_uv_atlas_backend",
    "interpolate",
    "load_colmap_projection",
    "load_projection_masks",
    "masked_charbonnier_loss",
    "optimize_texture_atlas",
    "pad_texture_atlas",
    "project_texture_atlas",
    "prepare_mesh_for_baking",
    "rasterize",
    "read_colmap_text_model",
    "resolve_texture_images_path",
    "render_textured_mesh",
    "seam_consistency_loss",
    "texture",
    "unwrap_mesh_uv",
    "unwrap_uv",
]
__version__ = "0.4.0"
