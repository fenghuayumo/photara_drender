"""Python interface for asdiff_render."""

from ._asdiff_render import DeviceInfo, Rasterizer, enumerate_devices

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
    "DeviceInfo",
    "Rasterizer",
    "TexturedRender",
    "atlas_total_variation",
    "enumerate_devices",
    "interpolate",
    "masked_charbonnier_loss",
    "optimize_texture_atlas",
    "rasterize",
    "render_textured_mesh",
    "seam_consistency_loss",
    "texture",
]
__version__ = "0.2.0"
