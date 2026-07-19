"""Python interface for asdiff_render."""

from ._asdiff_render import DeviceInfo, Rasterizer, enumerate_devices

try:
    from .torch import interpolate, rasterize
except ImportError:
    interpolate = None
    rasterize = None

__all__ = ["DeviceInfo", "Rasterizer", "enumerate_devices", "interpolate", "rasterize"]
__version__ = "0.1.0"
