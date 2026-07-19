# Third-party notices

Microsoft UVAtlas is optionally fetched and statically linked for UV unwrapping. UVAtlas is copyright Microsoft Corporation
and distributed under the MIT License: <https://github.com/microsoft/UVAtlas>.

The Vulkan SDK, DXC, pybind11, NumPy, and optional PyTorch dependencies retain their respective licenses.

The optional standalone `asdiff_mesh_preprocessor` uses CGAL Surface Mesh Simplification. That CGAL package is available
under GPL-3.0-or-later or a separate commercial CGAL license. It is disabled by default and is never linked into the MIT
`asdiff_render` library. When explicitly enabled, the standalone executable is included in the Python wheel.

When the CGAL tool is enabled on Windows, its GNU MP runtime is copied beside the executable. GNU MP retains its
LGPL-3.0-or-later or GPL-2.0-or-later dual license.
