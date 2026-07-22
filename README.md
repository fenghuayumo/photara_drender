# asdiff_render

`asdiff_render` 是使用 C++20、Vulkan 1.2 compute 和 HLSL 从零实现的跨平台可微渲染与纹理烘焙库。
它不依赖 CUDA 或 OpenGL 上下文，提供 C++、NumPy 和 PyTorch 接口。

当前 `0.4.0` 版本包含：

- tile-binned 可微三角形光栅化、透视正确重心坐标和像素导数；
- 顶点属性插值及其反向传播；
- 可微双线性纹理采样、atlas texel/UV 梯度；
- Microsoft UVAtlas 展 UV，保留 vertex remap、face chart id 和 stretch；
- atlas 空间位置/法线展开；
- 多相机 texture projection、PCF shadow map 可见性和置信度融合；
- `VK_KHR_ray_query` BLAS/TLAS 精确遮挡校验，以及不支持设备上的 shadow-map 降级；
- 多视图纹理图集可微优化、chart-aware TV 和 seam consistency loss。

项目定义的函数、变量、文件和命名空间使用 `snake_case`，类型使用 `PascalCase`，常量使用
`UPPER_SNAKE_CASE`。所有 shader 均使用 HLSL，经 DXC 编译为 Vulkan SPIR-V。

## 构建

需要 CMake 3.24+、C++20 编译器、Vulkan SDK 1.2+ 和 DXC。Windows 默认下载并静态链接固定版本的
Microsoft UVAtlas；也可以使用系统安装包或关闭该后端。

```powershell
cmake -S . -B build -DASDIFF_BUILD_PYTHON=ON -DASDIFF_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

相关选项：

- `ASDIFF_ENABLE_UVATLAS=OFF`：只构建渲染和投影模块；
- `ASDIFF_FETCH_UVATLAS=OFF`：要求环境中存在 `uvatlas` CMake package；
- `ASDIFF_UVATLAS_USE_OPENMP=ON`：启用 UVAtlas 上游的 OpenMP chart 参数化；
- `ASDIFF_BUILD_MESH_TOOLS=ON`：构建使用 CGAL 的流形修复与 Decimation 工具；
- `ASDIFF_ENABLE_VALIDATION=ON`：启用 Vulkan validation layer。

Python wheel 可以使用 `python -m pip wheel .` 构建。

## UV 展开与纹理投影

```python
import numpy as np
from asdiff_render import project_texture_atlas, unwrap_mesh_uv

mesh = unwrap_mesh_uv(
    positions,
    indices,
    normals,
    resolution=(2048, 2048),
    gutter=1.0,
)

baked = project_texture_atlas(
    mesh,
    images,
    world_to_clip,       # [view_count, 4, 4]，row-major
    camera_positions,    # [view_count, 3]
    visibility_masks=masks,
    resolution=(2048, 2048),
    visibility_mode="ray_query",
    blend_mode="weighted_average",
)

atlas_rgba = baked.color
confidence = baked.confidence
source_view = baked.source_view
```

输入照片和输出 atlas 均为 `[height, width, channels]` float32，数组第 0 行对应图像顶部。使用
`load_colmap_projection()` 时可直接传入 OpenCV/PIL 图像，无需垂直翻转。详细矩阵、viewport 和 AIHoloImager 接入约定见
[纹理烘焙指南](docs/texture_baking.md)。

## 扫描 mesh 完整流水线

`prepare_mesh_for_baking()` / C++ `asdiff_mesh::prepare_for_baking()` 在内存中完成
CGAL 流形修复、保边界 Decimation 和并行 UVAtlas，不再经中间 PLY/子进程。质量档位为
`high=100万`（默认）、`medium=50万`、`low=10万` 面；Instant Meshes 可按需开启。

```python
prepared = prepare_mesh_for_baking(
    "mesh.ply",
    options=MeshPreparationOptions(quality="high", atlas_parallel_partitions=4),
)
projection = load_colmap_projection(
    "sparse/0",
    "images",
    mesh_positions=prepared.mesh.positions,
)
masks = load_projection_masks("masks", projection.image_names)
baked = project_texture_atlas(
    prepared.mesh,
    projection.images,
    projection.world_to_clip,
    projection.camera_positions,
    visibility_masks=masks,
    visibility_mode="ray_query",
)
```

预处理命令行见 `examples/prepare_mesh.py`，烘焙命令行见 `examples/bake_colmap.py`，拓扑保证和构建方式见
[mesh 预处理说明](docs/mesh_preprocessing.md)。

## 可微 atlas 优化

初始投影 atlas 可以继续传给 `render_textured_mesh()` 和 `optimize_texture_atlas()`，通过照片损失反向优化
atlas texel。详见 [可微纹理图集优化](docs/texture_atlas.md)。

## 当前边界

- 纹理投影本身是离散初始化过程，不对 shadow/ray visibility 求导；生成的 atlas 可进入现有可微优化器。
- 光栅器尚未实现跨近裁剪面的齐次裁剪。
- mipmapped/trilinear 可微采样和轮廓抗锯齿仍是后续质量模块。
- 当前使用同步 host-visible buffer；大场景吞吐还需要 device-local 资源池、异步批处理和 PyTorch 零拷贝。
