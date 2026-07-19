# asdiff_render

`asdiff_render` 是一个从零实现的跨平台可微光栅化库。核心使用 C++20 与 Vulkan 1.2 compute，公开接口参考
nvdiffrast 的张量语义，但不依赖 CUDA、OpenGL 上下文或厂商专有扩展。

当前版本是可验证的 portable baseline：支持 tile-binned 三角形光栅化、透视正确重心坐标、像素导数、
深度选择、面剔除、可微顶点属性插值，以及重心坐标损失到齐次裁剪空间顶点的解析反向传播。后端避免使用
浮点原子，通过角点梯度、顶点邻接表和确定性归约保证不同 Vulkan 厂商上的可执行性。

## 输出约定

`raster[..., :]` 为 `[u, v, z/w, triangle_id + 1]`，背景为全零。图像布局是 `[height, width, channels]`，
坐标原点位于左下角，与 nvdiffrast 的离屏张量约定一致。`barycentric_derivatives` 为
`[du/dx, du/dy, dv/dx, dv/dy]`。

## 构建

需要 CMake 3.24+、C++20 编译器、Vulkan SDK 1.2+ 及 DXC。所有 compute shader 使用 HLSL 编写，构建时由
DXC 的 SPIR-V 后端生成 Vulkan 1.2 字节码。Python 模块还需要 Python 3.9+、NumPy 与 pybind11。

```powershell
cmake -S . -B build -DASDIFF_BUILD_PYTHON=ON -DASDIFF_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

也可以使用 `pip install .` 构建 wheel。运行时可通过 `ASDIFF_SHADER_DIR` 指向编译后的 SPIR-V 目录。

## Python 示例

```python
import numpy as np
from asdiff_render import Rasterizer

rasterizer = Rasterizer(device_index=0)
positions = np.array([
    [-0.75, -0.75, 0.0, 1.0],
    [ 0.75, -0.75, 0.0, 1.0],
    [ 0.00,  0.75, 0.0, 1.0],
], dtype=np.float32)
indices = np.array([[0, 1, 2]], dtype=np.uint32)
raster, raster_db = rasterizer.forward(positions, indices, (512, 512))

grad_raster = np.zeros_like(raster)
grad_raster[..., 0] = 1.0
grad_positions = rasterizer.backward(positions, indices, raster, grad_raster)

colors = np.eye(3, dtype=np.float32)
image = rasterizer.interpolate_forward(colors, indices, raster)
grad_colors, grad_raster = rasterizer.interpolate_backward(
    colors, indices, raster, np.ones_like(image))
```

PyTorch 可通过 autograd 适配器直接训练；传入持久的 `Rasterizer` 可复用 Vulkan device 与 pipeline：

```python
import torch
from asdiff_render import Rasterizer, interpolate, rasterize

context = Rasterizer()
positions = torch.tensor([
    [-0.75, -0.75, 0.0, 1.0],
    [ 0.75, -0.75, 0.0, 1.0],
    [ 0.00,  0.75, 0.0, 1.0],
], dtype=torch.float32, requires_grad=True)
indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
raster, raster_db = rasterize(positions, indices, (512, 512), rasterizer=context)
colors = torch.eye(3, dtype=torch.float32, requires_grad=True)
image = interpolate(colors, raster, indices, rasterizer=context)
image.square().mean().backward()
```

该适配器通过 host staging 避免绑定特定 PyTorch ABI，CPU/CUDA 张量均可使用；高性能零拷贝互操作仍属于后续后端。

## 当前边界与演进方向

- portable baseline 采用 CPU tile binning；超大动态场景后续应切换为 GPU binning、device-local buffer pool 与异步批处理。
- 当前要求三角形三个顶点的 `w > 0`，尚未实现跨近裁剪面的齐次裁剪。
- 反向传播覆盖 `u/v`，与 nvdiffrast 一样不对离散 triangle id 求导；像素导数的二阶反传尚未开放。
- Python 提供 NumPy 原生绑定和 PyTorch autograd 适配器；Vulkan/CUDA/CPU 外部内存零拷贝互操作属于下一阶段。
- 顶点属性插值已包含前向、属性梯度与重心坐标梯度；纹理采样和轮廓抗锯齿会继续作为独立可微算子加入。
