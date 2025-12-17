# PyMetal API Reference

This document provides a comprehensive reference for the PyMetal API.

## Table of Contents

- [Module Organization](#module-organization)
- [Device Management](#device-management)
- [Command Submission](#command-submission)
- [Memory Resources](#memory-resources)
- [Shader Compilation](#shader-compilation)
- [Compute Pipeline](#compute-pipeline)
- [Graphics Pipeline](#graphics-pipeline)
- [Render Pass](#render-pass)
- [Sampling and Textures](#sampling-and-textures)
- [Depth/Stencil](#depthstencil)
- [Advanced Features](#advanced-features)
- [Shader Preprocessing](#shader-preprocessing)
- [Enumerations](#enumerations)
- [Constants](#constants)
- [Exceptions](#exceptions)

---

## Module Organization

PyMetal provides organized submodules for cleaner imports while maintaining backward compatibility at the top level.

```python
import pymetal as pm

# Top-level access (all symbols available)
device = pm.create_system_default_device()
pm.StorageMode.Shared

# Submodule access
from pymetal import enums, types, compute, graphics, advanced, shader

# Submodules:
# - pymetal.enums    - All enumeration types
# - pymetal.types    - Utility types (Origin, Size, Range, ClearColor)
# - pymetal.compute  - Compute pipeline classes
# - pymetal.graphics - Graphics/render pipeline classes
# - pymetal.advanced - Advanced features (events, heaps, indirect commands)
# - pymetal.shader   - Shader preprocessing utilities
```

---

## Device Management

### `create_system_default_device()`

Get the system's default Metal device.

```python
device = pm.create_system_default_device()
```

**Returns:** `Device` - The default Metal GPU device.

### `copy_all_devices()`

Enumerate all available Metal devices on the system.

```python
devices = pm.copy_all_devices()
for d in devices:
    print(f"{d.name}: low_power={d.is_low_power}")
```

**Returns:** `list[Device]` - All Metal-capable GPUs.

### `Device`

Represents a GPU device capable of executing Metal commands.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable GPU name (e.g., "Apple M1 Pro") |
| `max_threads_per_threadgroup` | `tuple[int, int, int]` | Maximum threads per threadgroup (w, h, d) |
| `is_low_power` | `bool` | True for integrated/low-power GPUs |
| `is_headless` | `bool` | True if no display attached |
| `is_removable` | `bool` | True for external GPUs (eGPU) |
| `has_unified_memory` | `bool` | True if CPU/GPU share memory (Apple Silicon) |
| `registry_id` | `int` | Unique identifier in IORegistry |
| `recommended_max_working_set_size` | `int` | Recommended max memory in bytes |
| `max_buffer_length` | `int` | Maximum buffer size in bytes |

**Methods:**

```python
# Command queue
queue = device.new_command_queue()

# Buffers
buffer = device.new_buffer(length, options)
buffer = device.new_buffer_with_data(data, length, options)

# Shaders
library = device.new_library_with_source(source_string)

# Compute pipeline
pipeline = device.new_compute_pipeline_state(function)

# Textures
texture = device.new_texture(descriptor)

# Samplers
sampler = device.new_sampler_state(descriptor)

# Render pipeline
pipeline = device.new_render_pipeline_state(descriptor)

# Depth/stencil
state = device.new_depth_stencil_state(descriptor)

# Memory management
heap = device.new_heap(descriptor)
fence = device.new_fence()

# Events
event = device.new_event()
shared_event = device.new_shared_event()

# Indirect commands
icb = device.new_indirect_command_buffer(descriptor, max_count, options)

# Binary archives
archive = device.new_binary_archive(descriptor)
```

**Thread Safety:** Device methods are thread-safe. Multiple threads can create resources concurrently.

---

## Command Submission

### `CommandQueue`

A queue for submitting command buffers to the GPU.

```python
queue = device.new_command_queue()
cmd_buffer = queue.command_buffer()
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `device` | `Device` | The device that created this queue |
| `label` | `str` | Debug label (read/write) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `command_buffer()` | `CommandBuffer` | Create a new command buffer |

**Thread Safety:** Command queues are thread-safe. Multiple threads can create command buffers from the same queue.

### `CommandBuffer`

Stores encoded commands to submit to the GPU.

```python
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.compute_command_encoder()
# ... encode work ...
encoder.end_encoding()
cmd_buffer.commit()
cmd_buffer.wait_until_completed()
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `device` | `Device` | The device that created this buffer |
| `status` | `CommandBufferStatus` | Current status |
| `label` | `str` | Debug label (read/write) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute_command_encoder()` | `ComputeCommandEncoder` | Create compute encoder |
| `render_command_encoder(render_pass)` | `RenderCommandEncoder` | Create render encoder |
| `blit_command_encoder()` | `BlitCommandEncoder` | Create blit encoder |
| `commit()` | `None` | Submit buffer for execution |
| `wait_until_completed()` | `None` | Block until GPU finishes (releases GIL) |
| `wait_until_scheduled()` | `None` | Block until scheduled (releases GIL) |
| `encode_signal_event(event, value)` | `None` | Signal an event |
| `encode_wait_for_event(event, value)` | `None` | Wait for an event |

**Thread Safety:** Command buffers are NOT thread-safe. Use one command buffer per thread.

### `CommandBufferStatus`

```python
pm.CommandBufferStatus.NotEnqueued
pm.CommandBufferStatus.Enqueued
pm.CommandBufferStatus.Committed
pm.CommandBufferStatus.Scheduled
pm.CommandBufferStatus.Completed
pm.CommandBufferStatus.Error
```

---

## Memory Resources

### `Buffer`

GPU memory buffer for storing data.

```python
# Create buffer
buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

# Access contents (zero-copy with NumPy)
import numpy as np
data = np.frombuffer(buffer.contents(), dtype=np.float32)
data[:] = [1.0, 2.0, 3.0, 4.0]
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `length` | `int` | Buffer size in bytes |
| `device` | `Device` | The device that created this buffer |
| `label` | `str` | Debug label (read/write) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `contents()` | `memoryview` | Get buffer contents for CPU access |

**Thread Safety:** Buffer object is thread-safe, but contents access must be synchronized.

### `Texture`

GPU texture for image data.

```python
desc = pm.TextureDescriptor.texture2d_descriptor(
    pm.PixelFormat.RGBA8Unorm, 512, 512, False
)
texture = device.new_texture(desc)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `width` | `int` | Texture width |
| `height` | `int` | Texture height |
| `depth` | `int` | Texture depth |
| `mipmap_level_count` | `int` | Number of mipmap levels |
| `array_length` | `int` | Number of array elements |
| `sample_count` | `int` | MSAA sample count |
| `texture_type` | `TextureType` | Texture dimensionality |
| `pixel_format` | `PixelFormat` | Pixel format |
| `label` | `str` | Debug label (read/write) |

### `TextureDescriptor`

Configuration for creating textures.

```python
# 2D texture
desc = pm.TextureDescriptor.texture2d_descriptor(
    pixel_format,  # PixelFormat
    width,         # int
    height,        # int
    mipmapped      # bool
)

# Cube texture
desc = pm.TextureDescriptor.texturecube_descriptor(
    pixel_format, size, mipmapped
)
```

**Properties (read/write):**

| Property | Type |
|----------|------|
| `texture_type` | `TextureType` |
| `pixel_format` | `PixelFormat` |
| `width` | `int` |
| `height` | `int` |
| `depth` | `int` |
| `mipmap_level_count` | `int` |
| `sample_count` | `int` |
| `array_length` | `int` |
| `storage_mode` | `StorageMode` |
| `cpu_cache_mode` | `CPUCacheMode` |
| `usage` | `int` |

---

## Shader Compilation

### `Library`

Container for compiled Metal functions.

```python
shader_source = """
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(device float* data [[buffer(0)]],
                      uint id [[thread_position_in_grid]]) {
    data[id] *= 2.0;
}
"""

library = device.new_library_with_source(shader_source)
function = library.new_function("my_kernel")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `new_function(name)` | `Function` | Get a function by name |
| `function_names` | `list[str]` | Get all function names |

**Raises:** `CompileError` if shader compilation fails.

### `Function`

A compiled Metal shader function.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Function name |
| `function_type` | `FunctionType` | Type (vertex/fragment/kernel) |
| `device` | `Device` | The device that created this function |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `new_argument_encoder(index)` | `ArgumentEncoder` | Create argument encoder |

### `FunctionType`

```python
pm.FunctionType.Vertex
pm.FunctionType.Fragment
pm.FunctionType.Kernel
```

---

## Compute Pipeline

### `ComputePipelineState`

Compiled compute pipeline ready for execution.

```python
pipeline = device.new_compute_pipeline_state(function)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `device` | `Device` | The device that created this pipeline |
| `max_total_threads_per_threadgroup` | `int` | Max threads per threadgroup |
| `thread_execution_width` | `int` | SIMD width |
| `label` | `str` | Debug label |

**Raises:** `PipelineError` if pipeline creation fails.

### `ComputeCommandEncoder`

Encodes compute commands into a command buffer.

```python
encoder = cmd_buffer.compute_command_encoder()
encoder.set_compute_pipeline_state(pipeline)
encoder.set_buffer(buffer, 0, 0)  # buffer, offset, index
encoder.dispatch_threadgroups(
    groups_x, groups_y, groups_z,    # Grid size
    threads_x, threads_y, threads_z  # Threads per group
)
encoder.end_encoding()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `set_compute_pipeline_state(pipeline)` | Set the pipeline |
| `set_buffer(buffer, offset, index)` | Bind buffer (index 0-31) |
| `set_texture(texture, index)` | Bind texture (index 0-31) |
| `set_sampler_state(sampler, index)` | Bind sampler (index 0-15) |
| `set_bytes(data, length, index)` | Set inline bytes |
| `set_threadgroup_memory_length(length, index)` | Set threadgroup memory |
| `dispatch_threadgroups(gx, gy, gz, tx, ty, tz)` | Dispatch work |
| `dispatch_threads(tx, ty, tz, tpg_x, tpg_y, tpg_z)` | Dispatch with thread count |
| `update_fence(fence)` | Update fence after work |
| `wait_for_fence(fence)` | Wait for fence before work |
| `end_encoding()` | Finish encoding |

**Validation:** Buffer index must be 0-31, texture index 0-31, sampler index 0-15. Threadgroup dimensions must be > 0.

**Thread Safety:** Encoders are NOT thread-safe. Use one encoder per thread.

---

## Graphics Pipeline

### `RenderPipelineState`

Compiled render pipeline for graphics operations.

```python
desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
desc.vertex_function = vertex_func
desc.fragment_function = fragment_func
desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm

pipeline = device.new_render_pipeline_state(desc)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `device` | `Device` | The device that created this pipeline |
| `label` | `str` | Debug label |

### `RenderPipelineDescriptor`

Configuration for creating render pipelines.

```python
desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
```

**Properties (read/write):**

| Property | Type |
|----------|------|
| `vertex_function` | `Function` |
| `fragment_function` | `Function` |
| `vertex_descriptor` | `VertexDescriptor` |
| `depth_attachment_pixel_format` | `PixelFormat` |
| `stencil_attachment_pixel_format` | `PixelFormat` |

**Methods:**

| Method | Returns |
|--------|---------|
| `color_attachment(index)` | `RenderPipelineColorAttachmentDescriptor` |

### `RenderPipelineColorAttachmentDescriptor`

**Properties (read/write):**

| Property | Type |
|----------|------|
| `pixel_format` | `PixelFormat` |
| `blending_enabled` | `bool` |
| `source_rgb_blend_factor` | `BlendFactor` |
| `destination_rgb_blend_factor` | `BlendFactor` |
| `rgb_blend_operation` | `BlendOperation` |
| `source_alpha_blend_factor` | `BlendFactor` |
| `destination_alpha_blend_factor` | `BlendFactor` |
| `alpha_blend_operation` | `BlendOperation` |
| `write_mask` | `int` |

### `RenderCommandEncoder`

Encodes rendering commands.

```python
encoder = cmd_buffer.render_command_encoder(render_pass)
encoder.set_render_pipeline_state(pipeline)
encoder.set_vertex_buffer(buffer, 0, 0)
encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
encoder.end_encoding()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `set_render_pipeline_state(pipeline)` | Set render pipeline |
| `set_vertex_buffer(buffer, offset, index)` | Bind vertex buffer |
| `set_fragment_buffer(buffer, offset, index)` | Bind fragment buffer |
| `set_vertex_texture(texture, index)` | Bind vertex texture |
| `set_fragment_texture(texture, index)` | Bind fragment texture |
| `set_vertex_sampler_state(sampler, index)` | Bind vertex sampler |
| `set_fragment_sampler_state(sampler, index)` | Bind fragment sampler |
| `set_cull_mode(mode)` | Set face culling |
| `set_front_facing_winding(winding)` | Set front face winding |
| `set_depth_stencil_state(state)` | Set depth/stencil state |
| `set_viewport(x, y, w, h, near, far)` | Set viewport |
| `set_scissor_rect(x, y, w, h)` | Set scissor rectangle |
| `draw_primitives(type, start, count)` | Draw primitives |
| `draw_primitives_instanced(type, start, count, instances)` | Instanced draw |
| `draw_indexed_primitives(type, count, index_type, index_buffer, offset)` | Indexed draw |
| `end_encoding()` | Finish encoding |

### `VertexDescriptor`

Describes vertex buffer layout.

```python
desc = pm.VertexDescriptor.vertex_descriptor()
desc.attribute(0).format = pm.VertexFormat.Float3
desc.attribute(0).offset = 0
desc.attribute(0).buffer_index = 0
desc.layout(0).stride = 12
desc.layout(0).step_function = pm.VertexStepFunction.PerVertex
```

**Methods:**

| Method | Returns |
|--------|---------|
| `attribute(index)` | `VertexAttributeDescriptor` |
| `layout(index)` | `VertexBufferLayoutDescriptor` |

---

## Render Pass

### `RenderPassDescriptor`

Configuration for a render pass.

```python
render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
color_att = render_pass.color_attachment(0)
color_att.texture = texture
color_att.load_action = pm.LoadAction.Clear
color_att.store_action = pm.StoreAction.Store
color_att.clear_color = pm.ClearColor(0.0, 0.0, 0.0, 1.0)
```

**Methods:**

| Method | Returns |
|--------|---------|
| `color_attachment(index)` | `RenderPassColorAttachmentDescriptor` |
| `depth_attachment` | `RenderPassDepthAttachmentDescriptor` |

### `RenderPassColorAttachmentDescriptor`

**Properties (read/write):**

| Property | Type |
|----------|------|
| `texture` | `Texture` |
| `load_action` | `LoadAction` |
| `store_action` | `StoreAction` |
| `clear_color` | `ClearColor` |

### `ClearColor`

```python
clear = pm.ClearColor(red, green, blue, alpha)  # 0.0-1.0
```

---

## Sampling and Textures

### `SamplerState`

Configured texture sampler.

```python
desc = pm.SamplerDescriptor.sampler_descriptor()
desc.min_filter = pm.SamplerMinMagFilter.Linear
desc.mag_filter = pm.SamplerMinMagFilter.Linear
sampler = device.new_sampler_state(desc)
```

### `SamplerDescriptor`

**Properties (read/write):**

| Property | Type |
|----------|------|
| `min_filter` | `SamplerMinMagFilter` |
| `mag_filter` | `SamplerMinMagFilter` |
| `mip_filter` | `SamplerMipFilter` |
| `s_address_mode` | `SamplerAddressMode` |
| `t_address_mode` | `SamplerAddressMode` |
| `r_address_mode` | `SamplerAddressMode` |
| `max_anisotropy` | `int` |
| `compare_function` | `CompareFunction` |
| `lod_min_clamp` | `float` |
| `lod_max_clamp` | `float` |
| `normalized_coordinates` | `bool` |

---

## Depth/Stencil

### `DepthStencilState`

```python
desc = pm.DepthStencilDescriptor.depth_stencil_descriptor()
desc.depth_compare_function = pm.CompareFunction.Less
desc.depth_write_enabled = True
state = device.new_depth_stencil_state(desc)
```

### `DepthStencilDescriptor`

**Properties (read/write):**

| Property | Type |
|----------|------|
| `depth_compare_function` | `CompareFunction` |
| `depth_write_enabled` | `bool` |
| `front_face_stencil` | `StencilDescriptor` |
| `back_face_stencil` | `StencilDescriptor` |

### `StencilDescriptor`

**Properties (read/write):**

| Property | Type |
|----------|------|
| `stencil_compare_function` | `CompareFunction` |
| `stencil_failure_operation` | `StencilOperation` |
| `depth_failure_operation` | `StencilOperation` |
| `depth_stencil_pass_operation` | `StencilOperation` |
| `read_mask` | `int` |
| `write_mask` | `int` |

---

## Advanced Features

### Heap

Memory heap for resource suballocation.

```python
desc = pm.HeapDescriptor.heap_descriptor()
desc.size = 1024 * 1024  # 1MB
desc.storage_mode = pm.StorageMode.Shared
heap = device.new_heap(desc)

buffer = heap.new_buffer(4096, pm.ResourceStorageModeShared)
```

**Properties:**

| Property | Type |
|----------|------|
| `size` | `int` |
| `used_size` | `int` |
| `current_allocated_size` | `int` |
| `label` | `str` |

**Methods:**

| Method | Returns |
|--------|---------|
| `max_available_size(alignment)` | `int` |
| `new_buffer(length, options)` | `Buffer` |
| `new_texture(descriptor)` | `Texture` |

### Fence

Synchronization between encoders.

```python
fence = device.new_fence()

# In encoder 1
encoder1.update_fence(fence)

# In encoder 2 (after encoder 1)
encoder2.wait_for_fence(fence)
```

### Event / SharedEvent

Fine-grained GPU synchronization.

```python
event = device.new_event()
shared_event = device.new_shared_event()
shared_event.signaled_value = 1

cmd_buffer.encode_signal_event(shared_event, 2)
cmd_buffer.encode_wait_for_event(shared_event, 2)
```

### BlitCommandEncoder

Memory copy operations.

```python
encoder = cmd_buffer.blit_command_encoder()
encoder.copy_from_buffer(src, 0, dst, 0, length)
encoder.fill_buffer(buffer, range, value)
encoder.generate_mipmaps(texture)
encoder.end_encoding()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `copy_from_buffer(src, src_offset, dst, dst_offset, size)` | Copy buffer to buffer |
| `copy_from_texture(...)` | Copy texture to texture |
| `copy_from_buffer_to_texture(...)` | Copy buffer to texture |
| `copy_from_texture_to_buffer(...)` | Copy texture to buffer |
| `fill_buffer(buffer, range, value)` | Fill buffer with byte value |
| `generate_mipmaps(texture)` | Generate texture mipmaps |
| `synchronize_resource(resource)` | Synchronize managed resource |

### IndirectCommandBuffer

GPU-driven rendering commands.

```python
desc = pm.IndirectCommandBufferDescriptor.indirect_command_buffer_descriptor()
desc.command_types = pm.IndirectCommandTypeDraw
desc.max_vertex_buffer_bind_count = 4
icb = device.new_indirect_command_buffer(desc, 100, pm.ResourceStorageModeShared)
```

### BinaryArchive

Pipeline caching for faster load times.

```python
desc = pm.BinaryArchiveDescriptor.binary_archive_descriptor()
desc.url = "/path/to/cache.metallib"
archive = device.new_binary_archive(desc)
```

### CaptureScope / CaptureManager

GPU debugging integration with Xcode.

```python
manager = pm.shared_capture_manager()
scope = manager.new_capture_scope_with_device(device)
scope.label = "Debug Capture"

scope.begin_scope()
# ... GPU work ...
scope.end_scope()
```

---

## Shader Preprocessing

The `pymetal.shader` module provides utilities for preprocessing Metal shaders.

### ShaderPreprocessor

```python
from pymetal.shader import ShaderPreprocessor

pp = ShaderPreprocessor()
pp.add_include_path("./shaders")
pp.define("BLOCK_SIZE", "256")
pp.define("USE_FAST_MATH")

source = pp.process('''
    #include "common.metal"
    #ifdef USE_FAST_MATH
    // Fast math enabled
    #endif
    kernel void kernel(...) {
        int size = BLOCK_SIZE;  // Becomes 256
    }
''')
```

**Methods:**

| Method | Description |
|--------|-------------|
| `add_include_path(path)` | Add directory to search path |
| `define(name, value="1")` | Define a macro |
| `define_many(dict)` | Define multiple macros |
| `undefine(name)` | Remove a macro |
| `process(source, ...)` | Preprocess shader source |
| `process_file(filepath, ...)` | Load and preprocess file |
| `clear_cache()` | Clear include cache |

### ShaderTemplate

```python
from pymetal.shader import ShaderTemplate

template = ShaderTemplate('''
    kernel void {name}(device {dtype}* data [[buffer(0)]],
                       uint idx [[thread_position_in_grid]]) {{
        data[idx] = data[idx] {op};
    }}
''')
template.set_default("dtype", "float")
source = template.render(name="double", op="* 2.0")
```

### create_compute_kernel

Helper for generating compute kernels.

```python
from pymetal.shader import create_compute_kernel

source = create_compute_kernel(
    name="vector_add",
    body="c[idx] = a[idx] + b[idx];",
    buffers=[
        ("a", "float", "read"),
        ("b", "float", "read"),
        ("c", "float", "write"),
    ]
)
```

### compute_shader_hash

```python
from pymetal.shader import compute_shader_hash

hash_value = compute_shader_hash(source)  # SHA-256 hex string
```

---

## Enumerations

### Storage and Cache Modes

```python
pm.StorageMode.Shared      # CPU and GPU accessible
pm.StorageMode.Managed     # Explicit synchronization
pm.StorageMode.Private     # GPU only (fastest)
pm.StorageMode.Memoryless  # Tile memory only

pm.CPUCacheMode.DefaultCache
pm.CPUCacheMode.WriteCombined
```

### Load/Store Actions

```python
pm.LoadAction.DontCare
pm.LoadAction.Load
pm.LoadAction.Clear

pm.StoreAction.DontCare
pm.StoreAction.Store
pm.StoreAction.MultisampleResolve
```

### Pixel Formats

```python
pm.PixelFormat.RGBA8Unorm
pm.PixelFormat.BGRA8Unorm
pm.PixelFormat.R32Float
pm.PixelFormat.Depth32Float
# ... and many more
```

### Primitive Types

```python
pm.PrimitiveType.Point
pm.PrimitiveType.Line
pm.PrimitiveType.LineStrip
pm.PrimitiveType.Triangle
pm.PrimitiveType.TriangleStrip
```

### Texture Types

```python
pm.TextureType.Type1D
pm.TextureType.Type2D
pm.TextureType.Type3D
pm.TextureType.TypeCube
pm.TextureType.Type2DArray
# ... and more
```

### Sampler Modes

```python
pm.SamplerMinMagFilter.Nearest
pm.SamplerMinMagFilter.Linear

pm.SamplerMipFilter.NotMipmapped
pm.SamplerMipFilter.Nearest
pm.SamplerMipFilter.Linear

pm.SamplerAddressMode.ClampToEdge
pm.SamplerAddressMode.Repeat
pm.SamplerAddressMode.MirrorRepeat
```

### Compare Functions

```python
pm.CompareFunction.Never
pm.CompareFunction.Less
pm.CompareFunction.Equal
pm.CompareFunction.LessEqual
pm.CompareFunction.Greater
pm.CompareFunction.NotEqual
pm.CompareFunction.GreaterEqual
pm.CompareFunction.Always
```

### Blend Factors and Operations

```python
pm.BlendFactor.Zero
pm.BlendFactor.One
pm.BlendFactor.SourceAlpha
pm.BlendFactor.OneMinusSourceAlpha
# ... and more

pm.BlendOperation.Add
pm.BlendOperation.Subtract
pm.BlendOperation.Min
pm.BlendOperation.Max
```

### Culling and Winding

```python
pm.CullMode.None_
pm.CullMode.Front
pm.CullMode.Back

pm.Winding.Clockwise
pm.Winding.CounterClockwise
```

---

## Constants

### Resource Options (bitmask)

```python
pm.ResourceStorageModeShared
pm.ResourceStorageModeManaged
pm.ResourceStorageModePrivate
pm.ResourceStorageModeMemoryless
pm.ResourceCPUCacheModeDefaultCache
pm.ResourceCPUCacheModeWriteCombined
pm.ResourceHazardTrackingModeUntracked
```

### Color Write Mask (bitmask)

```python
pm.ColorWriteMaskNone
pm.ColorWriteMaskRed
pm.ColorWriteMaskGreen
pm.ColorWriteMaskBlue
pm.ColorWriteMaskAlpha
pm.ColorWriteMaskAll
```

### Indirect Command Types (bitmask)

```python
pm.IndirectCommandTypeDraw
pm.IndirectCommandTypeDrawIndexed
pm.IndirectCommandTypeDrawPatches
pm.IndirectCommandTypeDrawIndexedPatches
```

---

## Exceptions

PyMetal uses a custom exception hierarchy for better error handling.

```python
try:
    library = device.new_library_with_source(invalid_shader)
except pm.CompileError as e:
    print(f"Shader compilation failed: {e}")
except pm.MetalError as e:
    print(f"Metal error: {e}")
```

### Exception Hierarchy

```sh
Exception
└── MetalError          # Base class for all Metal errors
    ├── CompileError    # Shader compilation failures
    ├── PipelineError   # Pipeline state creation failures
    ├── ResourceError   # Resource allocation failures
    └── ValidationError # Input validation failures
```

### Catching Errors

```python
# Catch specific error
try:
    pipeline = device.new_compute_pipeline_state(function)
except pm.PipelineError as e:
    print(f"Pipeline creation failed: {e}")

# Catch any Metal error
try:
    encoder.set_buffer(buffer, 0, 100)  # Invalid index
except pm.MetalError as e:
    print(f"Metal error: {e}")

# ValidationError for input validation
try:
    encoder.set_buffer(buffer, 0, 32)  # Index > 31
except pm.ValidationError as e:
    print(f"Validation failed: {e}")
```

---

## Utility Types

### Origin

```python
origin = pm.Origin(x, y, z)
```

### Size

```python
size = pm.Size(width, height, depth)
```

### Range

```python
range = pm.Range(location, length)
```

---

## Thread Safety Summary

| Object | Thread-Safe? | Notes |
|--------|--------------|-------|
| Device | Yes | All methods safe from any thread |
| CommandQueue | Yes | Multiple threads can create buffers |
| CommandBuffer | **No** | Use from one thread only |
| ComputeCommandEncoder | **No** | Use from one thread only |
| RenderCommandEncoder | **No** | Use from one thread only |
| BlitCommandEncoder | **No** | Use from one thread only |
| Buffer | Partially | Object safe, contents need sync |
| Texture | Partially | Object safe, contents need sync |
| Library | Yes | Functions can be retrieved concurrently |
| Pipeline States | Yes | Immutable after creation |

See [thread_safety.md](thread_safety.md) for detailed threading guidelines.
