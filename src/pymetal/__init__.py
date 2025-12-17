"""
pymetal - Python bindings for Apple's Metal GPU API

This package provides Python bindings to Metal via metal-cpp and nanobind,
enabling GPU compute and graphics programming from Python.

Submodules:
    pymetal.enums - All enumeration types
    pymetal.types - Utility types (Origin, Size, Range, ClearColor)
    pymetal.compute - Compute pipeline classes
    pymetal.graphics - Graphics/render pipeline classes
    pymetal.advanced - Advanced features (events, indirect commands, etc.)
    pymetal.shader - Shader preprocessing utilities
"""

from .exceptions import (
    MetalError,
    CompileError,
    PipelineError,
    ResourceError,
    ValidationError,
)

from ._pymetal import (
    # Device management
    Device,
    create_system_default_device,
    copy_all_devices,
    # Command submission
    CommandQueue,
    CommandBuffer,
    CommandBufferStatus,
    # Memory resources
    Buffer,
    Texture,
    TextureDescriptor,
    # Shader compilation
    Library,
    Function,
    FunctionType,
    # Compute pipeline
    ComputePipelineState,
    ComputeCommandEncoder,
    # Graphics pipeline
    RenderPipelineState,
    RenderPipelineDescriptor,
    RenderPipelineColorAttachmentDescriptor,
    RenderCommandEncoder,
    # Render pass
    RenderPassDescriptor,
    RenderPassAttachmentDescriptor,
    RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor,
    ClearColor,
    # Sampling
    SamplerState,
    SamplerDescriptor,
    # Vertex descriptors
    VertexDescriptor,
    VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor,
    # Display integration
    MetalLayer,
    MetalDrawable,
    # Phase 2 Advanced: Blit encoder
    BlitCommandEncoder,
    # Phase 2 Advanced: Depth/stencil testing
    DepthStencilState,
    DepthStencilDescriptor,
    StencilDescriptor,
    # Phase 2 Advanced: Memory management
    Heap,
    HeapDescriptor,
    # Phase 2 Advanced: Synchronization
    Fence,
    # Utility structures
    Origin,
    Size,
    Range,
    # Phase 3: Event system
    Event,
    SharedEvent,
    # Phase 3: Argument buffers
    ArgumentEncoder,
    ArgumentDescriptor,
    # Phase 3: Indirect commands
    IndirectCommandBuffer,
    IndirectCommandBufferDescriptor,
    # Phase 3: Binary archive
    BinaryArchive,
    BinaryArchiveDescriptor,
    # Phase 3: Capture/debugging
    CaptureScope,
    CaptureManager,
    shared_capture_manager,
    # Phase 1 Enumerations
    StorageMode,
    CPUCacheMode,
    LoadAction,
    StoreAction,
    # Phase 2 Enumerations
    PixelFormat,
    PrimitiveType,
    IndexType,
    VertexFormat,
    VertexStepFunction,
    CullMode,
    Winding,
    TextureType,
    SamplerMinMagFilter,
    SamplerMipFilter,
    SamplerAddressMode,
    CompareFunction,
    BlendFactor,
    BlendOperation,
    StencilOperation,
    # Phase 3 Enumerations
    DataType,
    BindingAccess,
    # ResourceOptions constants (bitmask values)
    ResourceCPUCacheModeDefaultCache,
    ResourceCPUCacheModeWriteCombined,
    ResourceStorageModeShared,
    ResourceStorageModeManaged,
    ResourceStorageModePrivate,
    ResourceStorageModeMemoryless,
    ResourceHazardTrackingModeUntracked,
    # ColorWriteMask constants (bitmask values)
    ColorWriteMaskNone,
    ColorWriteMaskRed,
    ColorWriteMaskGreen,
    ColorWriteMaskBlue,
    ColorWriteMaskAlpha,
    ColorWriteMaskAll,
    # IndirectCommandType constants (bitmask values)
    IndirectCommandTypeDraw,
    IndirectCommandTypeDrawIndexed,
    IndirectCommandTypeDrawPatches,
    IndirectCommandTypeDrawIndexedPatches,
)

__version__ = "0.1.4"

# Import submodules for namespace organization
from . import enums
from . import types
from . import compute
from . import graphics
from . import advanced
from . import shader

__all__ = [
    # Submodules
    "enums",
    "types",
    "compute",
    "graphics",
    "advanced",
    "shader",
    # Exceptions
    "MetalError",
    "CompileError",
    "PipelineError",
    "ResourceError",
    "ValidationError",
    # Device management
    "Device",
    "create_system_default_device",
    "copy_all_devices",
    # Command submission
    "CommandQueue",
    "CommandBuffer",
    "CommandBufferStatus",
    # Memory resources
    "Buffer",
    "Texture",
    "TextureDescriptor",
    # Shader compilation
    "Library",
    "Function",
    "FunctionType",
    # Compute pipeline
    "ComputePipelineState",
    "ComputeCommandEncoder",
    # Graphics pipeline
    "RenderPipelineState",
    "RenderPipelineDescriptor",
    "RenderPipelineColorAttachmentDescriptor",
    "RenderCommandEncoder",
    # Render pass
    "RenderPassDescriptor",
    "RenderPassAttachmentDescriptor",
    "RenderPassColorAttachmentDescriptor",
    "RenderPassDepthAttachmentDescriptor",
    "ClearColor",
    # Sampling
    "SamplerState",
    "SamplerDescriptor",
    # Vertex descriptors
    "VertexDescriptor",
    "VertexAttributeDescriptor",
    "VertexBufferLayoutDescriptor",
    # Display integration
    "MetalLayer",
    "MetalDrawable",
    # Phase 2 Advanced: Blit encoder
    "BlitCommandEncoder",
    # Phase 2 Advanced: Depth/stencil testing
    "DepthStencilState",
    "DepthStencilDescriptor",
    "StencilDescriptor",
    # Phase 2 Advanced: Memory management
    "Heap",
    "HeapDescriptor",
    # Phase 2 Advanced: Synchronization
    "Fence",
    # Utility structures
    "Origin",
    "Size",
    "Range",
    # Phase 3: Event system
    "Event",
    "SharedEvent",
    # Phase 3: Argument buffers
    "ArgumentEncoder",
    "ArgumentDescriptor",
    # Phase 3: Indirect commands
    "IndirectCommandBuffer",
    "IndirectCommandBufferDescriptor",
    # Phase 3: Binary archive
    "BinaryArchive",
    "BinaryArchiveDescriptor",
    # Phase 3: Capture/debugging
    "CaptureScope",
    "CaptureManager",
    "shared_capture_manager",
    # Phase 1 Enumerations
    "StorageMode",
    "CPUCacheMode",
    "LoadAction",
    "StoreAction",
    # Phase 2 Enumerations
    "PixelFormat",
    "PrimitiveType",
    "IndexType",
    "VertexFormat",
    "VertexStepFunction",
    "CullMode",
    "Winding",
    "TextureType",
    "SamplerMinMagFilter",
    "SamplerMipFilter",
    "SamplerAddressMode",
    "CompareFunction",
    "BlendFactor",
    "BlendOperation",
    "StencilOperation",
    # Phase 3 Enumerations
    "DataType",
    "BindingAccess",
    # ResourceOptions constants
    "ResourceCPUCacheModeDefaultCache",
    "ResourceCPUCacheModeWriteCombined",
    "ResourceStorageModeShared",
    "ResourceStorageModeManaged",
    "ResourceStorageModePrivate",
    "ResourceStorageModeMemoryless",
    "ResourceHazardTrackingModeUntracked",
    # ColorWriteMask constants
    "ColorWriteMaskNone",
    "ColorWriteMaskRed",
    "ColorWriteMaskGreen",
    "ColorWriteMaskBlue",
    "ColorWriteMaskAlpha",
    "ColorWriteMaskAll",
    # IndirectCommandType constants
    "IndirectCommandTypeDraw",
    "IndirectCommandTypeDrawIndexed",
    "IndirectCommandTypeDrawPatches",
    "IndirectCommandTypeDrawIndexedPatches",
]
