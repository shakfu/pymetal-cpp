"""
Type stubs for pymetal - Python bindings for Apple's Metal GPU API.

This file provides type hints for IDE support and static type checking.
"""

from typing import Optional, Tuple, Callable
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

__version__: str

# =============================================================================
# Exceptions
# =============================================================================

class MetalError(Exception):
    """Base exception for all Metal-related errors."""
    ...

class CompileError(MetalError):
    """Raised when Metal shader compilation fails."""
    ...

class PipelineError(MetalError):
    """Raised when pipeline state creation fails."""
    ...

class ResourceError(MetalError):
    """Raised when resource creation or allocation fails."""
    ...

class ValidationError(MetalError):
    """Raised when input validation fails."""
    ...

# =============================================================================
# Phase 1: Enumerations
# =============================================================================

class StorageMode(IntEnum):
    """Memory storage mode for Metal resources."""
    Shared: int
    Managed: int
    Private: int
    Memoryless: int

class CPUCacheMode(IntEnum):
    """CPU cache mode for Metal resources."""
    DefaultCache: int
    WriteCombined: int

class LoadAction(IntEnum):
    """Load action for render pass attachments."""
    DontCare: int
    Load: int
    Clear: int

class StoreAction(IntEnum):
    """Store action for render pass attachments."""
    DontCare: int
    Store: int
    MultisampleResolve: int
    StoreAndMultisampleResolve: int
    Unknown: int
    CustomSampleDepthStore: int

class CommandBufferStatus(IntEnum):
    """Status of a command buffer."""
    NotEnqueued: int
    Enqueued: int
    Committed: int
    Scheduled: int
    Completed: int
    Error: int

class FunctionType(IntEnum):
    """Type of a Metal function."""
    Vertex: int
    Fragment: int
    Kernel: int

# =============================================================================
# Phase 2: Graphics Enumerations
# =============================================================================

class PixelFormat(IntEnum):
    """Pixel format for textures and render targets."""
    Invalid: int
    A8Unorm: int
    R8Unorm: int
    R8Snorm: int
    R8Uint: int
    R8Sint: int
    R16Unorm: int
    R16Snorm: int
    R16Uint: int
    R16Sint: int
    R16Float: int
    RG8Unorm: int
    RG8Snorm: int
    RG8Uint: int
    RG8Sint: int
    R32Uint: int
    R32Sint: int
    R32Float: int
    RG16Unorm: int
    RG16Snorm: int
    RG16Uint: int
    RG16Sint: int
    RG16Float: int
    RGBA8Unorm: int
    RGBA8Unorm_sRGB: int
    RGBA8Snorm: int
    RGBA8Uint: int
    RGBA8Sint: int
    BGRA8Unorm: int
    BGRA8Unorm_sRGB: int
    RGB10A2Unorm: int
    RGB10A2Uint: int
    RG11B10Float: int
    RGB9E5Float: int
    RG32Uint: int
    RG32Sint: int
    RG32Float: int
    RGBA16Unorm: int
    RGBA16Snorm: int
    RGBA16Uint: int
    RGBA16Sint: int
    RGBA16Float: int
    RGBA32Uint: int
    RGBA32Sint: int
    RGBA32Float: int
    Depth16Unorm: int
    Depth32Float: int
    Stencil8: int
    Depth24Unorm_Stencil8: int
    Depth32Float_Stencil8: int

class PrimitiveType(IntEnum):
    """Primitive type for rendering."""
    Point: int
    Line: int
    LineStrip: int
    Triangle: int
    TriangleStrip: int

class IndexType(IntEnum):
    """Index type for indexed drawing."""
    UInt16: int
    UInt32: int

class VertexFormat(IntEnum):
    """Vertex attribute format."""
    Invalid: int
    UChar2: int
    UChar3: int
    UChar4: int
    Char2: int
    Char3: int
    Char4: int
    UChar2Normalized: int
    UChar3Normalized: int
    UChar4Normalized: int
    Char2Normalized: int
    Char3Normalized: int
    Char4Normalized: int
    UShort2: int
    UShort3: int
    UShort4: int
    Short2: int
    Short3: int
    Short4: int
    UShort2Normalized: int
    UShort3Normalized: int
    UShort4Normalized: int
    Short2Normalized: int
    Short3Normalized: int
    Short4Normalized: int
    Half2: int
    Half3: int
    Half4: int
    Float: int
    Float2: int
    Float3: int
    Float4: int
    Int: int
    Int2: int
    Int3: int
    Int4: int
    UInt: int
    UInt2: int
    UInt3: int
    UInt4: int

class VertexStepFunction(IntEnum):
    """Vertex step function for instanced rendering."""
    Constant: int
    PerVertex: int
    PerInstance: int
    PerPatch: int
    PerPatchControlPoint: int

class CullMode(IntEnum):
    """Face culling mode."""
    None_: int
    Front: int
    Back: int

class Winding(IntEnum):
    """Winding order for front-facing primitives."""
    Clockwise: int
    CounterClockwise: int

class TextureType(IntEnum):
    """Texture dimensionality."""
    Type1D: int
    Type1DArray: int
    Type2D: int
    Type2DArray: int
    Type2DMultisample: int
    TypeCube: int
    TypeCubeArray: int
    Type3D: int
    Type2DMultisampleArray: int
    TypeTextureBuffer: int

class SamplerMinMagFilter(IntEnum):
    """Texture minification/magnification filter."""
    Nearest: int
    Linear: int

class SamplerMipFilter(IntEnum):
    """Texture mipmap filter."""
    NotMipmapped: int
    Nearest: int
    Linear: int

class SamplerAddressMode(IntEnum):
    """Texture address mode."""
    ClampToEdge: int
    MirrorClampToEdge: int
    Repeat: int
    MirrorRepeat: int
    ClampToZero: int
    ClampToBorderColor: int

class CompareFunction(IntEnum):
    """Comparison function for depth/stencil testing."""
    Never: int
    Less: int
    Equal: int
    LessEqual: int
    Greater: int
    NotEqual: int
    GreaterEqual: int
    Always: int

class BlendFactor(IntEnum):
    """Blend factor for color blending."""
    Zero: int
    One: int
    SourceColor: int
    OneMinusSourceColor: int
    SourceAlpha: int
    OneMinusSourceAlpha: int
    DestinationColor: int
    OneMinusDestinationColor: int
    DestinationAlpha: int
    OneMinusDestinationAlpha: int
    SourceAlphaSaturated: int
    BlendColor: int
    OneMinusBlendColor: int
    BlendAlpha: int
    OneMinusBlendAlpha: int

class BlendOperation(IntEnum):
    """Blend operation for color blending."""
    Add: int
    Subtract: int
    ReverseSubtract: int
    Min: int
    Max: int

class StencilOperation(IntEnum):
    """Stencil operation."""
    Keep: int
    Zero: int
    Replace: int
    IncrementClamp: int
    DecrementClamp: int
    Invert: int
    IncrementWrap: int
    DecrementWrap: int

# =============================================================================
# Phase 3: Advanced Enumerations
# =============================================================================

class DataType(IntEnum):
    """Data type for argument buffers."""
    None_: int
    Struct: int
    Array: int
    Float: int
    Float2: int
    Float3: int
    Float4: int
    Float2x2: int
    Float2x3: int
    Float2x4: int
    Float3x2: int
    Float3x3: int
    Float3x4: int
    Float4x2: int
    Float4x3: int
    Float4x4: int
    Half: int
    Half2: int
    Half3: int
    Half4: int
    Half2x2: int
    Half2x3: int
    Half2x4: int
    Half3x2: int
    Half3x3: int
    Half3x4: int
    Half4x2: int
    Half4x3: int
    Half4x4: int
    Int: int
    Int2: int
    Int3: int
    Int4: int
    UInt: int
    UInt2: int
    UInt3: int
    UInt4: int
    Short: int
    Short2: int
    Short3: int
    Short4: int
    UShort: int
    UShort2: int
    UShort3: int
    UShort4: int
    Char: int
    Char2: int
    Char3: int
    Char4: int
    UChar: int
    UChar2: int
    UChar3: int
    UChar4: int
    Bool: int
    Bool2: int
    Bool3: int
    Bool4: int
    Texture: int
    Sampler: int
    Pointer: int

class BindingAccess(IntEnum):
    """Binding access mode for argument buffers."""
    ReadOnly: int
    ReadWrite: int
    WriteOnly: int

# =============================================================================
# Resource Option Constants
# =============================================================================

ResourceCPUCacheModeDefaultCache: int
ResourceCPUCacheModeWriteCombined: int
ResourceStorageModeShared: int
ResourceStorageModeManaged: int
ResourceStorageModePrivate: int
ResourceStorageModeMemoryless: int
ResourceHazardTrackingModeUntracked: int

# ColorWriteMask constants
ColorWriteMaskNone: int
ColorWriteMaskRed: int
ColorWriteMaskGreen: int
ColorWriteMaskBlue: int
ColorWriteMaskAlpha: int
ColorWriteMaskAll: int

# IndirectCommandType constants
IndirectCommandTypeDraw: int
IndirectCommandTypeDrawIndexed: int
IndirectCommandTypeDrawPatches: int
IndirectCommandTypeDrawIndexedPatches: int

# =============================================================================
# Utility Structures
# =============================================================================

class Origin:
    """3D origin (x, y, z)."""
    x: int
    y: int
    z: int
    def __init__(self, x: int = 0, y: int = 0, z: int = 0) -> None: ...

class Size:
    """3D size (width, height, depth)."""
    width: int
    height: int
    depth: int
    def __init__(self, width: int = 0, height: int = 0, depth: int = 0) -> None: ...

class Range:
    """Range with location and length."""
    location: int
    length: int
    def __init__(self, location: int = 0, length: int = 0) -> None: ...

class ClearColor:
    """RGBA clear color."""
    red: float
    green: float
    blue: float
    alpha: float
    def __init__(
        self, red: float = 0.0, green: float = 0.0, blue: float = 0.0, alpha: float = 1.0
    ) -> None: ...

# =============================================================================
# Phase 1: Core Classes
# =============================================================================

class Device:
    """Metal GPU device."""

    @property
    def name(self) -> str:
        """Device name."""
        ...

    @property
    def max_threads_per_threadgroup(self) -> Tuple[int, int, int]:
        """Maximum threads per threadgroup (width, height, depth)."""
        ...

    def new_command_queue(self) -> CommandQueue:
        """Create a new command queue."""
        ...

    def new_buffer(self, length: int, options: int) -> Buffer:
        """Create a new buffer with specified length and options."""
        ...

    def new_buffer_with_data(self, data: bytes, length: int, options: int) -> Buffer:
        """Create a new buffer initialized with data."""
        ...

    def new_library_with_source(self, source: str) -> Library:
        """Compile a new library from Metal shader source code.

        Raises:
            CompileError: If shader compilation fails.
        """
        ...

    def new_compute_pipeline_state(self, function: Function) -> ComputePipelineState:
        """Create a compute pipeline state from a kernel function.

        Raises:
            PipelineError: If pipeline creation fails.
        """
        ...

    def new_texture(self, descriptor: TextureDescriptor) -> Texture:
        """Create a new texture."""
        ...

    def new_sampler_state(self, descriptor: SamplerDescriptor) -> SamplerState:
        """Create a new sampler state."""
        ...

    def new_render_pipeline_state(
        self, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineState:
        """Create a render pipeline state from a descriptor.

        Raises:
            PipelineError: If pipeline creation fails.
        """
        ...

    def new_depth_stencil_state(
        self, descriptor: DepthStencilDescriptor
    ) -> DepthStencilState:
        """Create a depth/stencil state."""
        ...

    def new_heap(self, descriptor: HeapDescriptor) -> Heap:
        """Create a new memory heap."""
        ...

    def new_fence(self) -> Fence:
        """Create a new fence for synchronization."""
        ...

    def new_event(self) -> Event:
        """Create a new event."""
        ...

    def new_shared_event(self) -> SharedEvent:
        """Create a new shared event."""
        ...

    def new_indirect_command_buffer(
        self, descriptor: IndirectCommandBufferDescriptor, max_count: int, options: int
    ) -> IndirectCommandBuffer:
        """Create a new indirect command buffer."""
        ...

    def new_binary_archive(self, descriptor: BinaryArchiveDescriptor) -> BinaryArchive:
        """Create a new binary archive.

        Raises:
            ResourceError: If archive creation fails.
        """
        ...

def create_system_default_device() -> Device:
    """Get the default Metal device."""
    ...

class CommandQueue:
    """Command queue for submitting work to the GPU."""

    @property
    def device(self) -> Device:
        """The device that created this queue."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def command_buffer(self) -> CommandBuffer:
        """Create a new command buffer."""
        ...

class CommandBuffer:
    """Command buffer containing GPU commands."""

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    @property
    def status(self) -> CommandBufferStatus:
        """Current status of the command buffer."""
        ...

    def compute_command_encoder(self) -> ComputeCommandEncoder:
        """Create a compute command encoder."""
        ...

    def render_command_encoder(
        self, descriptor: RenderPassDescriptor
    ) -> RenderCommandEncoder:
        """Create a render command encoder."""
        ...

    def blit_command_encoder(self) -> BlitCommandEncoder:
        """Create a blit command encoder."""
        ...

    def commit(self) -> None:
        """Commit the command buffer for execution."""
        ...

    def wait_until_completed(self) -> None:
        """Block until the command buffer has completed execution."""
        ...

    def wait_until_scheduled(self) -> None:
        """Block until the command buffer has been scheduled."""
        ...

    def encode_signal_event(self, event: Event, value: int) -> None:
        """Encode a signal event command."""
        ...

    def encode_wait_for_event(self, event: Event, value: int) -> None:
        """Encode a wait for event command."""
        ...

class Buffer:
    """GPU memory buffer."""

    @property
    def length(self) -> int:
        """Buffer size in bytes."""
        ...

    @property
    def gpu_address(self) -> int:
        """GPU virtual address."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def contents(self) -> NDArray[np.uint8]:
        """Get buffer contents as a numpy array (zero-copy for shared storage)."""
        ...

class Library:
    """Compiled Metal shader library."""

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def new_function(self, name: str) -> Function:
        """Get a function from the library by name."""
        ...

class Function:
    """Metal shader function."""

    @property
    def name(self) -> str:
        """Function name."""
        ...

    @property
    def function_type(self) -> FunctionType:
        """Function type (vertex, fragment, or kernel)."""
        ...

    def new_argument_encoder(self, buffer_index: int) -> ArgumentEncoder:
        """Create an argument encoder for the specified buffer index."""
        ...

class ComputePipelineState:
    """Compute pipeline state."""

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        """Maximum total threads per threadgroup."""
        ...

    @property
    def thread_execution_width(self) -> int:
        """Thread execution width (SIMD width)."""
        ...

class ComputeCommandEncoder:
    """Encoder for compute commands."""

    def set_compute_pipeline_state(self, state: ComputePipelineState) -> None:
        """Set the compute pipeline state."""
        ...

    def set_buffer(self, buffer: Buffer, offset: int, index: int) -> None:
        """Set a buffer at the specified index."""
        ...

    def set_texture(self, texture: Texture, index: int) -> None:
        """Set a texture at the specified index."""
        ...

    def set_sampler_state(self, sampler: SamplerState, index: int) -> None:
        """Set a sampler state at the specified index."""
        ...

    def dispatch_threads(
        self,
        threads_per_grid_w: int,
        threads_per_grid_h: int,
        threads_per_grid_d: int,
        threads_per_threadgroup_w: int,
        threads_per_threadgroup_h: int,
        threads_per_threadgroup_d: int,
    ) -> None:
        """Dispatch threads with specified grid and threadgroup sizes."""
        ...

    def dispatch_threadgroups(
        self,
        threadgroups_per_grid_w: int,
        threadgroups_per_grid_h: int,
        threadgroups_per_grid_d: int,
        threads_per_threadgroup_w: int,
        threads_per_threadgroup_h: int,
        threads_per_threadgroup_d: int,
    ) -> None:
        """Dispatch threadgroups with specified sizes."""
        ...

    def update_fence(self, fence: Fence) -> None:
        """Update a fence after encoding."""
        ...

    def wait_for_fence(self, fence: Fence) -> None:
        """Wait for a fence before encoding."""
        ...

    def end_encoding(self) -> None:
        """End encoding commands."""
        ...

# =============================================================================
# Phase 2: Graphics Classes
# =============================================================================

class Texture:
    """GPU texture resource."""

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        ...

    @property
    def depth(self) -> int:
        """Texture depth (for 3D textures)."""
        ...

    @property
    def pixel_format(self) -> PixelFormat:
        """Pixel format."""
        ...

    @property
    def texture_type(self) -> TextureType:
        """Texture type."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def replace_region(
        self,
        region_origin: Origin,
        region_size: Size,
        mipmap_level: int,
        data: bytes,
        bytes_per_row: int,
    ) -> None:
        """Replace a region of the texture with data."""
        ...

    def get_bytes(
        self,
        data: bytearray,
        bytes_per_row: int,
        region_origin: Origin,
        region_size: Size,
        mipmap_level: int,
    ) -> None:
        """Copy texture data to a buffer."""
        ...

class TextureDescriptor:
    """Descriptor for creating textures."""

    @staticmethod
    def texture2d_descriptor(
        pixel_format: PixelFormat, width: int, height: int, mipmapped: bool
    ) -> TextureDescriptor:
        """Create a 2D texture descriptor."""
        ...

    @property
    def texture_type(self) -> TextureType: ...
    @texture_type.setter
    def texture_type(self, value: TextureType) -> None: ...

    @property
    def pixel_format(self) -> PixelFormat: ...
    @pixel_format.setter
    def pixel_format(self, value: PixelFormat) -> None: ...

    @property
    def width(self) -> int: ...
    @width.setter
    def width(self, value: int) -> None: ...

    @property
    def height(self) -> int: ...
    @height.setter
    def height(self, value: int) -> None: ...

    @property
    def depth(self) -> int: ...
    @depth.setter
    def depth(self, value: int) -> None: ...

    @property
    def storage_mode(self) -> StorageMode: ...
    @storage_mode.setter
    def storage_mode(self, value: StorageMode) -> None: ...

    @property
    def usage(self) -> int: ...
    @usage.setter
    def usage(self, value: int) -> None: ...

class SamplerState:
    """Sampler state for texture sampling."""

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

class SamplerDescriptor:
    """Descriptor for creating sampler states."""

    @staticmethod
    def sampler_descriptor() -> SamplerDescriptor:
        """Create a sampler descriptor with default values."""
        ...

    @property
    def min_filter(self) -> SamplerMinMagFilter: ...
    @min_filter.setter
    def min_filter(self, value: SamplerMinMagFilter) -> None: ...

    @property
    def mag_filter(self) -> SamplerMinMagFilter: ...
    @mag_filter.setter
    def mag_filter(self, value: SamplerMinMagFilter) -> None: ...

    @property
    def mip_filter(self) -> SamplerMipFilter: ...
    @mip_filter.setter
    def mip_filter(self, value: SamplerMipFilter) -> None: ...

    @property
    def s_address_mode(self) -> SamplerAddressMode: ...
    @s_address_mode.setter
    def s_address_mode(self, value: SamplerAddressMode) -> None: ...

    @property
    def t_address_mode(self) -> SamplerAddressMode: ...
    @t_address_mode.setter
    def t_address_mode(self, value: SamplerAddressMode) -> None: ...

    @property
    def r_address_mode(self) -> SamplerAddressMode: ...
    @r_address_mode.setter
    def r_address_mode(self, value: SamplerAddressMode) -> None: ...

class RenderPipelineState:
    """Render pipeline state."""

    @property
    def label(self) -> str:
        """Debug label."""
        ...

class RenderPipelineDescriptor:
    """Descriptor for creating render pipeline states."""

    @staticmethod
    def render_pipeline_descriptor() -> RenderPipelineDescriptor:
        """Create a render pipeline descriptor."""
        ...

    @property
    def vertex_function(self) -> Optional[Function]: ...
    @vertex_function.setter
    def vertex_function(self, value: Function) -> None: ...

    @property
    def fragment_function(self) -> Optional[Function]: ...
    @fragment_function.setter
    def fragment_function(self, value: Function) -> None: ...

    @property
    def vertex_descriptor(self) -> Optional[VertexDescriptor]: ...
    @vertex_descriptor.setter
    def vertex_descriptor(self, value: VertexDescriptor) -> None: ...

    @property
    def depth_attachment_pixel_format(self) -> PixelFormat: ...
    @depth_attachment_pixel_format.setter
    def depth_attachment_pixel_format(self, value: PixelFormat) -> None: ...

    @property
    def stencil_attachment_pixel_format(self) -> PixelFormat: ...
    @stencil_attachment_pixel_format.setter
    def stencil_attachment_pixel_format(self, value: PixelFormat) -> None: ...

    def color_attachment(self, index: int) -> RenderPipelineColorAttachmentDescriptor:
        """Get the color attachment descriptor at the specified index."""
        ...

class RenderPipelineColorAttachmentDescriptor:
    """Color attachment descriptor for render pipelines."""

    @property
    def pixel_format(self) -> PixelFormat: ...
    @pixel_format.setter
    def pixel_format(self, value: PixelFormat) -> None: ...

    @property
    def blending_enabled(self) -> bool: ...
    @blending_enabled.setter
    def blending_enabled(self, value: bool) -> None: ...

    @property
    def source_rgb_blend_factor(self) -> BlendFactor: ...
    @source_rgb_blend_factor.setter
    def source_rgb_blend_factor(self, value: BlendFactor) -> None: ...

    @property
    def destination_rgb_blend_factor(self) -> BlendFactor: ...
    @destination_rgb_blend_factor.setter
    def destination_rgb_blend_factor(self, value: BlendFactor) -> None: ...

    @property
    def rgb_blend_operation(self) -> BlendOperation: ...
    @rgb_blend_operation.setter
    def rgb_blend_operation(self, value: BlendOperation) -> None: ...

    @property
    def source_alpha_blend_factor(self) -> BlendFactor: ...
    @source_alpha_blend_factor.setter
    def source_alpha_blend_factor(self, value: BlendFactor) -> None: ...

    @property
    def destination_alpha_blend_factor(self) -> BlendFactor: ...
    @destination_alpha_blend_factor.setter
    def destination_alpha_blend_factor(self, value: BlendFactor) -> None: ...

    @property
    def alpha_blend_operation(self) -> BlendOperation: ...
    @alpha_blend_operation.setter
    def alpha_blend_operation(self, value: BlendOperation) -> None: ...

    @property
    def write_mask(self) -> int: ...
    @write_mask.setter
    def write_mask(self, value: int) -> None: ...

class RenderPassDescriptor:
    """Descriptor for configuring a render pass."""

    @staticmethod
    def render_pass_descriptor() -> RenderPassDescriptor:
        """Create a render pass descriptor."""
        ...

    def color_attachment(self, index: int) -> RenderPassColorAttachmentDescriptor:
        """Get the color attachment descriptor at the specified index."""
        ...

    @property
    def depth_attachment(self) -> RenderPassDepthAttachmentDescriptor:
        """Get the depth attachment descriptor."""
        ...

class RenderPassAttachmentDescriptor:
    """Base class for render pass attachment descriptors."""

    @property
    def texture(self) -> Optional[Texture]: ...
    @texture.setter
    def texture(self, value: Texture) -> None: ...

    @property
    def level(self) -> int: ...
    @level.setter
    def level(self, value: int) -> None: ...

    @property
    def slice(self) -> int: ...
    @slice.setter
    def slice(self, value: int) -> None: ...

    @property
    def load_action(self) -> LoadAction: ...
    @load_action.setter
    def load_action(self, value: LoadAction) -> None: ...

    @property
    def store_action(self) -> StoreAction: ...
    @store_action.setter
    def store_action(self, value: StoreAction) -> None: ...

class RenderPassColorAttachmentDescriptor(RenderPassAttachmentDescriptor):
    """Color attachment descriptor for render passes."""

    @property
    def clear_color(self) -> ClearColor: ...
    @clear_color.setter
    def clear_color(self, value: ClearColor) -> None: ...

class RenderPassDepthAttachmentDescriptor(RenderPassAttachmentDescriptor):
    """Depth attachment descriptor for render passes."""

    @property
    def clear_depth(self) -> float: ...
    @clear_depth.setter
    def clear_depth(self, value: float) -> None: ...

class RenderCommandEncoder:
    """Encoder for render commands."""

    def set_render_pipeline_state(self, state: RenderPipelineState) -> None:
        """Set the render pipeline state."""
        ...

    def set_vertex_buffer(self, buffer: Buffer, offset: int, index: int) -> None:
        """Set a vertex buffer at the specified index."""
        ...

    def set_fragment_buffer(self, buffer: Buffer, offset: int, index: int) -> None:
        """Set a fragment buffer at the specified index."""
        ...

    def set_vertex_texture(self, texture: Texture, index: int) -> None:
        """Set a vertex texture at the specified index."""
        ...

    def set_fragment_texture(self, texture: Texture, index: int) -> None:
        """Set a fragment texture at the specified index."""
        ...

    def set_vertex_sampler_state(self, sampler: SamplerState, index: int) -> None:
        """Set a vertex sampler state at the specified index."""
        ...

    def set_fragment_sampler_state(self, sampler: SamplerState, index: int) -> None:
        """Set a fragment sampler state at the specified index."""
        ...

    def set_depth_stencil_state(self, state: DepthStencilState) -> None:
        """Set the depth/stencil state."""
        ...

    def set_cull_mode(self, mode: CullMode) -> None:
        """Set the face culling mode."""
        ...

    def set_front_facing_winding(self, winding: Winding) -> None:
        """Set the front-facing winding order."""
        ...

    def set_viewport(
        self,
        origin_x: float,
        origin_y: float,
        width: float,
        height: float,
        znear: float,
        zfar: float,
    ) -> None:
        """Set the viewport."""
        ...

    def set_scissor_rect(self, x: int, y: int, width: int, height: int) -> None:
        """Set the scissor rectangle."""
        ...

    def draw_primitives(
        self,
        primitive_type: PrimitiveType,
        vertex_start: int,
        vertex_count: int,
    ) -> None:
        """Draw primitives."""
        ...

    def draw_primitives_instanced(
        self,
        primitive_type: PrimitiveType,
        vertex_start: int,
        vertex_count: int,
        instance_count: int,
    ) -> None:
        """Draw instanced primitives."""
        ...

    def draw_indexed_primitives(
        self,
        primitive_type: PrimitiveType,
        index_count: int,
        index_type: IndexType,
        index_buffer: Buffer,
        index_buffer_offset: int,
    ) -> None:
        """Draw indexed primitives."""
        ...

    def draw_indexed_primitives_instanced(
        self,
        primitive_type: PrimitiveType,
        index_count: int,
        index_type: IndexType,
        index_buffer: Buffer,
        index_buffer_offset: int,
        instance_count: int,
    ) -> None:
        """Draw indexed instanced primitives."""
        ...

    def update_fence(self, fence: Fence) -> None:
        """Update a fence after encoding."""
        ...

    def wait_for_fence(self, fence: Fence) -> None:
        """Wait for a fence before encoding."""
        ...

    def end_encoding(self) -> None:
        """End encoding commands."""
        ...

class VertexDescriptor:
    """Vertex descriptor for configuring vertex input."""

    @staticmethod
    def vertex_descriptor() -> VertexDescriptor:
        """Create a vertex descriptor."""
        ...

    def attribute(self, index: int) -> VertexAttributeDescriptor:
        """Get the attribute descriptor at the specified index."""
        ...

    def layout(self, index: int) -> VertexBufferLayoutDescriptor:
        """Get the layout descriptor at the specified index."""
        ...

class VertexAttributeDescriptor:
    """Vertex attribute descriptor."""

    @property
    def format(self) -> VertexFormat: ...
    @format.setter
    def format(self, value: VertexFormat) -> None: ...

    @property
    def offset(self) -> int: ...
    @offset.setter
    def offset(self, value: int) -> None: ...

    @property
    def buffer_index(self) -> int: ...
    @buffer_index.setter
    def buffer_index(self, value: int) -> None: ...

class VertexBufferLayoutDescriptor:
    """Vertex buffer layout descriptor."""

    @property
    def stride(self) -> int: ...
    @stride.setter
    def stride(self, value: int) -> None: ...

    @property
    def step_function(self) -> VertexStepFunction: ...
    @step_function.setter
    def step_function(self, value: VertexStepFunction) -> None: ...

    @property
    def step_rate(self) -> int: ...
    @step_rate.setter
    def step_rate(self, value: int) -> None: ...

# =============================================================================
# Phase 2 Advanced: Depth/Stencil, Blit, Heap, Fence
# =============================================================================

class DepthStencilState:
    """Depth/stencil state."""

    @property
    def label(self) -> str:
        """Debug label."""
        ...

class DepthStencilDescriptor:
    """Descriptor for creating depth/stencil states."""

    @staticmethod
    def depth_stencil_descriptor() -> DepthStencilDescriptor:
        """Create a depth/stencil descriptor."""
        ...

    @property
    def depth_compare_function(self) -> CompareFunction: ...
    @depth_compare_function.setter
    def depth_compare_function(self, value: CompareFunction) -> None: ...

    @property
    def depth_write_enabled(self) -> bool: ...
    @depth_write_enabled.setter
    def depth_write_enabled(self, value: bool) -> None: ...

    @property
    def front_face_stencil(self) -> StencilDescriptor: ...
    @front_face_stencil.setter
    def front_face_stencil(self, value: StencilDescriptor) -> None: ...

    @property
    def back_face_stencil(self) -> StencilDescriptor: ...
    @back_face_stencil.setter
    def back_face_stencil(self, value: StencilDescriptor) -> None: ...

    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, value: str) -> None: ...

class StencilDescriptor:
    """Stencil descriptor."""

    @staticmethod
    def stencil_descriptor() -> StencilDescriptor:
        """Create a stencil descriptor."""
        ...

    @property
    def stencil_compare_function(self) -> CompareFunction: ...
    @stencil_compare_function.setter
    def stencil_compare_function(self, value: CompareFunction) -> None: ...

    @property
    def stencil_failure_operation(self) -> StencilOperation: ...
    @stencil_failure_operation.setter
    def stencil_failure_operation(self, value: StencilOperation) -> None: ...

    @property
    def depth_failure_operation(self) -> StencilOperation: ...
    @depth_failure_operation.setter
    def depth_failure_operation(self, value: StencilOperation) -> None: ...

    @property
    def depth_stencil_pass_operation(self) -> StencilOperation: ...
    @depth_stencil_pass_operation.setter
    def depth_stencil_pass_operation(self, value: StencilOperation) -> None: ...

    @property
    def read_mask(self) -> int: ...
    @read_mask.setter
    def read_mask(self, value: int) -> None: ...

    @property
    def write_mask(self) -> int: ...
    @write_mask.setter
    def write_mask(self, value: int) -> None: ...

class BlitCommandEncoder:
    """Encoder for blit (memory transfer) commands."""

    def copy_from_buffer_to_buffer(
        self,
        source: Buffer,
        source_offset: int,
        destination: Buffer,
        destination_offset: int,
        size: int,
    ) -> None:
        """Copy data between buffers."""
        ...

    def copy_from_texture_to_texture(
        self,
        source: Texture,
        source_slice: int,
        source_level: int,
        source_origin: Origin,
        source_size: Size,
        destination: Texture,
        destination_slice: int,
        destination_level: int,
        destination_origin: Origin,
    ) -> None:
        """Copy data between textures."""
        ...

    def copy_from_buffer_to_texture(
        self,
        source: Buffer,
        source_offset: int,
        source_bytes_per_row: int,
        source_bytes_per_image: int,
        source_size: Size,
        destination: Texture,
        destination_slice: int,
        destination_level: int,
        destination_origin: Origin,
    ) -> None:
        """Copy data from a buffer to a texture."""
        ...

    def copy_from_texture_to_buffer(
        self,
        source: Texture,
        source_slice: int,
        source_level: int,
        source_origin: Origin,
        source_size: Size,
        destination: Buffer,
        destination_offset: int,
        destination_bytes_per_row: int,
        destination_bytes_per_image: int,
    ) -> None:
        """Copy data from a texture to a buffer."""
        ...

    def fill_buffer(
        self, buffer: Buffer, range: Range, value: int
    ) -> None:
        """Fill a buffer region with a byte value."""
        ...

    def generate_mipmaps(self, texture: Texture) -> None:
        """Generate mipmaps for a texture."""
        ...

    def synchronize_resource(self, resource: Buffer) -> None:
        """Synchronize a managed resource."""
        ...

    def update_fence(self, fence: Fence) -> None:
        """Update a fence after encoding."""
        ...

    def wait_for_fence(self, fence: Fence) -> None:
        """Wait for a fence before encoding."""
        ...

    def end_encoding(self) -> None:
        """End encoding commands."""
        ...

class Heap:
    """Memory heap for resource allocation."""

    @property
    def size(self) -> int:
        """Heap size in bytes."""
        ...

    @property
    def used_size(self) -> int:
        """Used size in bytes."""
        ...

    @property
    def current_allocated_size(self) -> int:
        """Currently allocated size in bytes."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def max_available_size(self, alignment: int) -> int:
        """Get the maximum available size for the given alignment."""
        ...

    def new_buffer(self, length: int, options: int) -> Buffer:
        """Allocate a buffer from the heap."""
        ...

    def new_texture(self, descriptor: TextureDescriptor) -> Texture:
        """Allocate a texture from the heap."""
        ...

class HeapDescriptor:
    """Descriptor for creating memory heaps."""

    @staticmethod
    def heap_descriptor() -> HeapDescriptor:
        """Create a heap descriptor."""
        ...

    @property
    def size(self) -> int: ...
    @size.setter
    def size(self, value: int) -> None: ...

    @property
    def storage_mode(self) -> StorageMode: ...
    @storage_mode.setter
    def storage_mode(self, value: StorageMode) -> None: ...

    @property
    def cpu_cache_mode(self) -> CPUCacheMode: ...
    @cpu_cache_mode.setter
    def cpu_cache_mode(self, value: CPUCacheMode) -> None: ...

class Fence:
    """Synchronization fence."""

    @property
    def device(self) -> Device:
        """The device that created this fence."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

# =============================================================================
# Display Integration
# =============================================================================

class MetalLayer:
    """Core Animation layer for Metal rendering."""

    @staticmethod
    def layer() -> MetalLayer:
        """Create a new Metal layer."""
        ...

    @property
    def device(self) -> Optional[Device]: ...
    @device.setter
    def device(self, value: Device) -> None: ...

    @property
    def pixel_format(self) -> PixelFormat: ...
    @pixel_format.setter
    def pixel_format(self, value: PixelFormat) -> None: ...

    @property
    def drawable_size(self) -> Tuple[float, float]:
        """Drawable size (width, height)."""
        ...

    @drawable_size.setter
    def drawable_size(self, value: Tuple[float, float]) -> None: ...

    @property
    def framebuffer_only(self) -> bool: ...
    @framebuffer_only.setter
    def framebuffer_only(self, value: bool) -> None: ...

    def next_drawable(self) -> Optional[MetalDrawable]:
        """Get the next drawable."""
        ...

class MetalDrawable:
    """Drawable for presenting to a Metal layer."""

    @property
    def texture(self) -> Texture:
        """The drawable's texture."""
        ...

    def present(self) -> None:
        """Present the drawable."""
        ...

# =============================================================================
# Phase 3: Events
# =============================================================================

class Event:
    """GPU event for synchronization."""

    @property
    def device(self) -> Device:
        """The device that created this event."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

class SharedEvent:
    """Shared event for cross-process synchronization."""

    @property
    def device(self) -> Device:
        """The device that created this event."""
        ...

    @property
    def signaled_value(self) -> int:
        """Current signaled value."""
        ...

    @signaled_value.setter
    def signaled_value(self, value: int) -> None: ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

# =============================================================================
# Phase 3: Argument Buffers
# =============================================================================

class ArgumentEncoder:
    """Encoder for argument buffers."""

    @property
    def encoded_length(self) -> int:
        """Length of the encoded argument buffer."""
        ...

    @property
    def alignment(self) -> int:
        """Required alignment for the argument buffer."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def set_argument_buffer(self, buffer: Buffer, offset: int) -> None:
        """Set the argument buffer to encode into."""
        ...

    def set_buffer(self, buffer: Buffer, offset: int, index: int) -> None:
        """Set a buffer in the argument buffer."""
        ...

    def set_texture(self, texture: Texture, index: int) -> None:
        """Set a texture in the argument buffer."""
        ...

    def set_sampler_state(self, sampler: SamplerState, index: int) -> None:
        """Set a sampler state in the argument buffer."""
        ...

    def constant_data(self, index: int) -> int:
        """Get the address for constant data at the specified index."""
        ...

class ArgumentDescriptor:
    """Descriptor for argument buffer entries."""

    @staticmethod
    def argument_descriptor() -> ArgumentDescriptor:
        """Create an argument descriptor."""
        ...

    @property
    def data_type(self) -> DataType: ...
    @data_type.setter
    def data_type(self, value: DataType) -> None: ...

    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, value: int) -> None: ...

    @property
    def array_length(self) -> int: ...
    @array_length.setter
    def array_length(self, value: int) -> None: ...

    @property
    def access(self) -> BindingAccess: ...
    @access.setter
    def access(self, value: BindingAccess) -> None: ...

    @property
    def texture_type(self) -> TextureType: ...
    @texture_type.setter
    def texture_type(self, value: TextureType) -> None: ...

    @property
    def constant_block_alignment(self) -> int: ...
    @constant_block_alignment.setter
    def constant_block_alignment(self, value: int) -> None: ...

# =============================================================================
# Phase 3: Indirect Command Buffers
# =============================================================================

class IndirectCommandBuffer:
    """Indirect command buffer for GPU-driven rendering."""

    @property
    def size(self) -> int:
        """Number of commands in the buffer."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def reset(self, range: Range) -> None:
        """Reset commands in the specified range."""
        ...

class IndirectCommandBufferDescriptor:
    """Descriptor for creating indirect command buffers."""

    @staticmethod
    def indirect_command_buffer_descriptor() -> IndirectCommandBufferDescriptor:
        """Create an indirect command buffer descriptor."""
        ...

    @property
    def command_types(self) -> int: ...
    @command_types.setter
    def command_types(self, value: int) -> None: ...

    @property
    def inherit_buffers(self) -> bool: ...
    @inherit_buffers.setter
    def inherit_buffers(self, value: bool) -> None: ...

    @property
    def inherit_pipeline_state(self) -> bool: ...
    @inherit_pipeline_state.setter
    def inherit_pipeline_state(self, value: bool) -> None: ...

    @property
    def max_vertex_buffer_bind_count(self) -> int: ...
    @max_vertex_buffer_bind_count.setter
    def max_vertex_buffer_bind_count(self, value: int) -> None: ...

    @property
    def max_fragment_buffer_bind_count(self) -> int: ...
    @max_fragment_buffer_bind_count.setter
    def max_fragment_buffer_bind_count(self, value: int) -> None: ...

# =============================================================================
# Phase 3: Binary Archives
# =============================================================================

class BinaryArchive:
    """Binary archive for caching compiled pipelines."""

    @property
    def device(self) -> Device:
        """The device that created this archive."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

class BinaryArchiveDescriptor:
    """Descriptor for creating binary archives."""

    @staticmethod
    def binary_archive_descriptor() -> BinaryArchiveDescriptor:
        """Create a binary archive descriptor."""
        ...

    @property
    def url(self) -> Optional[str]: ...
    @url.setter
    def url(self, value: str) -> None: ...

# =============================================================================
# Phase 3: Capture/Debugging
# =============================================================================

class CaptureScope:
    """Capture scope for GPU debugging."""

    @property
    def device(self) -> Device:
        """The device associated with this scope."""
        ...

    @property
    def command_queue(self) -> Optional[CommandQueue]:
        """The command queue associated with this scope."""
        ...

    @property
    def label(self) -> str:
        """Debug label."""
        ...

    @label.setter
    def label(self, value: str) -> None: ...

    def begin_scope(self) -> None:
        """Begin the capture scope."""
        ...

    def end_scope(self) -> None:
        """End the capture scope."""
        ...

class CaptureManager:
    """Manager for GPU capture operations."""

    @property
    def is_capturing(self) -> bool:
        """Whether a capture is currently in progress."""
        ...

    @property
    def default_capture_scope(self) -> Optional[CaptureScope]:
        """The default capture scope."""
        ...

    def new_capture_scope_with_device(self, device: Device) -> CaptureScope:
        """Create a new capture scope for a device."""
        ...

    def new_capture_scope_with_command_queue(self, queue: CommandQueue) -> CaptureScope:
        """Create a new capture scope for a command queue."""
        ...

def shared_capture_manager() -> CaptureManager:
    """Get the shared capture manager instance."""
    ...
