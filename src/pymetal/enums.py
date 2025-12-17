"""
pymetal.enums - Metal enumeration types

This module provides organized access to all Metal enumeration types
for storage modes, pixel formats, primitive types, and other constants.
"""

from ._pymetal import (
    # Phase 1 Enumerations
    StorageMode,
    CPUCacheMode,
    LoadAction,
    StoreAction,
    CommandBufferStatus,
    FunctionType,
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

__all__ = [
    # Phase 1 Enumerations
    "StorageMode",
    "CPUCacheMode",
    "LoadAction",
    "StoreAction",
    "CommandBufferStatus",
    "FunctionType",
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
