"""
pymetal.graphics - Metal graphics/render pipeline classes

This module provides classes for GPU rendering operations:
- Textures and texture descriptors
- Render pipelines and descriptors
- Render pass configuration
- Samplers and depth/stencil state
- Vertex descriptors
- Display integration (MetalLayer, MetalDrawable)
"""

from ._pymetal import (
    # Textures
    Texture,
    TextureDescriptor,
    # Render pipeline
    RenderPipelineState,
    RenderPipelineDescriptor,
    RenderPipelineColorAttachmentDescriptor,
    RenderCommandEncoder,
    # Render pass
    RenderPassDescriptor,
    RenderPassAttachmentDescriptor,
    RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor,
    # Sampling
    SamplerState,
    SamplerDescriptor,
    # Depth/stencil
    DepthStencilState,
    DepthStencilDescriptor,
    StencilDescriptor,
    # Vertex descriptors
    VertexDescriptor,
    VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor,
    # Display integration
    MetalLayer,
    MetalDrawable,
)

__all__ = [
    # Textures
    "Texture",
    "TextureDescriptor",
    # Render pipeline
    "RenderPipelineState",
    "RenderPipelineDescriptor",
    "RenderPipelineColorAttachmentDescriptor",
    "RenderCommandEncoder",
    # Render pass
    "RenderPassDescriptor",
    "RenderPassAttachmentDescriptor",
    "RenderPassColorAttachmentDescriptor",
    "RenderPassDepthAttachmentDescriptor",
    # Sampling
    "SamplerState",
    "SamplerDescriptor",
    # Depth/stencil
    "DepthStencilState",
    "DepthStencilDescriptor",
    "StencilDescriptor",
    # Vertex descriptors
    "VertexDescriptor",
    "VertexAttributeDescriptor",
    "VertexBufferLayoutDescriptor",
    # Display integration
    "MetalLayer",
    "MetalDrawable",
]
