"""
pymetal.advanced - Advanced Metal features

This module provides advanced Metal features:
- Event system for fine-grained synchronization
- Argument buffers for efficient resource binding
- Indirect command buffers for GPU-driven rendering
- Binary archives for pipeline caching
- Capture scopes for GPU debugging
- Heaps for manual memory management
- Fences for encoder synchronization
- Blit encoder for memory operations
"""

from ._pymetal import (
    # Blit encoder
    BlitCommandEncoder,
    # Memory management
    Heap,
    HeapDescriptor,
    # Synchronization
    Fence,
    Event,
    SharedEvent,
    # Argument buffers
    ArgumentEncoder,
    ArgumentDescriptor,
    # Indirect commands
    IndirectCommandBuffer,
    IndirectCommandBufferDescriptor,
    # Binary archive
    BinaryArchive,
    BinaryArchiveDescriptor,
    # Capture/debugging
    CaptureScope,
    CaptureManager,
    shared_capture_manager,
)

__all__ = [
    # Blit encoder
    "BlitCommandEncoder",
    # Memory management
    "Heap",
    "HeapDescriptor",
    # Synchronization
    "Fence",
    "Event",
    "SharedEvent",
    # Argument buffers
    "ArgumentEncoder",
    "ArgumentDescriptor",
    # Indirect commands
    "IndirectCommandBuffer",
    "IndirectCommandBufferDescriptor",
    # Binary archive
    "BinaryArchive",
    "BinaryArchiveDescriptor",
    # Capture/debugging
    "CaptureScope",
    "CaptureManager",
    "shared_capture_manager",
]
