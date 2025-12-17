"""
Custom exception hierarchy for pymetal.

This module provides specific exception types for different Metal error categories,
enabling more precise error handling in user code.
"""


class MetalError(Exception):
    """Base exception for all Metal-related errors.

    All pymetal exceptions inherit from this class, allowing users to catch
    all Metal errors with a single except clause if desired.

    Example:
        try:
            device.new_library_with_source(shader_code)
        except MetalError as e:
            print(f"Metal operation failed: {e}")
    """

    pass


class CompileError(MetalError):
    """Raised when Metal shader compilation fails.

    This exception is raised when new_library_with_source() encounters
    syntax errors or other compilation issues in Metal Shading Language code.

    Attributes:
        message: The error message from the Metal compiler.

    Example:
        try:
            library = device.new_library_with_source(shader_code)
        except CompileError as e:
            print(f"Shader compilation failed: {e}")
    """

    pass


class PipelineError(MetalError):
    """Raised when pipeline state creation fails.

    This exception is raised when creating compute or render pipeline states
    fails, typically due to invalid configuration or incompatible settings.

    Example:
        try:
            pipeline = device.new_compute_pipeline_state(function)
        except PipelineError as e:
            print(f"Pipeline creation failed: {e}")
    """

    pass


class ResourceError(MetalError):
    """Raised when resource creation or allocation fails.

    This exception is raised when creating Metal resources (buffers, textures,
    heaps, binary archives, etc.) fails, typically due to invalid parameters
    or insufficient memory.

    Example:
        try:
            buffer = device.new_buffer(size, options)
        except ResourceError as e:
            print(f"Resource creation failed: {e}")
    """

    pass


class ValidationError(MetalError):
    """Raised when input validation fails.

    This exception is raised when invalid parameters are passed to Metal
    operations, such as out-of-range indices or incompatible configurations.

    Example:
        try:
            encoder.set_buffer(buffer, 0, index=100)
        except ValidationError as e:
            print(f"Invalid parameter: {e}")
    """

    pass
