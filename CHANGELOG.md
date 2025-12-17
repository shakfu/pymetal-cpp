# Changelog

All notable changes to PyMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5]

### Fixed

- Fixed lint errors (unused imports in shader.py, test files, and type stubs)
- Fixed type checking errors in shader.py (variable shadowing)
- Fixed enum stub declarations to use `member = ...` syntax for mypy compatibility
- Moved imports to top of file in type stubs to satisfy E402

## [0.1.4]

### Added

- **Custom Exception Hierarchy**
  - `MetalError` - Base exception for all Metal-related errors
  - `CompileError` - Shader compilation failures
  - `PipelineError` - Pipeline state creation failures
  - `ResourceError` - Resource allocation failures
  - `ValidationError` - Input validation failures
- **Type Stub File** (`pymetal/__init__.pyi`)
  - Full type annotations for all 25+ enums and classes
  - IDE support for autocompletion and type checking
- **Input Validation**
  - Buffer index validation (0-31 range)
  - Texture index validation (0-31 range)
  - Sampler index validation (0-15 range)
  - Threadgroup dimension validation (must be > 0)
- **Thread Safety Documentation** (`docs/THREAD_SAFETY.md`)
  - Comprehensive guide to Metal threading model
  - Thread-safe vs non-thread-safe object documentation
  - Best practices and common pitfalls
  - GIL release behavior documentation
- **Performance Benchmark Tests** (`tests/test_benchmarks.py`)
  - Buffer allocation throughput tests
  - Shader compilation throughput tests
  - Compute kernel performance tests
  - Validation overhead tests
- **Edge Case Tests** (`tests/test_edge_cases.py`)
  - Minimal buffer sizes (1 byte, single float)
  - Boundary conditions (max buffer index)
  - Empty operations and multiple encoders
  - Texture edge cases (1x1, non-square, non-power-of-2)
  - Label edge cases (empty, long, unicode)
  - Shader edge cases (minimal, many buffers, structs)
- **API Docstrings**
  - Detailed docstrings for Device, CommandQueue, CommandBuffer, Buffer, ComputeCommandEncoder
  - Thread safety notes in class documentation
- **Namespace Organization** (submodules for cleaner imports)
  - `pymetal.enums` - All enumeration types
  - `pymetal.types` - Utility types (Origin, Size, Range, ClearColor)
  - `pymetal.compute` - Compute pipeline classes
  - `pymetal.graphics` - Graphics/render pipeline classes
  - `pymetal.advanced` - Advanced features (events, indirect commands, etc.)
  - `pymetal.shader` - Shader preprocessing utilities
- **Multi-Device Enumeration**
  - `copy_all_devices()` - Get all available Metal GPUs
  - New Device properties: `is_low_power`, `is_headless`, `is_removable`, `has_unified_memory`, `registry_id`, `recommended_max_working_set_size`, `max_buffer_length`
- **Shader Preprocessing Utilities** (`pymetal.shader`)
  - `ShaderPreprocessor` - #include, #define, #ifdef/#ifndef support
  - `ShaderTemplate` - Template-based shader generation
  - `create_compute_kernel()` - Helper for generating compute kernels
  - `compute_shader_hash()` - For shader caching

### Changed

- Test suite expanded from 41 to 110 tests
- Project status upgraded from BETA to STABLE

## [0.1.3]

### Fixed

- Fixed versions mismatch between code and pyproject.toml

## [0.1.2]

### Added

- Added `release` commands for multi-python version wheel building for pypi
  
## [0.1.1]

### Added

- Makefile commands for code quality tools:
  - `make lint` / `make lint-fix` - ruff linting
  - `make format` / `make format-check` - ruff formatting
  - `make typecheck` - mypy type checking
  - `make check` - twine package validation
  - `make publish-test` / `make publish` - PyPI publishing
- mypy configuration in pyproject.toml with overrides for compiled extensions
- PEP 561 `py.typed` marker for type checker support

### Fixed

- Lint issues in test files (unused variables, boolean comparisons)

## [0.1.0]

### Added

- Initial release of PyMetal
- **Compute Pipeline**
  - Device management and command queues
  - Buffer allocation and management
  - Shader compilation from Metal Shading Language source
  - Compute pipeline creation and execution
  - Thread group configuration and dispatch
  - Zero-copy NumPy buffer integration
- **Graphics Pipeline**
  - Texture creation and management
  - Render pipeline state with vertex/fragment shaders
  - Render pass descriptors with color/depth attachments
  - Sampler states for texture filtering
  - Offscreen rendering
  - Vertex descriptors and buffer layouts
  - Depth/stencil testing
  - Blit command encoder for memory operations
  - Heap-based resource allocation
  - Fence synchronization
  - Metal layer integration
- **Advanced Features**
  - Event system for fine-grained synchronization
  - Shared events for cross-process coordination
  - Argument buffers for efficient resource binding
  - Indirect command buffers for GPU-driven rendering
  - Binary archives for pipeline caching
  - Capture scopes for Xcode GPU debugging integration
- Example programs demonstrating all features
- Comprehensive test suite (41 tests)

[Unreleased]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shakfu/pymetal-cpp/releases/tag/v0.1.0
