# Changelog

All notable changes to PyMetal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2]

### Added
  - release commands for pypi
  
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
- **Phase 1: Compute Pipeline**
  - Device management and command queues
  - Buffer allocation and management
  - Shader compilation from Metal Shading Language source
  - Compute pipeline creation and execution
  - Thread group configuration and dispatch
  - Zero-copy NumPy buffer integration
- **Phase 2: Graphics Pipeline**
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
- **Phase 3: Advanced Features**
  - Event system for fine-grained synchronization
  - Shared events for cross-process coordination
  - Argument buffers for efficient resource binding
  - Indirect command buffers for GPU-driven rendering
  - Binary archives for pipeline caching
  - Capture scopes for Xcode GPU debugging integration
- Example programs demonstrating all features
- Comprehensive test suite (41 tests)

[Unreleased]: https://github.com/shakfu/pymetal-cpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shakfu/pymetal-cpp/releases/tag/v0.1.0
