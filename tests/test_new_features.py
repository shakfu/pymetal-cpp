"""
Tests for new features:
- Namespace organization (submodules)
- Multi-device enumeration
- Shader preprocessing
"""

import pytest
import pymetal as pm


class TestNamespaceOrganization:
    """Test that submodules are properly organized."""

    def test_enums_submodule_exists(self):
        """Test that pymetal.enums module exists."""
        assert hasattr(pm, "enums")
        from pymetal import enums
        assert hasattr(enums, "StorageMode")
        assert hasattr(enums, "PixelFormat")
        assert hasattr(enums, "ResourceStorageModeShared")

    def test_types_submodule_exists(self):
        """Test that pymetal.types module exists."""
        assert hasattr(pm, "types")
        from pymetal import types
        assert hasattr(types, "Origin")
        assert hasattr(types, "Size")
        assert hasattr(types, "Range")
        assert hasattr(types, "ClearColor")

    def test_compute_submodule_exists(self):
        """Test that pymetal.compute module exists."""
        assert hasattr(pm, "compute")
        from pymetal import compute
        assert hasattr(compute, "ComputePipelineState")
        assert hasattr(compute, "ComputeCommandEncoder")

    def test_graphics_submodule_exists(self):
        """Test that pymetal.graphics module exists."""
        assert hasattr(pm, "graphics")
        from pymetal import graphics
        assert hasattr(graphics, "Texture")
        assert hasattr(graphics, "RenderPipelineState")
        assert hasattr(graphics, "SamplerState")

    def test_advanced_submodule_exists(self):
        """Test that pymetal.advanced module exists."""
        assert hasattr(pm, "advanced")
        from pymetal import advanced
        assert hasattr(advanced, "Event")
        assert hasattr(advanced, "Heap")
        assert hasattr(advanced, "BlitCommandEncoder")

    def test_shader_submodule_exists(self):
        """Test that pymetal.shader module exists."""
        assert hasattr(pm, "shader")
        from pymetal import shader
        assert hasattr(shader, "ShaderPreprocessor")
        assert hasattr(shader, "ShaderTemplate")
        assert hasattr(shader, "create_compute_kernel")

    def test_backward_compatibility(self):
        """Test that top-level imports still work."""
        # These should still be available at the top level
        assert pm.Device is not None
        assert pm.create_system_default_device is not None
        assert pm.StorageMode is not None
        assert pm.ComputePipelineState is not None
        assert pm.Texture is not None


class TestMultiDeviceEnumeration:
    """Test multi-device enumeration feature."""

    def test_copy_all_devices_exists(self):
        """Test that copy_all_devices function exists."""
        assert hasattr(pm, "copy_all_devices")

    def test_copy_all_devices_returns_list(self):
        """Test that copy_all_devices returns a list."""
        devices = pm.copy_all_devices()
        assert isinstance(devices, list)

    def test_copy_all_devices_returns_devices(self):
        """Test that copy_all_devices returns Device objects."""
        devices = pm.copy_all_devices()
        assert len(devices) >= 1, "At least one device should be available"
        for device in devices:
            assert hasattr(device, "name")
            assert hasattr(device, "is_low_power")

    def test_default_device_in_all_devices(self):
        """Test that the default device is in the list of all devices."""
        default = pm.create_system_default_device()
        devices = pm.copy_all_devices()
        device_names = [d.name for d in devices]
        assert default.name in device_names

    def test_device_properties_for_selection(self):
        """Test that device properties for selection are available."""
        device = pm.create_system_default_device()
        # These should not raise
        _ = device.is_low_power
        _ = device.is_headless
        _ = device.is_removable
        _ = device.has_unified_memory
        _ = device.registry_id
        _ = device.recommended_max_working_set_size
        _ = device.max_buffer_length

    def test_device_selection_by_power(self):
        """Test selecting devices by power characteristics."""
        devices = pm.copy_all_devices()
        # Filter by is_low_power property
        low_power = [d for d in devices if d.is_low_power]
        high_power = [d for d in devices if not d.is_low_power]
        # At least the default device should exist
        assert len(devices) >= 1


class TestShaderPreprocessor:
    """Test shader preprocessing utilities."""

    def test_preprocessor_creation(self):
        """Test creating a ShaderPreprocessor."""
        from pymetal.shader import ShaderPreprocessor
        pp = ShaderPreprocessor()
        assert pp is not None

    def test_define_macro(self):
        """Test defining macros."""
        from pymetal.shader import ShaderPreprocessor
        pp = ShaderPreprocessor()
        pp.define("BLOCK_SIZE", "256")

        source = """
        #include <metal_stdlib>
        kernel void test(uint idx [[thread_position_in_grid]]) {
            int size = BLOCK_SIZE;
        }
        """
        processed = pp.process(source, process_includes=False)
        assert "256" in processed
        assert "BLOCK_SIZE" not in processed  # Should be substituted

    def test_method_chaining(self):
        """Test that methods support chaining."""
        from pymetal.shader import ShaderPreprocessor
        pp = (
            ShaderPreprocessor()
            .define("A", "1")
            .define("B", "2")
            .define("C", "3")
        )
        source = "A + B + C"
        processed = pp.process(source, process_includes=False)
        assert "1 + 2 + 3" in processed

    def test_conditional_ifdef(self):
        """Test #ifdef preprocessing."""
        from pymetal.shader import ShaderPreprocessor
        pp = ShaderPreprocessor()
        pp.define("FEATURE_ENABLED")

        source = """
#ifdef FEATURE_ENABLED
int feature = 1;
#else
int feature = 0;
#endif
"""
        processed = pp.process(source, process_includes=False, process_defines=False)
        assert "int feature = 1;" in processed
        assert "int feature = 0;" not in processed

    def test_conditional_ifndef(self):
        """Test #ifndef preprocessing."""
        from pymetal.shader import ShaderPreprocessor
        pp = ShaderPreprocessor()
        # Don't define FEATURE_DISABLED

        source = """
#ifndef FEATURE_DISABLED
int feature = 1;
#else
int feature = 0;
#endif
"""
        processed = pp.process(source, process_includes=False, process_defines=False)
        assert "int feature = 1;" in processed
        assert "int feature = 0;" not in processed


class TestShaderTemplate:
    """Test shader template functionality."""

    def test_template_creation(self):
        """Test creating a ShaderTemplate."""
        from pymetal.shader import ShaderTemplate
        template = ShaderTemplate("kernel void {name}() {{}}")
        assert template is not None

    def test_template_render(self):
        """Test rendering a template."""
        from pymetal.shader import ShaderTemplate
        template = ShaderTemplate("kernel void {name}() {{}}")
        source = template.render(name="my_kernel")
        assert "kernel void my_kernel()" in source

    def test_template_defaults(self):
        """Test template defaults."""
        from pymetal.shader import ShaderTemplate
        template = ShaderTemplate("kernel void {name}(device {type}* data) {{}}")
        template.set_default("type", "float")
        source = template.render(name="my_kernel")
        assert "device float* data" in source

    def test_template_override_defaults(self):
        """Test overriding template defaults."""
        from pymetal.shader import ShaderTemplate
        template = ShaderTemplate("device {type}* data")
        template.set_default("type", "float")
        source = template.render(type="int")
        assert "device int* data" in source


class TestCreateComputeKernel:
    """Test the create_compute_kernel helper."""

    def test_basic_kernel_generation(self):
        """Test generating a basic kernel."""
        from pymetal.shader import create_compute_kernel

        source = create_compute_kernel(
            name="double_values",
            body="data[idx] = data[idx] * 2.0;",
            buffers=[("data", "float", "readwrite")],
        )

        assert "kernel void double_values" in source
        assert "device float* data" in source
        assert "thread_position_in_grid" in source
        assert "#include <metal_stdlib>" in source

    def test_read_only_buffer(self):
        """Test generating a kernel with read-only buffer."""
        from pymetal.shader import create_compute_kernel

        source = create_compute_kernel(
            name="copy_values",
            body="out[idx] = in[idx];",
            buffers=[
                ("in", "float", "read"),
                ("out", "float", "write"),
            ],
        )

        assert "device const float* in" in source
        assert "device float* out" in source

    def test_compile_generated_kernel(self):
        """Test that a generated kernel actually compiles."""
        from pymetal.shader import create_compute_kernel

        source = create_compute_kernel(
            name="square_values",
            body="data[idx] = data[idx] * data[idx];",
            buffers=[("data", "float", "readwrite")],
        )

        device = pm.create_system_default_device()
        # This should not raise
        library = device.new_library_with_source(source)
        function = library.new_function("square_values")
        assert function is not None


class TestShaderHash:
    """Test shader hashing utility."""

    def test_compute_shader_hash(self):
        """Test computing shader hash."""
        from pymetal.shader import compute_shader_hash

        source1 = "kernel void test() {}"
        source2 = "kernel void test() {}"
        source3 = "kernel void test2() {}"

        hash1 = compute_shader_hash(source1)
        hash2 = compute_shader_hash(source2)
        hash3 = compute_shader_hash(source3)

        assert hash1 == hash2  # Same source = same hash
        assert hash1 != hash3  # Different source = different hash
        assert len(hash1) == 64  # SHA-256 hex = 64 chars
