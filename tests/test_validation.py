"""
Tests for input validation in PyMetal.

These tests verify that the custom exception types are raised correctly
when invalid input is provided to Metal operations.
"""

import pytest
import pymetal as pm


@pytest.fixture
def device():
    """Create a Metal device for testing."""
    return pm.create_system_default_device()


@pytest.fixture
def queue(device):
    """Create a command queue for testing."""
    return device.new_command_queue()


@pytest.fixture
def simple_pipeline(device):
    """Create a simple compute pipeline for testing."""
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void noop(device float* data [[buffer(0)]],
                    uint idx [[thread_position_in_grid]]) {
    }
    """
    library = device.new_library_with_source(shader_source)
    function = library.new_function("noop")
    return device.new_compute_pipeline_state(function)


class TestExceptionHierarchy:
    """Test that the exception hierarchy is set up correctly."""

    def test_metal_error_is_base_class(self):
        """Verify MetalError is the base for all Metal exceptions."""
        assert issubclass(pm.CompileError, pm.MetalError)
        assert issubclass(pm.PipelineError, pm.MetalError)
        assert issubclass(pm.ResourceError, pm.MetalError)
        assert issubclass(pm.ValidationError, pm.MetalError)

    def test_exceptions_are_exception_subclasses(self):
        """Verify all exceptions inherit from Exception."""
        assert issubclass(pm.MetalError, Exception)
        assert issubclass(pm.CompileError, Exception)
        assert issubclass(pm.PipelineError, Exception)
        assert issubclass(pm.ResourceError, Exception)
        assert issubclass(pm.ValidationError, Exception)


class TestCompileError:
    """Test CompileError exceptions."""

    def test_compile_error_on_invalid_shader(self, device):
        """CompileError should be raised for invalid shader code."""
        invalid_shader = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void bad(INVALID_TYPE* x [[buffer(0)]]) {}
        """
        with pytest.raises(pm.CompileError) as exc_info:
            device.new_library_with_source(invalid_shader)

        # Verify error message contains useful info
        assert "compilation failed" in str(exc_info.value).lower()

    def test_compile_error_caught_as_metal_error(self, device):
        """CompileError should be catchable as MetalError."""
        invalid_shader = "not valid metal code"
        with pytest.raises(pm.MetalError):
            device.new_library_with_source(invalid_shader)


class TestValidationError:
    """Test ValidationError exceptions for input validation."""

    def test_buffer_index_too_high_compute(self, device, queue, simple_pipeline):
        """ValidationError should be raised for buffer index > 31."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_pipeline)

        with pytest.raises(pm.ValidationError) as exc_info:
            encoder.set_buffer(buffer, 0, 32)  # Index 32 is too high

        assert "buffer index" in str(exc_info.value).lower()
        assert "32" in str(exc_info.value)
        encoder.end_encoding()

    def test_valid_buffer_index_works(self, device, queue, simple_pipeline):
        """Valid buffer indices should work without error."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_pipeline)

        # Index 31 should work (maximum valid index)
        encoder.set_buffer(buffer, 0, 31)
        # Index 0 should work (minimum valid index)
        encoder.set_buffer(buffer, 0, 0)
        encoder.end_encoding()

    def test_zero_threadgroup_dimensions(self, device, queue, simple_pipeline):
        """ValidationError should be raised for zero threadgroup dimensions."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_pipeline)
        encoder.set_buffer(buffer, 0, 0)

        with pytest.raises(pm.ValidationError) as exc_info:
            encoder.dispatch_threadgroups(0, 1, 1, 1, 1, 1)

        assert (
            "0" in str(exc_info.value) or "greater than" in str(exc_info.value).lower()
        )
        encoder.end_encoding()

    def test_zero_threads_per_group(self, device, queue, simple_pipeline):
        """ValidationError should be raised for zero threads per group."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_pipeline)
        encoder.set_buffer(buffer, 0, 0)

        with pytest.raises(pm.ValidationError):
            encoder.dispatch_threadgroups(1, 1, 1, 0, 1, 1)

        encoder.end_encoding()

    def test_validation_error_caught_as_metal_error(
        self, device, queue, simple_pipeline
    ):
        """ValidationError should be catchable as MetalError."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_pipeline)

        with pytest.raises(pm.MetalError):
            encoder.set_buffer(buffer, 0, 100)  # Way too high

        encoder.end_encoding()


class TestRenderEncoderValidation:
    """Test validation in render command encoder."""

    @pytest.fixture
    def render_setup(self, device):
        """Create a simple render pipeline for testing."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
        };

        vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
            VertexOut out;
            out.position = float4(0.0, 0.0, 0.0, 1.0);
            return out;
        }

        fragment float4 fragment_main() {
            return float4(1.0, 0.0, 0.0, 1.0);
        }
        """
        library = device.new_library_with_source(shader_source)
        vertex_func = library.new_function("vertex_main")
        fragment_func = library.new_function("fragment_main")

        # Create texture for render target
        tex_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, 64, 64, False
        )
        texture = device.new_texture(tex_desc)

        # Create render pipeline
        pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipeline_desc.vertex_function = vertex_func
        pipeline_desc.fragment_function = fragment_func
        pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm

        pipeline = device.new_render_pipeline_state(pipeline_desc)

        # Create render pass
        render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
        color_att = render_pass.color_attachment(0)
        color_att.texture = texture
        color_att.load_action = pm.LoadAction.Clear
        color_att.store_action = pm.StoreAction.Store

        return {
            "pipeline": pipeline,
            "render_pass": render_pass,
            "texture": texture,
        }

    def test_render_encoder_buffer_index_validation(self, device, queue, render_setup):
        """ValidationError should be raised for invalid buffer index in render encoder."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.render_command_encoder(render_setup["render_pass"])
        encoder.set_render_pipeline_state(render_setup["pipeline"])

        with pytest.raises(pm.ValidationError) as exc_info:
            encoder.set_vertex_buffer(buffer, 0, 32)  # Too high

        assert "buffer index" in str(exc_info.value).lower()
        encoder.end_encoding()

    def test_render_encoder_texture_index_validation(self, device, queue, render_setup):
        """ValidationError should be raised for invalid texture index."""
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.render_command_encoder(render_setup["render_pass"])
        encoder.set_render_pipeline_state(render_setup["pipeline"])

        with pytest.raises(pm.ValidationError) as exc_info:
            encoder.set_fragment_texture(render_setup["texture"], 32)  # Too high

        assert "texture index" in str(exc_info.value).lower()
        encoder.end_encoding()
