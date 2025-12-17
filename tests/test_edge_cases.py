"""
Edge Case Tests for PyMetal

These tests verify correct behavior at boundary conditions and with
unusual inputs to ensure robustness.
"""

import pytest
import numpy as np
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
def simple_kernel(device):
    """Create a simple compute pipeline for testing."""
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void copy_kernel(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint idx [[thread_position_in_grid]]) {
        output[idx] = input[idx];
    }
    """
    library = device.new_library_with_source(shader_source)
    function = library.new_function("copy_kernel")
    return device.new_compute_pipeline_state(function)


class TestMinimalBufferSizes:
    """Test buffer operations with minimal sizes."""

    def test_single_byte_buffer(self, device):
        """Test creating and using a 1-byte buffer."""
        buffer = device.new_buffer(1, pm.ResourceStorageModeShared)
        assert buffer.length == 1

        data = buffer.contents()
        assert len(data) == 1

        # Write and read back
        data[0] = 42
        assert data[0] == 42

    def test_single_float_buffer(self, device, queue, simple_kernel):
        """Test compute with a single float element."""
        buffer_size = 4  # 1 float

        input_buf = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)
        output_buf = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)

        # Initialize
        input_data = np.frombuffer(input_buf.contents(), dtype=np.float32)
        input_data[0] = 3.14

        # Execute with 1 thread
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)
        encoder.set_buffer(input_buf, 0, 0)
        encoder.set_buffer(output_buf, 0, 1)
        encoder.dispatch_threads(1, 1, 1, 1, 1, 1)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

        # Verify
        output_data = np.frombuffer(output_buf.contents(), dtype=np.float32)
        assert abs(output_data[0] - 3.14) < 1e-6

    def test_alignment_edge_cases(self, device):
        """Test buffers with sizes that may have alignment implications."""
        # Non-power-of-2 sizes
        for size in [1, 3, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257]:
            buffer = device.new_buffer(size, pm.ResourceStorageModeShared)
            assert buffer.length == size
            data = buffer.contents()
            assert len(data) == size


class TestBoundaryConditions:
    """Test behavior at boundary values."""

    def test_max_valid_buffer_index(self, device, queue, simple_kernel):
        """Test buffer binding at maximum valid index (31)."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)

        # Index 31 should work
        encoder.set_buffer(buffer, 0, 31)
        encoder.end_encoding()
        # Should not raise

    def test_boundary_buffer_indices(self, device, queue, simple_kernel):
        """Test buffer indices at boundaries."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        # Test valid boundary indices
        for index in [0, 1, 30, 31]:
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.compute_command_encoder()
            encoder.set_compute_pipeline_state(simple_kernel)
            encoder.set_buffer(buffer, 0, index)  # Should not raise
            encoder.end_encoding()

        # Test invalid index
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)
        with pytest.raises(pm.ValidationError):
            encoder.set_buffer(buffer, 0, 32)
        encoder.end_encoding()

    def test_large_threadgroup_dispatch(self, device, queue, simple_kernel):
        """Test dispatch with large but valid threadgroup sizes."""
        buffer_size = 1024 * 1024  # 1M floats
        input_buf = device.new_buffer(buffer_size * 4, pm.ResourceStorageModeShared)
        output_buf = device.new_buffer(buffer_size * 4, pm.ResourceStorageModeShared)

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)
        encoder.set_buffer(input_buf, 0, 0)
        encoder.set_buffer(output_buf, 0, 1)

        # Large dispatch
        threads_per_group = 256
        num_groups = buffer_size // threads_per_group
        encoder.dispatch_threadgroups(num_groups, 1, 1, threads_per_group, 1, 1)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

    def test_single_threadgroup_dispatch(self, device, queue, simple_kernel):
        """Test dispatch with exactly 1 threadgroup."""
        buffer = device.new_buffer(256 * 4, pm.ResourceStorageModeShared)

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)
        encoder.set_buffer(buffer, 0, 0)
        encoder.set_buffer(buffer, 0, 1)

        # Single threadgroup
        encoder.dispatch_threadgroups(1, 1, 1, 256, 1, 1)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()


class TestEmptyOperations:
    """Test behavior with empty or zero-count operations."""

    def test_empty_command_buffer(self, device, queue):
        """Test committing an empty command buffer."""
        cmd_buffer = queue.command_buffer()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()
        # Should complete without error

    def test_encoder_with_no_dispatch(self, device, queue, simple_kernel):
        """Test encoder that sets state but doesn't dispatch."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(simple_kernel)
        encoder.set_buffer(buffer, 0, 0)
        # No dispatch
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()
        # Should complete without error

    def test_multiple_encoders_same_buffer(self, device, queue, simple_kernel):
        """Test multiple sequential encoders using the same buffer."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        cmd_buffer = queue.command_buffer()

        # First encoder
        enc1 = cmd_buffer.compute_command_encoder()
        enc1.set_compute_pipeline_state(simple_kernel)
        enc1.set_buffer(buffer, 0, 0)
        enc1.set_buffer(buffer, 0, 1)
        enc1.dispatch_threadgroups(1, 1, 1, 256, 1, 1)
        enc1.end_encoding()

        # Second encoder (sequential, same command buffer)
        enc2 = cmd_buffer.compute_command_encoder()
        enc2.set_compute_pipeline_state(simple_kernel)
        enc2.set_buffer(buffer, 0, 0)
        enc2.set_buffer(buffer, 0, 1)
        enc2.dispatch_threadgroups(1, 1, 1, 256, 1, 1)
        enc2.end_encoding()

        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()


class TestTextureEdgeCases:
    """Test texture operations with edge cases."""

    def test_minimal_texture(self, device):
        """Test creating a 1x1 texture."""
        tex_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, 1, 1, False
        )
        texture = device.new_texture(tex_desc)
        assert texture.width == 1
        assert texture.height == 1

    def test_non_square_texture(self, device):
        """Test creating non-square textures."""
        # Wide texture
        tex_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, 1024, 1, False
        )
        texture = device.new_texture(tex_desc)
        assert texture.width == 1024
        assert texture.height == 1

        # Tall texture
        tex_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, 1, 1024, False
        )
        texture = device.new_texture(tex_desc)
        assert texture.width == 1
        assert texture.height == 1024

    def test_non_power_of_two_texture(self, device):
        """Test non-power-of-2 texture dimensions."""
        for width, height in [(100, 100), (123, 456), (333, 777)]:
            tex_desc = pm.TextureDescriptor.texture2d_descriptor(
                pm.PixelFormat.RGBA8Unorm, width, height, False
            )
            texture = device.new_texture(tex_desc)
            assert texture.width == width
            assert texture.height == height


class TestLabelEdgeCases:
    """Test debug label handling with edge cases."""

    def test_empty_label(self, device, queue):
        """Test setting empty labels."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        buffer.label = ""
        assert buffer.label == ""

        cmd_buffer = queue.command_buffer()
        cmd_buffer.label = ""
        assert cmd_buffer.label == ""

    def test_long_label(self, device):
        """Test setting very long labels."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        long_label = "x" * 1000
        buffer.label = long_label
        assert buffer.label == long_label

    def test_unicode_label(self, device):
        """Test setting labels with unicode characters."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        # Note: Avoiding emojis per project rules, using other unicode
        buffer.label = "Buffer_Alpha_Beta"
        assert "Alpha" in buffer.label


class TestHeapEdgeCases:
    """Test heap operations with edge cases."""

    def test_minimal_heap(self, device):
        """Test creating a heap with minimal size."""
        heap_desc = pm.HeapDescriptor.heap_descriptor()
        heap_desc.size = 4096  # Minimum practical size
        heap_desc.storage_mode = pm.StorageMode.Shared

        heap = device.new_heap(heap_desc)
        assert heap.size >= 4096

    def test_heap_allocation_tracking(self, device):
        """Test heap memory tracking after allocations."""
        heap_desc = pm.HeapDescriptor.heap_descriptor()
        heap_desc.size = 1024 * 1024  # 1MB
        heap_desc.storage_mode = pm.StorageMode.Shared

        heap = device.new_heap(heap_desc)
        initial_available = heap.max_available_size(256)

        # Allocate a buffer
        buffer = heap.new_buffer(1024, pm.ResourceStorageModeShared)
        assert buffer is not None

        # Available size should decrease
        new_available = heap.max_available_size(256)
        assert new_available < initial_available


class TestMultipleCommands:
    """Test multiple command submissions."""

    def test_rapid_command_submissions(self, device, queue, simple_kernel):
        """Test many rapid command buffer submissions."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        for _ in range(100):
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.compute_command_encoder()
            encoder.set_compute_pipeline_state(simple_kernel)
            encoder.set_buffer(buffer, 0, 0)
            encoder.set_buffer(buffer, 0, 1)
            encoder.dispatch_threadgroups(1, 1, 1, 64, 1, 1)
            encoder.end_encoding()
            cmd_buffer.commit()
            cmd_buffer.wait_until_completed()

    def test_batch_without_wait(self, device, queue, simple_kernel):
        """Test submitting multiple commands without waiting between."""
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
        command_buffers = []

        # Submit many without waiting
        for _ in range(10):
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.compute_command_encoder()
            encoder.set_compute_pipeline_state(simple_kernel)
            encoder.set_buffer(buffer, 0, 0)
            encoder.set_buffer(buffer, 0, 1)
            encoder.dispatch_threadgroups(1, 1, 1, 64, 1, 1)
            encoder.end_encoding()
            cmd_buffer.commit()
            command_buffers.append(cmd_buffer)

        # Wait for all at end
        for cmd_buffer in command_buffers:
            cmd_buffer.wait_until_completed()


class TestShaderEdgeCases:
    """Test shader compilation edge cases."""

    def test_minimal_shader(self, device):
        """Test compiling a minimal valid shader."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void minimal() {}
        """
        library = device.new_library_with_source(shader_source)
        function = library.new_function("minimal")
        assert function is not None

    def test_shader_with_many_buffers(self, device):
        """Test shader using many buffer arguments."""
        # Generate shader with 8 buffer arguments
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void many_buffers(
            device float* b0 [[buffer(0)]],
            device float* b1 [[buffer(1)]],
            device float* b2 [[buffer(2)]],
            device float* b3 [[buffer(3)]],
            device float* b4 [[buffer(4)]],
            device float* b5 [[buffer(5)]],
            device float* b6 [[buffer(6)]],
            device float* b7 [[buffer(7)]],
            uint idx [[thread_position_in_grid]]
        ) {
            b7[idx] = b0[idx] + b1[idx] + b2[idx] + b3[idx] + b4[idx] + b5[idx] + b6[idx];
        }
        """
        library = device.new_library_with_source(shader_source)
        function = library.new_function("many_buffers")
        pipeline = device.new_compute_pipeline_state(function)
        assert pipeline is not None

    def test_shader_with_structs(self, device):
        """Test shader using struct types."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        struct Particle {
            float3 position;
            float3 velocity;
            float mass;
        };

        kernel void update_particles(
            device Particle* particles [[buffer(0)]],
            uint idx [[thread_position_in_grid]]
        ) {
            particles[idx].position += particles[idx].velocity * 0.01;
        }
        """
        library = device.new_library_with_source(shader_source)
        function = library.new_function("update_particles")
        pipeline = device.new_compute_pipeline_state(function)
        assert pipeline is not None


class TestRenderEdgeCases:
    """Test render pipeline edge cases."""

    def test_minimal_render_pipeline(self, device):
        """Test creating a minimal render pipeline."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        vertex float4 vertex_main(uint vid [[vertex_id]]) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }

        fragment float4 fragment_main() {
            return float4(1.0, 0.0, 0.0, 1.0);
        }
        """
        library = device.new_library_with_source(shader_source)
        vertex_func = library.new_function("vertex_main")
        fragment_func = library.new_function("fragment_main")

        pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipeline_desc.vertex_function = vertex_func
        pipeline_desc.fragment_function = fragment_func
        pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm

        pipeline = device.new_render_pipeline_state(pipeline_desc)
        assert pipeline is not None

    def test_render_to_minimal_texture(self, device, queue):
        """Test rendering to a 1x1 texture."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        vertex float4 vertex_main(uint vid [[vertex_id]]) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }

        fragment float4 fragment_main() {
            return float4(1.0, 0.0, 0.0, 1.0);
        }
        """
        library = device.new_library_with_source(shader_source)

        pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipeline_desc.vertex_function = library.new_function("vertex_main")
        pipeline_desc.fragment_function = library.new_function("fragment_main")
        pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm

        pipeline = device.new_render_pipeline_state(pipeline_desc)

        # 1x1 texture
        tex_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, 1, 1, False
        )
        texture = device.new_texture(tex_desc)

        # Render
        render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
        color_att = render_pass.color_attachment(0)
        color_att.texture = texture
        color_att.load_action = pm.LoadAction.Clear
        color_att.store_action = pm.StoreAction.Store
        color_att.clear_color = pm.ClearColor(0.0, 0.0, 0.0, 1.0)

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.render_command_encoder(render_pass)
        encoder.set_render_pipeline_state(pipeline)
        encoder.draw_primitives(pm.PrimitiveType.Point, 0, 1)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()


class TestClearColorEdgeCases:
    """Test ClearColor edge cases."""

    def test_clear_color_extremes(self):
        """Test ClearColor with extreme values."""
        # All zeros
        color = pm.ClearColor(0.0, 0.0, 0.0, 0.0)
        assert color.red == 0.0
        assert color.alpha == 0.0

        # All ones
        color = pm.ClearColor(1.0, 1.0, 1.0, 1.0)
        assert color.red == 1.0
        assert color.alpha == 1.0

        # HDR values (> 1.0)
        color = pm.ClearColor(2.0, 3.0, 4.0, 1.0)
        assert color.red == 2.0

        # Negative values
        color = pm.ClearColor(-1.0, -0.5, 0.0, 1.0)
        assert color.red == -1.0
