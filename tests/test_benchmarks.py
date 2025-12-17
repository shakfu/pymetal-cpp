"""
Performance Benchmark Tests for PyMetal

These tests establish baseline performance metrics and can detect regressions.
They measure throughput for key operations like buffer allocation, shader
compilation, and compute kernel execution.

Run with: pytest tests/test_benchmarks.py -v
"""

import time
import pytest
import numpy as np
import pymetal as pm


# Minimum expected performance thresholds (operations per second or bytes per second)
# These are conservative baselines that should pass on any Apple Silicon Mac
MIN_BUFFER_ALLOC_OPS_PER_SEC = 1000  # Buffer allocations per second
MIN_SHADER_COMPILE_OPS_PER_SEC = 10  # Shader compilations per second
MIN_COMPUTE_THROUGHPUT_GBPS = 1.0  # GB/s for simple compute operations


@pytest.fixture(scope="module")
def device():
    """Shared device for all benchmark tests."""
    return pm.create_system_default_device()


@pytest.fixture(scope="module")
def queue(device):
    """Shared command queue for all benchmark tests."""
    return device.new_command_queue()


class TestBufferAllocationPerformance:
    """Benchmark buffer allocation operations."""

    def test_small_buffer_allocation_throughput(self, device):
        """Test throughput of small buffer allocations (1KB)."""
        buffer_size = 1024  # 1KB
        num_iterations = 500
        buffers = []

        start = time.perf_counter()
        for _ in range(num_iterations):
            buf = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)
            buffers.append(buf)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_iterations / elapsed
        print(f"\nSmall buffer allocation: {ops_per_sec:.0f} ops/sec")
        assert ops_per_sec > MIN_BUFFER_ALLOC_OPS_PER_SEC, (
            f"Buffer allocation too slow: {ops_per_sec:.0f} ops/sec "
            f"(expected > {MIN_BUFFER_ALLOC_OPS_PER_SEC})"
        )

    def test_large_buffer_allocation_throughput(self, device):
        """Test throughput of large buffer allocations (1MB)."""
        buffer_size = 1024 * 1024  # 1MB
        num_iterations = 100
        buffers = []

        start = time.perf_counter()
        for _ in range(num_iterations):
            buf = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)
            buffers.append(buf)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_iterations / elapsed
        mb_per_sec = (buffer_size * num_iterations) / (elapsed * 1024 * 1024)
        print(
            f"\nLarge buffer allocation: {ops_per_sec:.0f} ops/sec, {mb_per_sec:.0f} MB/sec"
        )
        # Large buffers are slower, so use a lower threshold
        assert ops_per_sec > MIN_BUFFER_ALLOC_OPS_PER_SEC / 10


class TestShaderCompilationPerformance:
    """Benchmark shader compilation operations."""

    def test_simple_shader_compilation_throughput(self, device):
        """Test throughput of simple shader compilations."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void simple_kernel(device float* data [[buffer(0)]],
                                 uint idx [[thread_position_in_grid]]) {
            data[idx] = data[idx] * 2.0;
        }
        """
        num_iterations = 20

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = device.new_library_with_source(shader_source)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_iterations / elapsed
        print(f"\nShader compilation: {ops_per_sec:.1f} ops/sec")
        assert ops_per_sec > MIN_SHADER_COMPILE_OPS_PER_SEC, (
            f"Shader compilation too slow: {ops_per_sec:.1f} ops/sec "
            f"(expected > {MIN_SHADER_COMPILE_OPS_PER_SEC})"
        )


class TestComputeKernelPerformance:
    """Benchmark compute kernel execution."""

    def test_vector_add_throughput(self, device, queue):
        """Test throughput of a simple vector addition kernel."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* c [[buffer(2)]],
                              uint idx [[thread_position_in_grid]]) {
            c[idx] = a[idx] + b[idx];
        }
        """

        # Setup
        size = 1024 * 1024  # 1M elements
        bytes_size = size * 4  # float32

        library = device.new_library_with_source(shader_source)
        function = library.new_function("vector_add")
        pipeline = device.new_compute_pipeline_state(function)

        a_buffer = device.new_buffer(bytes_size, pm.ResourceStorageModeShared)
        b_buffer = device.new_buffer(bytes_size, pm.ResourceStorageModeShared)
        c_buffer = device.new_buffer(bytes_size, pm.ResourceStorageModeShared)

        # Initialize data
        a_data = np.frombuffer(a_buffer.contents(), dtype=np.float32)
        b_data = np.frombuffer(b_buffer.contents(), dtype=np.float32)
        a_data[:] = np.random.randn(size).astype(np.float32)
        b_data[:] = np.random.randn(size).astype(np.float32)

        num_iterations = 50
        threads_per_group = 256
        num_groups = (size + threads_per_group - 1) // threads_per_group

        # Warmup
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.compute_command_encoder()
        encoder.set_compute_pipeline_state(pipeline)
        encoder.set_buffer(a_buffer, 0, 0)
        encoder.set_buffer(b_buffer, 0, 1)
        encoder.set_buffer(c_buffer, 0, 2)
        encoder.dispatch_threadgroups(num_groups, 1, 1, threads_per_group, 1, 1)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.compute_command_encoder()
            encoder.set_compute_pipeline_state(pipeline)
            encoder.set_buffer(a_buffer, 0, 0)
            encoder.set_buffer(b_buffer, 0, 1)
            encoder.set_buffer(c_buffer, 0, 2)
            encoder.dispatch_threadgroups(num_groups, 1, 1, threads_per_group, 1, 1)
            encoder.end_encoding()
            cmd_buffer.commit()
            cmd_buffer.wait_until_completed()
        elapsed = time.perf_counter() - start

        # Calculate throughput (3 buffers read/written per iteration)
        total_bytes = 3 * bytes_size * num_iterations
        throughput_gbps = total_bytes / (elapsed * 1e9)
        ops_per_sec = num_iterations / elapsed

        print(
            f"\nVector add: {throughput_gbps:.2f} GB/s, {ops_per_sec:.0f} kernel executions/sec"
        )
        assert throughput_gbps > MIN_COMPUTE_THROUGHPUT_GBPS, (
            f"Compute throughput too low: {throughput_gbps:.2f} GB/s "
            f"(expected > {MIN_COMPUTE_THROUGHPUT_GBPS})"
        )

    def test_memory_copy_throughput(self, device, queue):
        """Test memory copy throughput using blit encoder."""
        size = 16 * 1024 * 1024  # 16MB

        src_buffer = device.new_buffer(size, pm.ResourceStorageModeShared)
        dst_buffer = device.new_buffer(size, pm.ResourceStorageModeShared)

        # Initialize source
        src_data = np.frombuffer(src_buffer.contents(), dtype=np.uint8)
        src_data[:] = np.random.randint(0, 256, size, dtype=np.uint8)

        num_iterations = 20

        # Warmup
        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.blit_command_encoder()
        encoder.copy_from_buffer(src_buffer, 0, dst_buffer, 0, size)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.blit_command_encoder()
            encoder.copy_from_buffer(src_buffer, 0, dst_buffer, 0, size)
            encoder.end_encoding()
            cmd_buffer.commit()
            cmd_buffer.wait_until_completed()
        elapsed = time.perf_counter() - start

        total_bytes = size * num_iterations
        throughput_gbps = total_bytes / (elapsed * 1e9)

        print(f"\nMemory copy: {throughput_gbps:.2f} GB/s")
        assert throughput_gbps > MIN_COMPUTE_THROUGHPUT_GBPS


class TestValidationPerformance:
    """Test that validation doesn't add significant overhead."""

    def test_validation_overhead(self, device, queue):
        """Ensure input validation doesn't add significant overhead."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void noop(device float* data [[buffer(0)]],
                        uint idx [[thread_position_in_grid]]) {
            // Do nothing
        }
        """

        library = device.new_library_with_source(shader_source)
        function = library.new_function("noop")
        pipeline = device.new_compute_pipeline_state(function)
        buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

        num_iterations = 1000

        # Time set_buffer calls (which now include validation)
        start = time.perf_counter()
        for _ in range(num_iterations):
            cmd_buffer = queue.command_buffer()
            encoder = cmd_buffer.compute_command_encoder()
            encoder.set_compute_pipeline_state(pipeline)
            # Multiple set_buffer calls to test validation overhead
            encoder.set_buffer(buffer, 0, 0)
            encoder.set_buffer(buffer, 0, 1)
            encoder.set_buffer(buffer, 0, 2)
            encoder.dispatch_threadgroups(1, 1, 1, 1, 1, 1)
            encoder.end_encoding()
            cmd_buffer.commit()
        elapsed = time.perf_counter() - start

        ops_per_sec = num_iterations / elapsed
        print(f"\nCommand encoding with validation: {ops_per_sec:.0f} ops/sec")
        # Should be able to do at least 500 command buffer submissions per second
        assert ops_per_sec > 500, (
            f"Command encoding too slow: {ops_per_sec:.0f} ops/sec - "
            "validation may be adding too much overhead"
        )
