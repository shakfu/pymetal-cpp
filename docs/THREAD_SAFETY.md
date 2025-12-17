# Thread Safety Guide for PyMetal

This document describes the thread safety characteristics of PyMetal objects and provides guidelines for safe multithreaded usage.

## Overview

PyMetal follows Apple's Metal threading model. Understanding which objects are thread-safe is crucial for writing correct concurrent GPU code.

## Thread Safety Summary

| Object | Thread-Safe? | Notes |
|--------|-------------|-------|
| Device | Yes | All methods can be called from any thread |
| CommandQueue | Yes | Multiple threads can create command buffers |
| CommandBuffer | **No** | Use from one thread only |
| ComputeCommandEncoder | **No** | Use from one thread only |
| RenderCommandEncoder | **No** | Use from one thread only |
| BlitCommandEncoder | **No** | Use from one thread only |
| Buffer | Partially | Contents accessible, but synchronize access |
| Texture | Partially | Similar to Buffer |
| Library | Yes | Functions can be retrieved concurrently |
| Pipeline States | Yes | Immutable after creation |

## Detailed Guidelines

### Device

The `Device` object is fully thread-safe. You can safely:

- Create resources (buffers, textures, pipelines) from multiple threads
- Create command queues from multiple threads
- Query device properties from any thread

```python
import threading
import pymetal as pm

device = pm.create_system_default_device()

def create_resources():
    # Safe: Device methods are thread-safe
    buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
    queue = device.new_command_queue()

threads = [threading.Thread(target=create_resources) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Command Queue

Command queues are thread-safe for creating command buffers:

```python
queue = device.new_command_queue()

def submit_work():
    # Safe: Creating command buffers is thread-safe
    cmd_buffer = queue.command_buffer()
    # ... encode and commit

# Multiple threads can submit to the same queue
```

### Command Buffers and Encoders

**Command buffers and encoders are NOT thread-safe.** Each command buffer should only be used by one thread at a time.

**Unsafe pattern:**

```python
# DON'T DO THIS - race condition!
cmd_buffer = queue.command_buffer()

def encode_part1():
    encoder = cmd_buffer.compute_command_encoder()  # Race!
    # ...

def encode_part2():
    encoder = cmd_buffer.compute_command_encoder()  # Race!
    # ...
```

**Safe pattern - separate command buffers:**

```python
def worker():
    # Safe: Each thread has its own command buffer
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()
    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(my_buffer, 0, 0)
    encoder.dispatch_threadgroups(1, 1, 1, 256, 1, 1)
    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

threads = [threading.Thread(target=worker) for _ in range(4)]
```

### Buffers and Textures

Buffer and texture objects themselves are thread-safe, but **you must synchronize access to their contents**:

**CPU-GPU synchronization:**

```python
# Thread 1: Write data
buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
data = np.frombuffer(buffer.contents(), dtype=np.float32)
data[:] = input_array

# Submit GPU work
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.compute_command_encoder()
# ... encode ...
cmd_buffer.commit()

# IMPORTANT: Wait before reading results!
cmd_buffer.wait_until_completed()

# Now safe to read
result = np.frombuffer(buffer.contents(), dtype=np.float32).copy()
```

**Multi-threaded CPU access:**

```python
import threading

buffer = device.new_buffer(4096, pm.ResourceStorageModeShared)
lock = threading.Lock()

def write_section(start, end, values):
    data = np.frombuffer(buffer.contents(), dtype=np.float32)
    with lock:  # Protect concurrent writes
        data[start:end] = values
```

### Pipeline States

Pipeline states (ComputePipelineState, RenderPipelineState) are immutable after creation and fully thread-safe. Multiple command buffers can use the same pipeline state concurrently:

```python
# Create once
pipeline = device.new_compute_pipeline_state(function)

def worker():
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()
    # Safe: Pipeline state is immutable
    encoder.set_compute_pipeline_state(pipeline)
    # ...
```

## GIL and Blocking Operations

PyMetal releases the Python GIL during blocking operations, allowing other Python threads to run:

- `command_buffer.wait_until_completed()` - Releases GIL
- `command_buffer.wait_until_scheduled()` - Releases GIL
- `device.new_library_with_source()` - Releases GIL during compilation

```python
import threading
import time

def gpu_work():
    cmd_buffer = queue.command_buffer()
    # ... encode work ...
    cmd_buffer.commit()
    # GIL released here - other threads can run
    cmd_buffer.wait_until_completed()

def cpu_work():
    # This can run while GPU work is waiting
    time.sleep(0.1)

t1 = threading.Thread(target=gpu_work)
t2 = threading.Thread(target=cpu_work)
t1.start()
t2.start()  # Can execute during wait_until_completed
```

## Best Practices

### 1. One Command Buffer Per Thread

Create command buffers on the thread that will use them:

```python
def process_batch(batch_data, shared_queue, shared_pipeline):
    cmd_buffer = shared_queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()
    encoder.set_compute_pipeline_state(shared_pipeline)
    # ... encode batch-specific work ...
    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()
```

### 2. Avoid Sharing Encoders

Never pass encoders between threads. Keep encoder lifetime within a single thread:

```python
# Good: Encoder created and used in same thread
def worker():
    cmd = queue.command_buffer()
    enc = cmd.compute_command_encoder()
    # ... use enc ...
    enc.end_encoding()
    cmd.commit()
```

### 3. Use Fences for GPU-GPU Synchronization

For synchronizing work between command buffers:

```python
fence = device.new_fence()

# First command buffer updates fence after work
cmd1 = queue.command_buffer()
enc1 = cmd1.compute_command_encoder()
# ... encode work ...
enc1.update_fence(fence)
enc1.end_encoding()
cmd1.commit()

# Second command buffer waits for fence
cmd2 = queue.command_buffer()
enc2 = cmd2.compute_command_encoder()
enc2.wait_for_fence(fence)
# ... encode work that depends on cmd1 ...
enc2.end_encoding()
cmd2.commit()
```

### 4. Use Events for Cross-Queue Synchronization

Events allow synchronization between command buffers on different queues:

```python
event = device.new_event()

# Queue 1
cmd1 = queue1.command_buffer()
cmd1.encode_signal_event(event, 1)
cmd1.commit()

# Queue 2
cmd2 = queue2.command_buffer()
cmd2.encode_wait_for_event(event, 1)
# ... work that depends on cmd1 ...
cmd2.commit()
```

## Common Pitfalls

### 1. Reading Before Completion

```python
# WRONG: Reading before GPU is done
cmd_buffer.commit()
result = np.frombuffer(buffer.contents(), dtype=np.float32)  # Race condition!

# CORRECT: Wait first
cmd_buffer.commit()
cmd_buffer.wait_until_completed()
result = np.frombuffer(buffer.contents(), dtype=np.float32)
```

### 2. Modifying Buffers During Execution

```python
# WRONG: Modifying buffer while GPU is using it
cmd_buffer.commit()
data = np.frombuffer(buffer.contents(), dtype=np.float32)
data[:] = new_values  # Race condition!

# CORRECT: Wait or use double buffering
cmd_buffer.commit()
cmd_buffer.wait_until_completed()
data[:] = new_values  # Safe now
```

### 3. Sharing Command Buffers

```python
# WRONG: Sharing command buffer between threads
cmd = queue.command_buffer()
threading.Thread(target=lambda: encode_work_a(cmd)).start()
threading.Thread(target=lambda: encode_work_b(cmd)).start()

# CORRECT: Separate command buffers
def worker(work_fn):
    cmd = queue.command_buffer()
    work_fn(cmd)
    cmd.commit()
```

## Performance Considerations

1. **Minimize synchronization points** - Batch work into fewer command buffers
2. **Use double/triple buffering** - Keep GPU busy while CPU prepares next frame
3. **Prefer fences over wait_until_completed** - More fine-grained synchronization
4. **Share immutable objects** - Pipeline states, samplers can be shared freely

## Summary

- **Thread-safe**: Device, CommandQueue, Library, Pipeline states
- **Not thread-safe**: CommandBuffer, Encoders
- **Partial**: Buffers/Textures (object safe, contents need synchronization)
- **GIL released**: During wait operations and shader compilation
