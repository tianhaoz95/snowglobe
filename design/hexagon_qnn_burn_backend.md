# Design: Hexagon QNN Backend for Burn

## 1. Overview
This document outlines the strategy for implementing a Hexagon QNN (Qualcomm Neural Network) backend for the Burn deep learning framework. This backend will enable hardware-accelerated inference on Qualcomm Snapdragon NPUs, specifically targeting the Hexagon DSP.

The implementation will leverage the QNN kernels available in `ggml` (from `llama.cpp`) as a reference for efficient tensor operations on the Hexagon architecture.

## 2. Reference: GGML Hexagon Implementation
The `ggml-hexagon.cpp` implementation provides several key insights:
- **RPC Mechanism**: Communication with the Hexagon DSP often happens via a remote procedure call (RPC) mechanism (e.g., FastRPC).
- **Memory Management**: Uses specialized ION or DMA-BUF memory for zero-copy sharing between the CPU and NPU.
- **Kernel Specialization**: Provides optimized kernels for common LLM operations like `MatMul`, `RMSNorm`, and `RoPE`.
- **Power Management**: Explicitly manages Hexagon power levels (HMX) to balance performance and battery life on mobile devices.

## 3. Architecture for Burn
The Burn backend will be implemented by satisfying the `burn::tensor::backend::Backend` trait.

### 3.1. Tensor Representation
Tensors on Hexagon will be represented as handles to buffers allocated in the Hexagon-accessible memory space.
```rust
struct HexagonTensor {
    handle: u64, // Handle to QNN/Hexagon buffer
    shape: Shape,
    dtype: QnnDtype,
}
```

### 3.2. Integration Strategy
1. **FFI Layer**: Create a thin Rust wrapper over the QNN SDK and the reference kernels from GGML.
2. **Buffer Management**: Implement a custom allocator for Burn that uses `QnnMem_alloc` or similar Qualcomm-specific APIs.
3. **Graph Execution**: 
    - **Option A (Op-by-Op)**: Execute each Burn operator as a standalone QNN call. High overhead but easier to implement.
    - **Option B (Graph Capture)**: Record a sequence of Burn operations and compile them into a QNN graph for execution. Significant performance gains for LLMs.

### 3.3. Key Kernels to Port
- **Linear/MatMul**: Utilizing Hexagon Vector eXtensions (HVX) for high-throughput 8-bit or 16-bit quantization.
- **RoPE (Rotary Positional Embeddings)**: Implementing the rotation logic directly in Hexagon kernels to avoid CPU-GPU-NPU synchronization.
- **Softmax**: Optimized for the parallel processing capabilities of the Hexagon DSP.

## 4. Challenges & Mitigations
- **Quantization**: QNN performs best with `INT8` quantization. We must ensure Burn's quantization scheme aligns with QNN's requirements.
- **SDK Dependencies**: Requires the Qualcomm QNN SDK and Hexagon SDK to be present during the build and at runtime on the device.
- **Synchronization**: Minimizing the latency of moving data between the CPU (Burn orchestration) and the NPU (Hexagon execution).
