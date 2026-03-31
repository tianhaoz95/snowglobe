# Snowglobe Engine: Llama.cpp Integration Report

## 1. Overview
Snowglobe currently supports two inference backends: **Burn** (native Rust deep learning framework) and **ExecuTorch** (experimental integration for specialized hardware). Adding **llama.cpp** as a third backend will provide high-performance support for GGUF models, leveraging industry-standard optimizations for CPU (AVX, NEON, AMX) and GPU (Metal, CUDA, Vulkan).

## 2. Current Architecture Analysis
The engine uses a `QwenVariant` enum in `engine/src/model/mod.rs` to switch between backends. Each backend implements a `Model` trait, which is heavily coupled with `Burn`'s `Backend`, `Tensor`, and `KVCache` types.

- **Burn Backend**: Implements the transformer logic manually in Rust. It requires explicit KV cache management in each forward pass.
- **ExecuTorch Backend**: Uses a C++ adapter (`executorch_adapter.cpp`) and wraps it in a Burn-compatible `Model` implementation. It converts Burn tensors to C++ vectors and back, incurring overhead.

## 3. Proposed Model Runner Interface Design
To accommodate backends like `llama.cpp` and `ExecuTorch` more efficiently, we should decouple the inference logic from the `Burn` framework at the top level.

**New `ModelRunner` Trait:**
```rust
pub trait ModelRunner: Send + Sync {
    /// Load the model from the specified directory/file.
    fn load(path: &Path, config: &EngineConfig) -> Result<Box<Self>, String> 
    where Self: Sized;

    /// Perform a forward pass, returning logits for the last token.
    /// Manages state internally or via the provided session.
    fn forward(
        &self, 
        tokens: &[u32], 
        session: &mut EngineSession
    ) -> Result<Vec<f32>, String>;
}
```

**Unified Engine Variant:**
```rust
pub enum EngineVariant {
    Burn(Box<dyn ModelRunner>),
    ExecuTorch(Box<dyn ModelRunner>),
    LlamaCpp(Box<dyn ModelRunner>),
}
```

## 4. Implementing Llama.cpp Support

### A. Build System Integration
Add the `llama-cpp-2` crate to `engine/Cargo.toml`. This crate provides high-level, safe Rust bindings to `llama.h`.
```toml
[dependencies]
llama-cpp-2 = "0.1.0" # Example version
```
Update `build.rs` to ensure hardware acceleration flags (like `LLAMA_METAL=ON` on macOS or `LLAMA_VULKAN=ON` on Android) are passed to the underlying CMake build of `llama.cpp`.

### B. Adapter Layer
Create `engine/src/model/llama_cpp.rs`. This will wrap the `llama_cpp_2::model::LlamaModel` and `LlamaContext`.

- **State Management**: `llama.cpp` uses a `LlamaContext` to manage KV cache. The `EngineSession` should hold a mapping between a `session_id` and a `LlamaContext` to allow multi-turn conversations.
- **Prompt Processing**: Use `llama.cpp`'s internal batch processing (`LlamaBatch`) for prefilling the prompt.

### C. Initialization Workflow
1. **Model Loading**: Initialize `LlamaModel` from a `.gguf` file.
2. **Context Creation**: Create a `LlamaContext` with backend-specific parameters (e.g., number of GPU layers to offload).
3. **Inference**: In the `forward` pass, add the new tokens to the batch, call `decode()`, and retrieve the logits for the last index.

## 5. Implementation Action Plan

1. **Introduce the `ModelRunner` Trait**: Refactor existing `Burn` and `ExecuTorch` logic to implement this trait.
2. **Add `llama-cpp-2` Dependency**: Integrate the crate and verify the build on target platforms (Android/iOS).
3. **Implement `LlamaCppEngine`**: 
    - Map `GGUF` model parameters to the engine's internal configuration.
    - Implement the `forward` pass using `llama_cpp_2`'s batch API.
4. **Update `InitConfig`**: Add `LlamaCpp` as a valid backend option in the configuration passed from Flutter.
5. **Benchmarking**: Compare tokens-per-second (TPS) and memory usage across all three backends to ensure `llama.cpp` provides the expected performance gains.
