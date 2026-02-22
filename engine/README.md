# Snowglobe Engine

Snowglobe is a high-performance LLM inference engine built with Rust, leveraging the **Burn** deep learning framework and **ExecuTorch** for cross-platform efficiency.

## Architecture Overview

The engine is organized into several key modules:

- **`src/lib.rs`**: The main entry point. Handles global state (tokenizer, model instance), session management, and the high-level `init` and `generate_response` APIs.
- **`src/model/`**: Contains model architectures and traits.
    - `mod.rs`: Defines the `Model` trait, providing a unified interface for different backends.
    - `qwen.rs`: Implementation of the Qwen architecture using the **Burn** framework.
    - `qwen_pte.rs`: Implementation for Qwen inference using **ExecuTorch** (`.pte` models).
- **`src/layer/`**: Custom neural network layers.
    - `large_vocab.rs`: Implements vocabulary sharding (`LargeVocabEmbedding`/`LargeVocabLinear`) to handle Qwen's 151k vocabulary on mobile GPUs with memory binding limits.
- **`src/utils/`**: General utilities.
    - `downloader.rs`: Automated downloading of models, tokenizers, and configs from Hugging Face.
- **`src/adapter/`**: C++/Rust FFI bridge for ExecuTorch integration.
- **`src/weight.rs`**: Logic for deserializing and loading Safetensors into model records.
- **`src/rope.rs`**: Rotary Positional Embedding (RoPE) implementation.

## Testing Commands

The engine supports various testing configurations to verify performance across different backends and models.

### CPU Inference (NdArray)
Verify basic logic and correctness on the CPU using the `NdArray` backend.
```bash
# Standard CPU test
cargo test tests::test_one_plus_one -- --nocapture

# Ensure clean CPU-only run (no GPU dependencies)
cargo test tests::test_one_plus_one --no-default-features -- --nocapture
```

### GPU Inference (WGPU)
Test high-performance inference using the `WGPU` backend (Metal on macOS/iOS, Vulkan on Android/Linux).
```bash
# Standard GPU test
cargo test tests::test_one_plus_one --features high_perf -- --nocapture

# Test vocabulary sharding logic on GPU
cargo test tests::test_sharded_one_plus_one --features high_perf -- --nocapture
```

### Model Specific Tests
Verify the engine against specific Qwen versions.
```bash
# Test Qwen 2.5 (0.5B Instruct)
cargo test tests::test_one_plus_one_qwen2 --features high_perf -- --nocapture

# Test Qwen 3 (0.6B)
cargo test tests::test_one_plus_one_qwen3 --features high_perf -- --nocapture
```

### ExecuTorch (.pte) Inference
Verify the experimental ExecuTorch path. This requires the ExecuTorch static libraries and a pre-exported `.pte` model.
```bash
# Note: --release is recommended for ExecuTorch tests to ensure ABI compatibility 
# and reasonable performance.
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
cargo test tests::test_one_plus_one_pte --release -- --nocapture

# For macOS (MPS Backend)
EXECUTORCH_USE_MPS=1 \
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
cargo test tests::test_one_plus_one_pte --release -- --nocapture
```
