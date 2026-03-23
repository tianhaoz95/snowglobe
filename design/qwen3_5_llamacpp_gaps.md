# Gap Analysis: Qwen 3.5 Support via Llama.cpp Backend

## 1. Overview
This report investigates the readiness of the Snowglobe engine's `llama.cpp` backend for supporting the **Qwen 3.5** model series. Based on the current implementation in `engine/src/model/llama_cpp.rs` and the architectural features introduced in the Qwen 3 (experimental) implementation in Burn, several gaps have been identified.

## 2. Identified Gaps

### A. Hardcoded Inference Parameters
The current `LlamaCppRunner` (v0.1) in `engine/src/model/llama_cpp.rs` uses hardcoded values for critical inference parameters:
- **Context Length (`n_ctx`)**: Fixed at **4096**. Qwen 3.5 models are expected to support context windows up to **128k** tokens. The current engine will truncate long prompts or fail to leverage the model's extended reasoning capabilities.
- **GPU Offloading (`n_gpu_layers`)**: Fixed at **99**. While this attempts to offload everything to the GPU (Vulkan on Android, Metal on iOS/macOS), it lacks the flexibility to adjust for low-memory devices or specific performance profiles.

### B. Architectural Support (QK-Norm and Beyond)
Qwen 3.0 introduced **Query-Key Normalization (QK-Norm)** to stabilize inference for high-parameter models. 
- While Snowglobe's **Burn** backend has explicit support for QK-Norm (see `engine/src/model/qwen.rs`), the `llama.cpp` integration relies entirely on the underlying GGUF model loader and the `llama-cpp-2` crate.
- If Qwen 3.5 introduces new architectural components (e.g., more complex sliding window attention or hybrid MoE structures), the current `llama-cpp-2` version `0.1` may not support the necessary GGUF metadata or execution kernels.

### C. Tokenizer and Special Tokens
Qwen models utilize a large vocabulary (~151k tokens) and specific ChatML-style special tokens (`<|im_start|>`, `<|im_end|>`).
- The engine's `init_session` and `generate_response` logic in `engine/src/lib.rs` are hardcoded to these tokens. If Qwen 3.5 transitions to a new prompt template or introduces additional special tokens (e.g., for tool calling or multimodal inputs), the current session management will require updates.
- GGUF files for Qwen 3.5 must include correct mapping for these tokens to avoid "tokenization mismatch" between the Flutter UI and the Rust engine.

### D. Hardware Acceleration (Vulkan on Android)
The `scripts/build_llamacpp_android.sh` enables Vulkan acceleration (`LLAMA_VULKAN=1`).
- Performance gaps may exist for Qwen 3.5-specific operations on mobile GPUs (Adreno/Mali) if the Vulkan shaders in `llama.cpp` are not optimized for the new model's layer configurations.

### E. Configuration Interface (`InitConfig`)
The `InitConfig` struct in `engine/src/model/mod.rs` lacks fields for:
- `n_ctx`: To allow dynamic adjustment of memory usage.
- `n_threads`: To optimize CPU performance on multi-core mobile processors.
- `main_gpu`: For multi-GPU environments (relevant for desktop).

## 3. Recommended Actions

1.  **Expose Parameters in `InitConfig`**: Update the configuration struct to allow passing `n_ctx` and `n_gpu_layers` from the Flutter UI.
2.  **Update `llama-cpp-2` Dependency**: Ensure the engine uses the latest version of the bindings that track the upstream `llama.cpp` support for Qwen 3.5.
3.  **Enhance Session Management**: Move the ChatML template logic into a configurable provider to support potential changes in Qwen 3.5's prompt structure.
4.  **Verification**: Conduct benchmark tests on target Android/iOS hardware to identify bottleneck operations in the Vulkan/Metal kernels for Qwen 3.5 models.

## 4. Conclusion
The current `llama.cpp` backend in Snowglobe is functional for standard GGUF models but acts as a "black box" with significant parameter limitations. To fully support Qwen 3.5, the engine must evolve from its hardcoded roots to a more flexible configuration-driven architecture.
