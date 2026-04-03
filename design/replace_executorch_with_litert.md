# Replacing ExecuTorch with LiteRT (Core)

This document outlines the plan to remove ExecuTorch support from Snowglobe and replace it with **LiteRT Core** (formerly TensorFlow Lite). Unlike the high-level LiteRT-LM, LiteRT Core provides the low-level access to tensors and stateful variables required for Snowglobe's advanced features like manual KV cache management and speculative decoding.

## 1. Rationale

- **Low-Level Control**: Direct access to the `Interpreter` and `TfLiteTensor` structures.
- **Stateful KV Cache**: Use TFLite **Variables** (stateful tensors) to persist KV cache across inference steps without copying buffers back to the host.
- **Speculative Decoding**: Ability to execute arbitrary sequences of `prefill` and `decode` signatures to support draft-and-verify loops.
- **Multi-Signature Support**: Leverage models exported via `litert-torch` with dedicated `prefill` (parallel) and `decode` (token-by-token) entry points.
- **Hardware Acceleration**: Access to XNNPACK (CPU), GPU (Metal/Vulkan), and NPU (NNAPI/QNN) via the Delegate API.

## 2. Implementation Plan

### Phase 1: Cleanup (ExecuTorch Removal)

1.  **Remove Files**:
    - `engine/src/model/qwen_pte.rs`
    - `engine/src/adapter/executorch_adapter.cpp`
    - `engine/src/adapter/executorch_adapter.h`
    - `executorch-android/` directory.
    - `scripts/build_executorch_android.sh`.
2.  **Update `engine/Cargo.toml`**:
    - Remove `has_executorch` or related build-time flags.
3.  **Update `engine/src/model/mod.rs`**:
    - Remove `EngineVariant::ExecuTorch`.
    - Remove `BackendType::ExecuTorch`.

### Phase 2: LiteRT Core Adapter (C++ & Rust)

1.  **New C++ Adapter**: Create `engine/src/adapter/litert_adapter.cpp` and `.h`.
    - Use `tensorflow::lite::Interpreter` and `SignatureRunner`.
    - Provide C-linkage functions to load models and invoke specific signatures.
2.  **Rust Bridge**: Update `engine/src/adapter/mod.rs` to interface with the LiteRT C API.
3.  **LiteRT Model Runner**: Create `engine/src/model/litert.rs`.
    - Implement `ModelRunner` trait.
    - Track `current_kv_len` and pass it to the model's `pos` input.

#### **Draft C API (`litert_adapter.h`):**
```cpp
extern "C" {
    void* litert_model_load(const char* model_path);
    void litert_model_destroy(void* model);
    
    // Invoke the 'prefill' signature
    int litert_model_prefill(
        void* model,
        const int32_t* input_ids,
        size_t length,
        float* output_logits
    );
    
    // Invoke the 'decode' signature
    int litert_model_decode(
        void* model,
        int32_t input_id,
        int32_t pos,
        float* output_logits
    );

    // Advanced: Access/Reset stateful variables (KV Cache)
    int litert_model_reset_state(void* model);
}
```

### Phase 3: Model Conversion & Support

1.  **Conversion Script**: Add `converter/convert_to_litert.py` using the `litert-torch` (formerly `ai-edge-torch`) library.
    - Export models with `kv_cache_max_len` and stateful variables.
2.  **State Management**: 
    - For simple generation, the state stays inside the model.
    - For speculative decoding, if a draft is rejected, we need to "rewind" the KV cache. This is done by passing the correct `pos` to the next `decode` call (LiteRT stateful models typically use `pos` to index into the internal variable).

### Phase 4: Flutter UI & Bridge

1.  **Update `api.rs`**:
    - Add `BackendType::LiteRT`.
2.  **Update `main.dart`**:
    - Add `InferenceBackend::liteRT`.
    - Look for `.tflite` files in `models/litert/`.

## 3. Build System & CI

1.  **LiteRT Library**: Link against `libtensorflowlite.so` (Android) or `TensorFlowLiteC.framework` (iOS).
2.  **Delegates**: Ensure the build includes XNNPACK and GPU delegates for performance.

## 4. Risks & Considerations

- **Statefulness**: Since the KV cache is inside the model, parallel requests (sessions) in the same process require separate `Interpreter` instances or careful state resetting.
- **Quantization**: LiteRT supports various quantization schemes (INT8, FP16). We should prioritize models optimized for the target hardware's NPU/GPU.

## 5. Timeline

- **Task 1: Cleanup**: 0.5 days.
- **Task 2: C++/Rust Adapter (Core)**: 3 days.
- **Task 3: Integration & Testing**: 2 days.
- **Task 4: Conversion Script**: 1 day.
- **Total**: ~1.5 weeks.
