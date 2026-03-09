# Llama.cpp Integration: Implementation & Testing Tasks

This document breaks down the integration of `llama.cpp` into the Snowglobe engine into actionable tasks.

## Phase 1: Core Abstraction (ModelRunner)
**Goal:** Decouple inference from Burn-specific types to allow diverse backends.

- [ ] **Task 1.1: Define `ModelRunner` Trait**
    - Create `engine/src/model/runner.rs`.
    - Implement `ModelRunner` trait with `load` and `forward` (using raw slices/vectors instead of `burn::Tensor`).
- [ ] **Task 1.2: Refactor Burn Backend**
    - Implement `ModelRunner` for the existing `Qwen<B>` model.
    - Wrap the existing `forward` logic to handle the transition from `&[u32]` to `Tensor<B, 2, Int>`.
- [ ] **Task 1.3: Refactor ExecuTorch Backend**
    - Implement `ModelRunner` for `QwenPte`.
    - Simplify `QwenPte::forward` by removing redundant Burn tensor conversions where possible.

## Phase 2: Llama.cpp Integration
**Goal:** Implement the GGUF model runner using `llama-cpp-2`.

- [ ] **Task 2.1: Dependency & Build Setup**
    - Add `llama-cpp-2` to `engine/Cargo.toml`.
    - Update `engine/build.rs` to handle `llama.cpp` compilation flags for Android (Vulkan/OpenCL) and iOS (Metal).
- [ ] **Task 2.2: Implement `LlamaCppRunner`**
    - Create `engine/src/model/llama_cpp.rs`.
    - Implement `load` to initialize `LlamaModel`.
    - Implement `forward` using `LlamaBatch` for efficient token processing.
- [ ] **Task 2.3: Session & State Management**
    - Update `SessionState` in `engine/src/lib.rs` to store backend-specific context (e.g., `LlamaContext`).
    - Ensure KV cache is correctly mapped between `ModelRunner` calls.

## Phase 3: Integration & API Updates
**Goal:** Expose the new backend to the Flutter UI.

- [ ] **Task 3.1: Update `InitConfig`**
    - Add `backend: BackendType` enum to `InitConfig` (Values: `Burn`, `ExecuTorch`, `LlamaCpp`).
- [ ] **Task 3.2: Update `init_model` in `lib.rs`**
    - Add a match arm for `BackendType::LlamaCpp`.
    - Implement GGUF file detection logic (looking for `model.gguf` in the cache directory).
- [ ] **Task 3.3: GGUF Downloader**
    - Add `download_qwen_gguf` to `engine/src/utils/downloader.rs`.

## Phase 4: Testing & Validation
**Goal:** Ensure correctness and performance parity.

- [ ] **Task 4.1: Unit Testing (Rust Engine)**
    - [ ] **Correctness Test:** Create `tests::test_one_plus_one_gguf` in `lib.rs` (mirroring `test_one_plus_one_pte`).
    - [ ] **Multi-turn Test:** Create `tests::test_multi_turn_gguf` to verify KV cache persistence in `llama.cpp`.
- [ ] **Task 4.2: Integration Testing (Flutter Demo)**
    - [ ] **Backend Switch Test:** Update `demo/integration_test/chat_test.dart` to verify the engine starts correctly with the `LlamaCpp` backend selected.
    - [ ] **Performance Benchmarking:** Run the integration test and record `TTFT` (Prefill) and `tok/s` (Generation) for comparison in `report/performance_investigation.md`.
- [ ] **Task 4.3: Cross-Platform Verification**
    - [ ] **Android:** Build and run on a physical device using `scripts/build_executorch_android.sh` as a template for a new `scripts/build_llamacpp_android.sh`.
    - [ ] **iOS/MacOS:** Verify Metal acceleration is active in `llama.cpp` logs during inference.
