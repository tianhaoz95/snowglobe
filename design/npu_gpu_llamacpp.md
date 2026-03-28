# Llama.cpp GPU and NPU (Hexagon) Support on Android

## Research

### 1. GPU Support (Adreno)
- **Mechanism:** `llama.cpp` natively supports Adreno GPUs on Android via the **Vulkan** backend.
- **Current State:** The project currently attempts to export `LLAMA_VULKAN=1` in `build_llamacpp_android.sh`, but the Rust bindings (`llama-cpp-sys-2`) configure CMake based on Cargo features, not this environment variable.
- **Solution:** Add a `vulkan` feature to the `engine` crate that propagates to `llama-cpp-2/vulkan`. Then, conditionally enable this feature in the Flutter build for Android.

### 2. NPU Support (Hexagon / QNN)
- **Mechanism:** Snapdragon NPUs are supported in `llama.cpp` either via the **QNN (Qualcomm Neural Network)** backend or the **RPC (Hexagon DSP)** backend. QNN is the officially upstreamed and recommended path.
- **Current State:** The `llama-cpp-sys-2` crate (used for Rust bindings) does not expose a Cargo feature for QNN or RPC. 
- **Solution:**
  1. We must either use a local patch of `llama-cpp-sys-2` using `[patch.crates-io]` in `Cargo.toml` to enable the `GGML_QNN=ON` CMake flag, OR use `cmake` environment injection if supported.
  2. To avoid requiring the proprietary QNN SDK on every developer's machine during the regular build, we can fetch the pre-built QNN SDK libraries or rely on the device's system libraries if possible.
  3. Actually, building QNN backend requires the QNN SDK headers at compile time (`QNN_SDK_ROOT`).

## Tasks

1. **Fix GPU (Vulkan) Build:**
   - Update `engine/Cargo.toml` to add a `vulkan` feature that turns on `llama-cpp-2/vulkan`.
   - Update `demo/rust/Cargo.toml` to expose this feature.
   - Update `scripts/build_llamacpp_android.sh` to pass the `--features vulkan` flag to the flutter/rust build via `flutter build apk --dart-define=USE_VULKAN=true` or modify the `cargokit` setup to pass it.

2. **Implement NPU Build (Patch `llama-cpp-sys-2`):**
   - Create a local patch for `llama-cpp-sys-2` (e.g. in `./third_party/llama-cpp-sys-2-patch`) to add a `qnn` feature that injects `GGML_QNN=ON` and configures the SDK path.
   - Update `engine/Cargo.toml` to use this patched version.

3. **Update Demo App Indicator:**
   - Modify the UI in `demo/` to correctly indicate whether GPU (Vulkan), CPU, or NPU (QNN) is being used.
   - The indicator can parse the `Runtime Info` from the engine, so the Rust backend needs to return "llama.cpp (Vulkan)" or "llama.cpp (QNN)" in `init_engine` or `get_info`.

4. **Integration Testing:**
   - Run the integration test on the connected Lenovo tablet.
   - Verify the indicator outputs GPU/NPU and the model output passes.

5. **Final Reporting & Documentation:**
   - Summarize the implementation in `design/npu_gpu_implementation.md`.
   - Update `GEMINI.md` with instructions on how to use GPU and NPU.
