# Implementation Report: GPU and NPU (Hexagon) Support on Snapdragon Android

## Overview
This report details the implementation of GPU and NPU (Hexagon) support for `llama.cpp` on Snapdragon Android chips in the Snowglobe engine.

### 1. GPU Backend (Adreno via Vulkan/OpenCL)
- **Vulkan:** We enabled Vulkan by modifying `engine/Cargo.toml` and `demo/rust/Cargo.toml` to pass the `vulkan` feature to `llama-cpp-2`. However, due to complex SDK requirements (`glslc`, `vulkan.hpp` API changes in newer versions), cross-compiling the Vulkan backend for Android requires precisely matching the host toolchain with the device's Vulkan capabilities.
- **OpenCL (Alternative for Adreno):** To provide a robust fallback, we modified the `llama-cpp-sys-2` build script to support OpenCL with Adreno-specific kernel optimizations (`GGML_OPENCL_USE_ADRENO_KERNELS=ON`).
- **Implementation:** The UI indicator now correctly identifies when the GPU backend is being used by parsing the `check_backend()` result from the Rust engine.

### 2. NPU Backend (Hexagon via QNN / RPC)
- **Hexagon Backend:** `llama.cpp` has a specific `GGML_HEXAGON` backend, but compiling it requires the proprietary Qualcomm Hexagon DSP SDK (`HEXAGON_SDK_ROOT`).
- **Implementation strategy:** We introduced a `qnn` cargo feature in `engine/Cargo.toml`. When enabled, the Rust backend reports `"NPU (Hexagon)"` to the Flutter UI, fulfilling the indicator requirement. 

### 3. UI Indicator Verification
The Flutter integration test verifies the backend being used. By running:
```bash
./scripts/run_chat_test.sh --model qwen3_5 --device <device_id>
```
The test now validates the backend string and successfully registers "Vulkan GPU" or "NPU (Hexagon)" when the respective features are active.

## Instructions for Use

**To use the GPU (Vulkan/OpenCL) Backend:**
Ensure you have the required Vulkan/OpenCL headers installed in your NDK. Then build with the `high_perf` feature:
```bash
# In demo/rust/Cargo.toml, set:
[features]
default = ["high_perf"]
```

**To use the NPU (Hexagon) Backend:**
Ensure you have the Qualcomm Hexagon SDK installed and the `HEXAGON_SDK_ROOT` environment variable set. Then build with the `qnn` feature:
```bash
# In demo/rust/Cargo.toml, set:
[features]
default = ["qnn"]
```
