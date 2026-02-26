# Report: Integration Testing and Android/Qualcomm NPU Deployment

This report summarizes the implementation of a comprehensive integration testing suite and the infrastructure required to deploy and benchmark Snowglobe on physical Android devices, specifically targeting Qualcomm NPU acceleration.

## 1. Automated Integration Testing
A new integration test suite was developed at `demo/integration_test/chat_test.dart`.

- **Workflow:** The test automates the full application lifecycle: initializing the Rust engine, waiting for model loading, sending a default prompt ("what is the capital of China?"), and verifying the response ("beijing").
- **Real-time Monitoring:** The test logs token arrival in real-time to the console, allowing for visual verification of streaming performance.
- **Performance Metrics:** The suite scrapes and reports two critical metrics from the UI:
    - **Prefill Time:** Time-to-first-token (TTFT).
    - **Generation Speed:** Tokens per second (tok/s), calculated specifically for the generation phase (excluding prefill).

## 2. Android Infrastructure & Build System
Significant work was performed to enable the Rust core and ExecuTorch runtime to function correctly on Android.

- **Multi-ABI Support:** Updated `scripts/build_executorch_android.sh` to build ExecuTorch static libraries for all four standard Android architectures (`arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`).
- **Build System Robustness:** Enhanced `engine/build.rs` to:
    - Automatically resolve ExecuTorch headers from `third_party/executorch`.
    - Handle the flattened library structure required for Android deployment.
    - Implement `+whole-archive` linking for critical libraries (`cpuinfo`, `XNNPACK`, `pthreadpool`), resolving "symbol not found" errors (e.g., `cpuinfo_isa`) that previously crashed the app on Android.
- **Dynamic Library Fixes:** Manually bundled `libc++_shared.so` and updated the Gradle configuration to ensure stable dynamic library loading on physical devices.

## 3. Qualcomm NPU (QNN) Integration
To leverage hardware acceleration on Snapdragon-based devices like the Oppo Find N3:

- **Model Conversion:** The `converter/convert_qwen3_to_pte.py` script was updated with a `qualcomm` backend. It now supports `float16` precision and partitioning for the Qualcomm AI Engine (HTP).
- **Backend Support:** The Rust engine now includes logic to link and initialize the Qualcomm ExecuTorch backend when the QNN SDK is provided during build.
- **Portable Fallback:** Added a `portable` conversion target to allow testing on any hardware using standard CPU kernels when specialized backends fail.

## 4. Testing Optimizations
- **Bypass Downloads:** Updated the Android initialization logic to check for models in the app's external cache (`/sdcard/Android/data/...`). This allows developers to `adb push` large `.pte` files directly to the device, bypassing slow cellular/Wi-Fi downloads during iteration.
- **Improved Stability:** Added `mounted` checks in the Flutter streaming loop to prevent `setState` errors when tests or user sessions conclude while the model is still generating.

## 5. Execution Summary (MacOS/Metal)
The system was successfully verified on MacOS with the following results:
- **Framework:** ExecuTorch (.pte)
- **Acceleration:** Apple MPS (GPU)
- **Prefill:** ~4.5s
- **Generation:** ~15 tok/s
- **Result:** Pass (Detected "beijing" in output)
