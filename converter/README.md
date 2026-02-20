# Qwen3 ExecuTorch Converter

This directory contains the script to convert Qwen3 models to the ExecuTorch `.pte` format for high-performance on-device inference.

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
   pip install executorch transformers safetensors huggingface_hub
   ```

2. **ExecuTorch C++ Runtime**:
   The Rust engine links against the ExecuTorch C++ static libraries. You must build them manually:
   ```bash
   cd third_party/executorch
   git submodule update --init --recursive
   mkdir -p cmake-out && cd cmake-out
   cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
      -DEXECUTORCH_BUILD_MPS=ON \
      -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      ..
   make -j$(sysctl -n hw.ncpu)
   ```

## Model Conversion

Run the conversion script to download weights from Hugging Face and export to `qwen3_0.6b.pte`:

```bash
python converter/convert_qwen3_to_pte.py
```

*Note: The script currently targets a fixed sequence length of 128 tokens and uses the MPS (Metal) partitioner by default.*

## Running PTE Inference Tests

To run the `test_one_plus_one_pte` in the engine, ensure you point the build process to your compiled ExecuTorch libraries:

```bash
# From the project root
export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$(pwd)/third_party/executorch/cmake-out
cd engine
cargo test tests::test_one_plus_one_pte --release -- --nocapture
```

### Linking Configuration
The engine uses a `build.rs` to link the static libraries. If you add new backends (like XNNPACK), update `engine/build.rs` to include the additional search paths and libraries. For macOS MPS support, the following frameworks are linked:
- Foundation
- Metal
- MetalPerformanceShaders
- MetalPerformanceShadersGraph

## Android Development

To enable ExecuTorch inference on Android devices, follow these steps to build the runtime and deploy the model.

### 1. Build ExecuTorch for Android
Use the provided helper script to cross-compile the ExecuTorch static libraries for `arm64-v8a` and `x86_64` (Simulator). This requires the **Android NDK** to be installed.

```bash
# Ensure ANDROID_NDK is set, or the script will try a default macOS path
export ANDROID_NDK=/path/to/your/ndk
./scripts/build_executorch_android.sh
```
This script populates `executorch-android/` with architecture-specific folders (`arm64-v8a/`, `x86_64/`) containing all necessary `.a` files.

### 2. Run the Flutter App
The engine's `build.rs` is configured to find the Android libraries using the `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` environment variable.

```bash
export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$(pwd)/executorch-android
cd demo
flutter run
```

### 3. Deploy the PTE Model to Android
Since the model is large and not bundled by default, you must push it to the device's internal storage. 

1.  **Find the path**: When the app starts, look for the log line:
    `Application Support Directory: /data/user/0/com.example.snowglobedemo/files`
2.  **Push the model**:
    ```bash
    # Push to a temporary location first
    adb push qwen3_0.6b.pte /data/local/tmp/
    # Move to the app's internal directory (requires the app to be debuggable)
    adb shell "run-as com.example.snowglobedemo cp /data/local/tmp/qwen3_0.6b.pte /data/user/0/com.example.snowglobedemo/files/"
    ```

### 4. Test Inference
In the "SNOWGLOBE" app:
- Type a prompt starting with the prefix `pte:`.
- Example: `pte:Explain quantum physics in one sentence.`
- The app will load `qwen3_0.6b.pte` from its files directory and run inference using the ExecuTorch XNNPACK backend.
