# Snowglobe Demo & Benchmark

A high-performance demonstration of Qwen LLM inference using Rust and Flutter.

## Build Requirements

### Common
- **Rust**: Nightly recommended.
- **Flutter**: Latest stable.
- **flutter_rust_bridge_codegen**: `cargo install flutter_rust_bridge_codegen`.

### Android Specific
- **Android NDK**: Version 28.x or later.
- **ExecuTorch Source**: Submodule in `third_party/executorch`.

---

## 🚀 Desktop (MacOS)

### 1. Build & Run (Metal Acceleration)
```bash
# Generate the bridge
flutter_rust_bridge_codegen generate

# Run with Metal (GPU) enabled for ExecuTorch
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
EXECUTORCH_USE_MPS=1 \
flutter run -d macos --release
```

---

## 📱 Android Deployment

### 1. Build ExecuTorch for Android
This step generates the static libraries for all Android ABIs (arm64, armeabi-v7a, x86_64, x86).

```bash
# From the project root
bash scripts/build_executorch_android.sh
```

### 2. Run on Android Device
Ensure your physical device or emulator is connected.

```bash
export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$(pwd)/../executorch-android

# Run the integration test
flutter test integration_test/chat_test.dart -d <device_id>

# Run the full application
flutter run -d <device_id> --release
```

### 3. Qualcomm NPU Acceleration (Optional)
To use the Snapdragon NPU (Qualcomm AI Engine):

1. **Build with QNN SDK**:
   ```bash
   export QNN_SDK_ROOT=/path/to/qnn/sdk
   bash scripts/build_executorch_android.sh
   ```
2. **Convert Model for HTP**:
   ```bash
   python3 converter/convert_qwen3_to_pte.py --backend qualcomm
   ```
3. **Deploy QNN Libraries**:
   Push the required `.so` files from the QNN SDK to `/sdcard/Android/data/com.example.snowglobedemo/cache/` on your device.

---

## 🛠 Model Management

To iterate faster, you can push models directly to the device cache instead of waiting for downloads:

```bash
# For Android
adb push qwen3_0.6b.pte /sdcard/Android/data/com.example.snowglobedemo/cache/model.pte
```

## 🧪 Integration Testing

The integration test measures **Prefill Time** and **Generation Speed**.

```bash
# MacOS
EXECUTORCH_USE_MPS=1 flutter test integration_test/chat_test.dart -d macos

# Android
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$(pwd)/../executorch-android \
flutter test integration_test/chat_test.dart -d <device_id>
```
