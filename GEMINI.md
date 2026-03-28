# Snowglobe

Snowglobe is a high-performance, cross-platform LLM (Large Language Model) inference engine built with Rust and Flutter. It leverages the **Burn** deep learning framework to provide efficient inference on both CPU and GPU across mobile (iOS, Android) and desktop platforms.

## Core Architecture

The project follows a hybrid architecture combining a high-performance Rust core with a modern Flutter UI:

- **Rust Engine (`engine/`)**: The heart of the project. It implements the LLM inference logic (specifically supporting Qwen 2.5 and Qwen 3 models) using the Burn framework. It features dynamic backend switching between CPU (NdArray) and GPU (WGPU/Vulkan/Metal).
- **Flutter Integration**: Uses `flutter_rust_bridge` to provide seamless, type-safe communication between the Flutter UI and the Rust engine.

## Key Project Structure

- `engine/`: The core Rust library containing the model definitions, tensor operations, and inference logic.
- `demo/`: A Flutter demonstration and benchmark application showcasing the engine's capabilities, including model downloading and streaming response generation.
- `app/`: The primary Flutter application workspace.
- `packages/snowglobe_openai`: A Flutter package exposing an OpenAI-compatible API for the Snowglobe engine.

## Development Workflow

### Git Branching
At the start of a task, if the local repository is on the `main` branch, always check out to a new branch with a name reflecting the task that will be implemented.
```bash
git checkout -b <task-name>
```

## Development & Testing

When making changes to the Rust **engine**, verify the implementation using the following commands:

```bash
# In the engine/ directory

# 1. Verify Qwen 3 (Burn backend)
cargo test tests::test_one_plus_one_qwen3 --features high_perf -- --nocapture

# 2. Verify ExecuTorch implementation (MPS on Mac)
EXECUTORCH_USE_MPS=1 \
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
cargo test tests::test_one_plus_one_pte --release -- --nocapture
```

### ExecuTorch (Experimental)
To run inference using ExecuTorch `.pte` models:

1. **Export the model**:
   ```bash
   # For MPS (Fastest on Mac)
   python converter/convert_qwen3_to_pte.py --backend mps
   # For XNNPACK (Standard CPU)
   python converter/convert_qwen3_to_pte.py --backend xnnpack
   ```

2. **Run the test**:
   ```bash
   cd engine
   
   # For MPS (Mac)
   EXECUTORCH_USE_MPS=1 \
   EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
   cargo test tests::test_one_plus_one_pte --release -- --nocapture
   
   # For XNNPACK (CPU)
   unset EXECUTORCH_USE_MPS
   EXECUTORCH_RS_EXECUTORCH_LIB_DIR=../third_party/executorch/cmake-out \
   cargo test tests::test_one_plus_one_pte --release -- --nocapture
   ```

### Integration Testing (Flutter Demo)
To verify the full integration between the Flutter UI and the Rust core, including model downloading and streaming inference, run the chat integration test:

```bash
cd demo
# Run on MacOS (with Metal Acceleration for ExecuTorch)
EXECUTORCH_USE_MPS=1 flutter test integration_test/chat_test.dart -d macos

# Run on Android
flutter test integration_test/chat_test.dart -d <device_id>
```

### Dependency Management (Local vs. Remote)
For testing purposes, you can switch the `demo` app's `snowglobe_openai` dependency between the local source code and the version published on `pub.dev`.

**To test the published package (e.g., to verify prebuilt binaries):**
```bash
cd demo

# 1. Switch to the version from pub.dev
./scripts/toggle_snowglobe_openai.sh pub

# 2. Run or build the app (Android users will not need NDK for prebuilt architectures)
flutter build apk --release
```

**To switch back to local development:**
```bash
# Switch back to the local source code
./scripts/toggle_snowglobe_openai.sh local
```

*Note: This uses a `pubspec_overrides.yaml` file (automatically added to `.gitignore`) to override the local path defined in `pubspec.yaml` without modifying the core configuration.*

### Optimized Integration Testing (Model Caching)
To avoid downloading large model files (~500MB) every time the integration test is run, use the optimized test runner. This script caches the model assets in `.test_assets/` and pushes them to the device before running the test.

```bash
# Run the optimized integration test (Android)
./scripts/run_chat_test.sh --model qwen3_5 --device <device_id>
```

The runner performs the following:
1.  **Downloads once:** Fetches model weights and metadata to `.test_assets/` on the host.
2.  **Side-loads to device:** Uses `adb push` to copy assets to `/data/local/tmp/snowglobe/qwen3_5/`.
3.  **Bypasses app downloads:** The Flutter app detects these files and copies them to its local cache instead of downloading them.

#### Interpreting Logs
The test provides detailed feedback during execution:
- **CHAT TEST - RUNTIME INFO**: Identifies the active hardware backend (e.g., CPU/GPU) and the orchestration framework (Burn vs. ExecuTorch).
- **PERFORMANCE METRICS**:
    - **Prefill**: The "Time to First Token" (TTFT) in seconds.
    - **Generation Speed**: The sustained inference speed in tokens per second (tok/s).
- **Received tokens**: Shows the real-time stream of the model's response to the default prompt ("what is the capital of China?").

## Deployment

### Firebase App Distribution (Android)
To deploy the Android demo app to Firebase App Distribution, run the following commands from the `demo/` directory.

**Prerequisites & Best Practices:**
- **Versioning:** Increase the version name and build number in `pubspec.yaml` (e.g., `1.1.2+5` becomes `1.1.3+6`) before building.
- **Environment:** Ensure `ANDROID_NDK_ROOT` is set to your NDK path (e.g., `export ANDROID_NDK_ROOT="/path/to/ndk/28.2.13676358/"`) to avoid build failures in Rust crates like `llama-cpp-sys-2`.
- **Target Platforms:** Use `--target-platform android-arm64,android-x64` to avoid issues with unsupported architectures (like `armv7`) and speed up the build.

```bash
cd demo

# 1. Build the release APK
export ANDROID_NDK_ROOT="/path/to/your/ndk/"
flutter build apk --release --target-platform android-arm64,android-x64

# 2. Distribute to Firebase
# The App ID can be found in demo/lib/firebase_options.dart (under FirebaseOptions android)
firebase appdistribution:distribute build/app/outputs/flutter-apk/app-release.apk \\
  --app 1:946016760428:android:ba36c7d7f3b50497a71e49 \\
  --release-notes \"Detailed description of the current feature or improvement\" \\
  --groups \"dev\"
```

### Publishing `snowglobe_openai` to pub.dev

The `snowglobe_openai` package is a self-contained Flutter FFI plugin that wraps the Rust inference engine. To ensure a "zero-config" experience for users (no NDK required), follow these steps to prebuild binaries before publishing.

**1. Prebuild Android Binaries:**
Use the `demo` app's build system to generate the optimized Rust shared libraries.
```bash
cd demo
export ANDROID_NDK_ROOT="/path/to/your/ndk/"

# Build for both arm64 and x64
flutter build apk --release --target-platform android-arm64,android-x64
```

**2. Package Binaries into the Plugin:**
Copy the compiled `.so` files from the build artifacts into the package's `jniLibs` directory.
```bash
# Create directories
mkdir -p ../packages/snowglobe_openai/android/src/main/jniLibs/arm64-v8a
mkdir -p ../packages/snowglobe_openai/android/src/main/jniLibs/x86_64

# Copy binaries
cp build/snowglobe_openai/build/aarch64-linux-android/release/librust_lib_snowglobe_openai.so \
   ../packages/snowglobe_openai/android/src/main/jniLibs/arm64-v8a/
cp build/snowglobe_openai/build/x86_64-linux-android/release/librust_lib_snowglobe_openai.so \
   ../packages/snowglobe_openai/android/src/main/jniLibs/x86_64/
```

**3. Validate & Publish:**
- Boost the version in `packages/snowglobe_openai/pubspec.yaml`.
- Update `CHANGELOG.md` to note the included prebuilt binaries.
- Run the publication:
  ```bash
  cd ../packages/snowglobe_openai
  flutter_rust_bridge_codegen generate
  flutter pub publish --dry-run
  flutter pub publish
  ```

*Note: The package uses hybrid logic. If prebuilt binaries are present in `jniLibs`, the user's build will skip the Rust compilation. If they are missing, it will automatically fall back to a source build (requiring Rust/NDK).*

### Targeting Different Backends

The Snowglobe engine supports multiple backends. You can target them using build-time features or runtime configuration.

#### 1. Hardware Acceleration (GPU vs. CPU vs. NPU)
Hardware acceleration is controlled via Rust features in `demo/rust/Cargo.toml`.
- **CPU (Default):** The `default` feature list is empty. Uses Burn's `NdArray` backend and `llama.cpp` CPU backend.
- **GPU (Vulkan/OpenCL/Metal):** Enable the `high_perf` feature by editing `demo/rust/Cargo.toml`:
  ```toml
  [features]
  default = ["high_perf"]
  ```
  *Note for Android (Adreno GPUs): You must have the Vulkan or OpenCL headers installed in your NDK to compile the GPU backend.*
- **NPU (Hexagon):** Enable the `qnn` feature by editing `demo/rust/Cargo.toml`:
  ```toml
  [features]
  default = ["qnn"]
  ```
  *Note for Android (Snapdragon NPUs): You must have the Qualcomm Hexagon SDK installed and the `HEXAGON_SDK_ROOT` environment variable configured before building.*

#### 2. Inference Orchestration (llama.cpp, ExecuTorch, Burn)
The orchestration layer is selected at runtime in the Flutter app based on `--dart-define` flags and model file availability.

- **llama.cpp (Default):** Optimized for GGUF models.
  ```bash
  flutter build apk --release --dart-define=USE_LLAMACPP=true
  ```
- **ExecuTorch (Experimental):** Optimized for `.pte` models.
  ```bash
  flutter build apk --release --dart-define=USE_LLAMACPP=false --dart-define=USE_EXECUTORCH=true
  ```
- **Burn (Safetensors):** Native Burn implementation for `.safetensors` models.
  ```bash
  flutter build apk --release --dart-define=USE_LLAMACPP=false --dart-define=USE_EXECUTORCH=false
  ```

**Note:** The app also performs automatic model detection. If a `model.pte` file is found in the application's cache directory, it will attempt to use the ExecuTorch backend regardless of the build flags.

## Tech Stack

- **Frontend**: Flutter (Dart)
- **Engine Core**: Rust
- **Deep Learning Framework**: [Burn](https://burn.dev/)
- **Bridge**: [flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge)
- **Model Support**: Qwen 2.5 & Qwen 3 (Safetensors)
