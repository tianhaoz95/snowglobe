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

#### Interpreting Logs
The test provides detailed feedback during execution:
- **CHAT TEST - RUNTIME INFO**: Identifies the active hardware backend (e.g., CPU/GPU) and the orchestration framework (Burn vs. ExecuTorch).
- **PERFORMANCE METRICS**:
    - **Prefill**: The "Time to First Token" (TTFT) in seconds.
    - **Generation Speed**: The sustained inference speed in tokens per second (tok/s).
- **Received tokens**: Shows the real-time stream of the model's response to the default prompt ("what is the capital of China?").

## Tech Stack

- **Frontend**: Flutter (Dart)
- **Engine Core**: Rust
- **Deep Learning Framework**: [Burn](https://burn.dev/)
- **Bridge**: [flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge)
- **Model Support**: Qwen 2.5 & Qwen 3 (Safetensors)
