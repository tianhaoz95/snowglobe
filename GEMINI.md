# Snowglobe

Snowglobe is a high-performance, cross-platform LLM (Large Language Model) inference engine built with Rust and Flutter. It leverages the **Burn** deep learning framework to provide efficient inference on both CPU and GPU across mobile (iOS, Android) and desktop platforms.

## Core Architecture

The project follows a hybrid architecture combining a high-performance Rust core with a modern Flutter UI:

- **Rust Engine (`engine/`)**: The heart of the project. It implements the LLM inference logic (specifically supporting Qwen 2.5 models) using the Burn framework. It features dynamic backend switching between CPU (NdArray) and GPU (WGPU/Vulkan/Metal).
- **Flutter Integration**: Uses `flutter_rust_bridge` to provide seamless, type-safe communication between the Flutter UI and the Rust engine.

## Key Project Structure

- `engine/`: The core Rust library containing the model definitions, tensor operations, and inference logic.
- `demo/`: A Flutter demonstration and benchmark application showcasing the engine's capabilities, including model downloading and streaming response generation.
- `app/`: The primary Flutter application workspace.

## Development & Testing

When making changes to the Rust **engine**, verify the implementation using the following commands (requires high-performance features enabled):

```bash
# In the engine/ directory
cargo test tests::test_one_plus_one --features high_perf -- --nocapture
cargo test tests::test_sharded_one_plus_one --features high_perf -- --nocapture
```

## Tech Stack

- **Frontend**: Flutter (Dart)
- **Engine Core**: Rust
- **Deep Learning Framework**: [Burn](https://burn.dev/)
- **Bridge**: [flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge)
- **Model Support**: Qwen 2.5 (Safetensors)
