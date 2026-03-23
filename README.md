# Snowglobe

High-performance LLM inference engine built with Rust and Flutter.

## Running Demo App on Android with Llama.cpp

You can build and run the Flutter demo app for Android using the Llama.cpp backend (with Vulkan enabled for GPU acceleration).

### Prerequisites

Ensure you have your Android NDK installed. If `ANDROID_HOME` or `ANDROID_NDK_ROOT` is not set, the build script will attempt to find it dynamically.

### Build and Run

1. **Build the APK**:
   We provide a script to handle the environment setup and build the release APK for `arm64-v8a`:
   ```bash
   ./scripts/build_llamacpp_android.sh
   ```
   After the build is complete, you can install it on a connected device via:
   ```bash
   cd demo
   flutter install
   ```

2. **Run directly via Flutter**:
   If you want to run the app directly (e.g. for development or debugging) on a connected Android device:
   ```bash
   cd demo
   
   # Enable Llama.cpp Vulkan backend
   export LLAMA_VULKAN=1
   
   # Provide NDK path for the Rust build script
   export ANDROID_NDK_ROOT=$(ls -1d ~/Android/Sdk/ndk/* | tail -n 1) # or set to your specific NDK path
   
   # Run the app
   flutter run
   ```

## Testing ExecuTorch (Experimental)

1. **Export Model**:
   ```bash
   # For Mac (Metal)
   python converter/convert_qwen3_to_pte.py --backend mps
   # For CPU
   python converter/convert_qwen3_to_pte.py --backend xnnpack
   ```

2. **Run Engine Test**:
   ```bash
   cd engine
   export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=~/github/snowglobe/third_party/executorch/cmake-out
   
   # Enable this only if you exported with --backend mps
   export EXECUTORCH_USE_MPS=1 
   
   cargo test tests::test_one_plus_one_pte --release -- --nocapture
   ```