# Snowglobe Demo & Benchmark

## Getting Started

```bash
flutter_rust_bridge_codegen generate --rust-features ""

# Build for Android with CPU only
RUST_FEATURES="" flutter run 

# Build with GPU (default)
flutter run

cp \
    ~/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
    ~/github/snowglobe/demo/build/rust_lib_snowglobedemo/jniLibs/debug/arm64-v8a
```
