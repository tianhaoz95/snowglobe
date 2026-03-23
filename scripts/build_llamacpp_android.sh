#!/bin/bash
set -e

echo "Building Snowglobe Android App with Llama.cpp (Vulkan enabled)..."

export LLAMA_VULKAN=1

# Find Android NDK dynamically if not set
if [ -z "$ANDROID_NDK_ROOT" ] && [ -n "$ANDROID_HOME" ]; then
    export ANDROID_NDK_ROOT=$(ls -1d $ANDROID_HOME/ndk/* | tail -n 1)
elif [ -z "$ANDROID_NDK_ROOT" ]; then
    export ANDROID_NDK_ROOT=$(ls -1d ~/Android/Sdk/ndk/* | tail -n 1)
fi

echo "Using NDK: $ANDROID_NDK_ROOT"

# Change to the demo directory to build the Flutter app
cd demo

# Ensure dependencies are fetched
flutter pub get

# Build the APK (arm64 only to avoid 32-bit cargo target feature panic in llama-cpp)
flutter build apk --release --target-platform android-arm64

echo "Android Llama.cpp build complete."
echo "To run on a connected device:"
echo "cd demo && flutter install"
