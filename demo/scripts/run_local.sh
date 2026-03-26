#!/bin/bash

# Find Android NDK dynamically if not set
if [ -z "$ANDROID_NDK_ROOT" ] && [ -n "$ANDROID_HOME" ]; then
    export ANDROID_NDK_ROOT=$(ls -1d $ANDROID_HOME/ndk/* | tail -n 1)
elif [ -z "$ANDROID_NDK_ROOT" ]; then
    export ANDROID_NDK_ROOT=$(ls -1d ~/Android/Sdk/ndk/* | tail -n 1)
fi

echo "Using NDK: $ANDROID_NDK_ROOT"

# Build the APK (arm64 only to avoid 32-bit cargo target feature panic in llama-cpp)
flutter build apk --release --target-platform android-arm64