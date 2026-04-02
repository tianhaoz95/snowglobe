#!/bin/bash
set -e

# Snowglobe Full Hardware Support Build & Install Script
# This script builds the Android app with Vulkan (GPU) and QNN (NPU) support
# and installs it on the first connected device.

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
DEMO_DIR="$PROJECT_ROOT/demo"

# Configuration - Update these paths if necessary
export ANDROID_NDK_ROOT="/home/tianhaoz/Android/Sdk/ndk/29.0.14206865"
export QNN_SDK_ROOT="/home/tianhaoz/SDK/qnn"
export HEXAGON_SDK_ROOT="/home/tianhaoz/SDK/qnn/lib/hexagon-v79"

echo "Checking for connected Android devices..."
DEVICE_ID=$(adb devices | grep -v "List" | grep "device" | head -n 1 | awk '{print $1}')

if [ -z "$DEVICE_ID" ]; then
    echo "Error: No Android device connected via ADB."
    exit 1
fi

echo "Using device: $DEVICE_ID"

echo "Building Snowglobe Demo (Flavor: full)..."
cd "$DEMO_DIR"

# Build release APK for arm64 (Standard for modern NPU/GPU devices)
flutter build apk --release --flavor full --target-platform android-arm64

APK_PATH="build/app/outputs/flutter-apk/app-full-release.apk"

if [ -f "$APK_PATH" ]; then
    echo "Installing APK to device $DEVICE_ID..."
    adb -s "$DEVICE_ID" install -r "$APK_PATH"
    echo "Installation complete. You can now launch Snowglobe Demo from your app drawer."
else
    echo "Error: APK not found at $APK_PATH"
    exit 1
fi
