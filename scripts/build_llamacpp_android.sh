#!/bin/bash
set -e

echo "Building Snowglobe Android App with Llama.cpp (Vulkan enabled)..."

export LLAMA_VULKAN=1

# Change to the demo directory to build the Flutter app
cd demo

# Ensure dependencies are fetched
flutter pub get

# Build the APK
flutter build apk --release

echo "Android Llama.cpp build complete."
echo "To run on a connected device:"
echo "cd demo && flutter install"
