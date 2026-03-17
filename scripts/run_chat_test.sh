#!/bin/bash
set -e

# Snowglobe Optimized Test Runner Script
# This script ensures that model assets are cached and pushed before running the integration test.

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Default to Qwen 3.5
MODEL_NAME="qwen3_5"
DEVICE_ID=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --device) DEVICE_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure ANDROID_NDK_ROOT is set for Rust builds
if [ -z "$ANDROID_NDK_ROOT" ]; then
    # Try to find NDK in common locations
    NDK_BASE="$HOME/Android/Sdk/ndk"
    if [ -d "$NDK_BASE" ]; then
        # Pick the latest version
        LATEST_NDK=$(ls -1 "$NDK_BASE" | sort -V | tail -n 1)
        if [ -n "$LATEST_NDK" ]; then
            export ANDROID_NDK_ROOT="$NDK_BASE/$LATEST_NDK"
            echo "Setting ANDROID_NDK_ROOT to $ANDROID_NDK_ROOT"
        fi
    fi
fi

echo "Ensuring model assets for $MODEL_NAME are synced and pushed..."
"$SCRIPTS_DIR/sync_test_assets.sh" --model "$MODEL_NAME" --push --device "$DEVICE_ID"

# Change to the demo directory to run the Flutter test
cd demo

# Ensure dependencies are fetched
flutter pub get

# Identify the device to use
if [ -z "$DEVICE_ID" ]; then
    # Try to find adb
    adb_cmd="adb"
    if ! command -v adb &> /dev/null; then
        if [ -f "$HOME/Android/Sdk/platform-tools/adb" ]; then
            adb_cmd="$HOME/Android/Sdk/platform-tools/adb"
        elif [ -f "/opt/android-sdk/platform-tools/adb" ]; then
            adb_cmd="/opt/android-sdk/platform-tools/adb"
        fi
    fi
    DEVICE_ID=$($adb_cmd devices | grep -v "List" | grep "device$" | head -n 1 | cut -f 1)
fi

if [ -z "$DEVICE_ID" ]; then
    echo "No Android device found. Defaulting to first available Flutter device."
    flutter test integration_test/chat_test.dart
else
    echo "Running integration test on device $DEVICE_ID..."
    flutter test integration_test/chat_test.dart -d "$DEVICE_ID"
fi
