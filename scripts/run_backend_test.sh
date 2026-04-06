#!/bin/bash
set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
DEVICE_ID=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) DEVICE_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Sync assets for both backends
echo "Syncing assets for llama.cpp..."
"$SCRIPTS_DIR/sync_test_assets.sh" --model "qwen3_5" --push --device "$DEVICE_ID"

echo "Syncing assets for liteRT..."
"$SCRIPTS_DIR/sync_test_assets.sh" --model "gemma4_e2b" --push --device "$DEVICE_ID"

# Change to demo directory
cd "$PROJECT_ROOT/demo"

# Identify the device to use if not provided
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

echo "Running backend integration test on device $DEVICE_ID..."
flutter test integration_test/backend_test.dart -d "$DEVICE_ID" --flavor highPerf
