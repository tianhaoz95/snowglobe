#!/bin/bash
set -e

# Snowglobe 'full' Flavor Management Script
# This flavor enables CPU + GPU (Vulkan) + NPU (QNN) support.
# Usage: ./scripts/build_full_android.sh [build|run] [device_id]

COMMAND=${1:-build}
DEVICE_ID=$2

# Navigate to the demo directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# 1. Environment Configuration
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/home/tianhaoz/Android/Sdk/ndk/28.2.13676358}"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/opt/qcom/qnn}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/opt/qcom/hexagon}"

# Shared flags for both build and run
FLUTTER_FLAGS=(
  "--flavor" "full"
  "--dart-define=USE_LLAMACPP=true"
  "--dart-define=USE_LITERT=true"
)

echo "--------------------------------------------------"
echo "Action: $COMMAND"
echo "NDK: $ANDROID_NDK_ROOT"
echo "QNN SDK: $QNN_SDK_ROOT"
if [ -n "$DEVICE_ID" ]; then
  echo "Device: $DEVICE_ID"
fi
echo "--------------------------------------------------"

case $COMMAND in
  build)
    # Build APK specifically for arm64
    flutter build apk --release --target-platform android-arm64 "${FLUTTER_FLAGS[@]}"
    echo ""
    echo "Success! Build artifacts:"
    echo "APK: demo/build/app/outputs/flutter-apk/app-full-release.apk"
    ;;
  run)
    DEVICE_ARGS=()
    if [ -n "$DEVICE_ID" ]; then
      DEVICE_ARGS=("-d" "$DEVICE_ID")
    fi
    # Use --release for run to ensure NPU/GPU performance is accurate
    # Flutter run detects the platform from the device automatically
    flutter run --release "${DEVICE_ARGS[@]}" "${FLUTTER_FLAGS[@]}"
    ;;
  *)
    echo "Error: Unknown command '$COMMAND'"
    echo "Usage: $0 [build|run] [device_id]"
    exit 1
    ;;
esac
