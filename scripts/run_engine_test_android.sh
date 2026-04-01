#!/bin/bash
set -e

# Snowglobe Android Engine Test Runner
# This script builds the Rust engine tests for Android and runs them on a connected device.

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
ENGINE_DIR="$PROJECT_ROOT/engine"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
REMOTE_DIR="/data/local/tmp/snowglobe_test"

# Default values
MODEL_NAME="qwen3"
DEVICE_ID=""
SKIP_BUILD=false
ACCELERATOR="npu" # Default to NPU

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --device) DEVICE_ID="$2"; shift ;;
        --skip-build) SKIP_BUILD=true ;;
        --accelerator) ACCELERATOR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

TEST_ASSETS_DIR="$PROJECT_ROOT/.test_assets/$MODEL_NAME"

if [[ ! "$ACCELERATOR" =~ ^(cpu|gpu|npu)$ ]]; then
    echo "Error: Accelerator must be cpu, gpu, or npu."
    exit 1
fi

echo "Target Accelerator: $ACCELERATOR"

# Ensure ANDROID_NDK_ROOT is set
if [ -z "$ANDROID_NDK_ROOT" ]; then
    NDK_BASE="$HOME/Android/Sdk/ndk"
    if [ -d "$NDK_BASE" ]; then
        LATEST_NDK=$(ls -1 "$NDK_BASE" | sort -V | tail -n 1)
        if [ -n "$LATEST_NDK" ]; then
            export ANDROID_NDK_ROOT="$NDK_BASE/$LATEST_NDK"
        fi
    fi
fi

if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT is not set and could not be found."
    exit 1
fi

NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
if [ ! -d "$NDK_BIN" ]; then
    echo "Error: NDK bin directory not found at $NDK_BIN"
    exit 1
fi

# Identify the device
if [ -z "$DEVICE_ID" ]; then
    DEVICE_ID=$(adb devices | grep -v "List" | grep "device$" | head -n 1 | cut -f 1)
fi

if [ -z "$DEVICE_ID" ]; then
    echo "Error: No Android device found via adb."
    exit 1
fi

echo "Using device: $DEVICE_ID"
echo "NDK: $ANDROID_NDK_ROOT"

if [ "$SKIP_BUILD" = false ]; then
    echo "Building engine tests for aarch64-linux-android..."
    cd "$ENGINE_DIR"
    
    # Set up environment for cross-compilation
    export PATH="$NDK_BIN:$PATH"
    export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang"
    export CXX_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang++"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_BIN/aarch64-linux-android24-clang"
    export EXECUTORCH_RS_EXECUTORCH_LIB_DIR="$PROJECT_ROOT/executorch-android"
    
    # If forcing CPU, we temporarily "hide" the qualcomm backend so build.rs doesn't link it
    QUALCOMM_DIR="$PROJECT_ROOT/executorch-android/arm64-v8a/backends/qualcomm"
    HIDDEN_QUALCOMM="$PROJECT_ROOT/executorch-android/arm64-v8a/backends/qualcomm_hidden"
    
    if [ "$ACCELERATOR" = "cpu" ]; then
        if [ -d "$QUALCOMM_DIR" ]; then
            echo "Temporarily hiding Qualcomm backend to force CPU build..."
            mv "$QUALCOMM_DIR" "$HIDDEN_QUALCOMM"
            # Use trap to ensure we restore even if build fails
            trap "mv '$HIDDEN_QUALCOMM' '$QUALCOMM_DIR' 2>/dev/null || true" EXIT
        fi
    fi

    cargo test --no-run --target aarch64-linux-android --release --features llamacpp,vulkan
    
    # Restore if we hid it
    if [ -d "$HIDDEN_QUALCOMM" ]; then
        mv "$HIDDEN_QUALCOMM" "$QUALCOMM_DIR"
        trap - EXIT
    fi

    TEST_BIN=$(find target/aarch64-linux-android/release/deps -name "snowglobe-*" -executable -not -name "*.so" | head -n 1)
    if [ -z "$TEST_BIN" ]; then
        echo "Error: Could not find compiled test binary."
        exit 1
    fi
    cp "$TEST_BIN" "$ENGINE_DIR/target/android_test_bin_$ACCELERATOR"
    echo "Build successful: target/android_test_bin_$ACCELERATOR"
fi

TEST_BIN="$ENGINE_DIR/target/android_test_bin_$ACCELERATOR"

echo "Preparing device directory $REMOTE_DIR..."
adb -s "$DEVICE_ID" shell "mkdir -p $REMOTE_DIR"

echo "Pushing test binary..."
adb -s "$DEVICE_ID" push "$TEST_BIN" "$REMOTE_DIR/snowglobe_test"

echo "Pushing model assets from $TEST_ASSETS_DIR..."
# Ensure assets exist
if [ ! -f "$TEST_ASSETS_DIR/model.pte" ]; then
    echo "Error: model.pte not found in $TEST_ASSETS_DIR. Running sync_test_assets.sh..."
    "$SCRIPTS_DIR/sync_test_assets.sh" --model "$MODEL_NAME"
fi

adb -s "$DEVICE_ID" push "$TEST_ASSETS_DIR/model.pte" "$REMOTE_DIR/model.pte"
adb -s "$DEVICE_ID" push "$TEST_ASSETS_DIR/tokenizer.json" "$REMOTE_DIR/tokenizer.json"
adb -s "$DEVICE_ID" push "$TEST_ASSETS_DIR/config.json" "$REMOTE_DIR/config.json"

echo "Pushing shared libraries..."
# Always push libc++_shared.so
LIBCXX_PATH=$(find "$ANDROID_NDK_ROOT" -name "libc++_shared.so" | grep "aarch64" | head -n 1)
if [ -n "$LIBCXX_PATH" ]; then
    echo "Pushing $LIBCXX_PATH..."
    adb -s "$DEVICE_ID" push "$LIBCXX_PATH" "$REMOTE_DIR/"
fi

# Push QNN libraries ONLY if accelerator is npu or gpu
if [ "$ACCELERATOR" != "cpu" ]; then
    echo "Pushing QNN/Qualcomm libraries..."
    # Always push the primary backend bridge
    adb -s "$DEVICE_ID" push "$PROJECT_ROOT/executorch-android/arm64-v8a/backends/qualcomm/libqnn_executorch_backend.so" "$REMOTE_DIR/"

    QNN_LIB_DIR="$HOME/SDK/qnn/lib/aarch64-android"
    if [ -d "$QNN_LIB_DIR" ]; then
        echo "Pushing QNN libraries from $QNN_LIB_DIR..."
        if [ "$ACCELERATOR" = "gpu" ]; then
            adb -s "$DEVICE_ID" push "$QNN_LIB_DIR/libQnnGpu.so" "$REMOTE_DIR/"
            adb -s "$DEVICE_ID" push "$QNN_LIB_DIR/libQnnSystem.so" "$REMOTE_DIR/"
        else
            # NPU: push everything
            adb -s "$DEVICE_ID" push "$QNN_LIB_DIR/"*.so "$REMOTE_DIR/"
        fi
    fi

    # Push Hexagon skeletons (needed for NPU)
    if [ "$ACCELERATOR" = "npu" ]; then
        HEXAGON_LIB_DIR=$(find "$HOME/SDK/qnn/lib" -name "hexagon-v*" -type d | sort -V | tail -n 1)
        if [ -n "$HEXAGON_LIB_DIR" ]; then
            HEXAGON_SKEL_DIR="$HEXAGON_LIB_DIR/unsigned"
            if [ -d "$HEXAGON_SKEL_DIR" ]; then
                echo "Pushing Hexagon skeletons from $HEXAGON_SKEL_DIR..."
                adb -s "$DEVICE_ID" push "$HEXAGON_SKEL_DIR/"*.so "$REMOTE_DIR/"
            fi
        fi
    fi
fi

echo "Running test on device (Accelerator: $ACCELERATOR, timeout: 5m)..."
# Set environment variables for backends
ENVS="export SNOWGLOBE_TEST_DIR=$REMOTE_DIR"
ENVS="$ENVS && export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH"

if [ "$ACCELERATOR" = "npu" ]; then
    ENVS="$ENVS && export ADSP_LIBRARY_PATH=$REMOTE_DIR"
elif [ "$ACCELERATOR" = "gpu" ]; then
    # Some QNN implementations check this
    ENVS="$ENVS && export QNN_BACKEND_PATH=$REMOTE_DIR/libQnnGpu.so"
fi

timeout 5m adb -s "$DEVICE_ID" shell "chmod +x $REMOTE_DIR/snowglobe_test && \
    $ENVS && \
    $REMOTE_DIR/snowglobe_test tests::test_one_plus_one_pte --nocapture" || echo "Test TIMED OUT after 5 minutes"
