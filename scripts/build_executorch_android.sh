#!/bin/bash
set -e

# Path to Android NDK
ANDROID_NDK=${ANDROID_NDK:-~/Library/Android/sdk/ndk/28.2.13676358}
# Resolve tilde if present
ANDROID_NDK="${ANDROID_NDK/#\~/$HOME}"

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: Android NDK not found at $ANDROID_NDK"
    exit 1
fi

EXECUTORCH_ROOT="$(pwd)/third_party/executorch"
OUT_DIR="$(pwd)/executorch-android"
mkdir -p "$OUT_DIR"

build_abi() {
    ABI=$1
    echo "Building ExecuTorch for $ABI..."
    BUILD_DIR="$EXECUTORCH_ROOT/cmake-android-$ABI"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Only enable QNN for arm64-v8a. QNN SDK and ExecuTorch Qualcomm backend
    # typically do not support 32-bit ARM or x86.
    ENABLE_QNN="OFF"
    if [ -n "$QNN_SDK_ROOT" ] && [ "$ABI" == "arm64-v8a" ]; then
        ENABLE_QNN="ON"
    fi

    cmake "$EXECUTORCH_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM=android-26 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_BUILD_QNN="$ENABLE_QNN" \
        -DQNN_SDK_ROOT="$QNN_SDK_ROOT"
        
    make -j$(sysctl -n hw.ncpu)
    
    # Create a unified structure for Rust build script
    ABI_OUT="$OUT_DIR/$ABI"
    mkdir -p "$ABI_OUT"
    find . -name "*.a" -exec cp {} "$ABI_OUT/" \;
    
    # Mirror subdirectory structure for backends/kernels if needed by build.rs
    mkdir -p "$ABI_OUT/backends/xnnpack/third-party/XNNPACK"
    cp backends/xnnpack/libxnnpack_backend.a "$ABI_OUT/backends/xnnpack/" || true
    cp backends/xnnpack/third-party/XNNPACK/libXNNPACK.a "$ABI_OUT/backends/xnnpack/third-party/XNNPACK/" || true
    cp backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a "$ABI_OUT/backends/xnnpack/third-party/XNNPACK/" || true
    
    # Copy Qualcomm backend if built
    if [ -d "backends/qualcomm" ]; then
        mkdir -p "$ABI_OUT/backends/qualcomm"
        if [ -f "backends/qualcomm/libqnn_executorch_backend.a" ]; then
            cp backends/qualcomm/libqnn_executorch_backend.a "$ABI_OUT/backends/qualcomm/"
        fi
        cp backends/qualcomm/libqnn_executorch_backend.so "$ABI_OUT/backends/qualcomm/" || true
        # Also copy any other .so libraries if they exist in build output
        find backends/qualcomm -name "*.so" -exec cp {} "$ABI_OUT/backends/qualcomm/" \;
    fi
    
    mkdir -p "$ABI_OUT/kernels/portable"
    cp kernels/portable/libportable_kernels.a "$ABI_OUT/kernels/portable/" || true
    cp kernels/portable/libportable_ops_lib.a "$ABI_OUT/kernels/portable/" || true
    
    mkdir -p "$ABI_OUT/kernels/optimized"
    cp kernels/optimized/liboptimized_kernels.a "$ABI_OUT/kernels/optimized/" || true
    cp kernels/optimized/liboptimized_ops_lib.a "$ABI_OUT/kernels/optimized/" || true
    
    # Copy generated headers (needed for the build script)
    # We only need to do this once, so we take it from the first ABI build
    if [ ! -d "$OUT_DIR/schema/include" ]; then
        echo "Preserving generated headers..."
        mkdir -p "$OUT_DIR/schema"
        cp -r schema/include "$OUT_DIR/schema/"
    fi

    # Cleanup the large build directory
    echo "Cleaning up build directory $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
    
    echo "Done building for $ABI"
}

build_abi "arm64-v8a"
build_abi "armeabi-v7a"
build_abi "x86_64"
build_abi "x86"

echo "Android ExecuTorch libraries built in $OUT_DIR"
echo "To build the Flutter app, run:"
echo "export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$OUT_DIR"
echo "cd demo && flutter run"
