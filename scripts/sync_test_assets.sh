#!/bin/bash
set -e

# Snowglobe Test Asset Sync Script
# This script manages model assets for integration testing to avoid repeated downloads.

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
ASSETS_DIR="$PROJECT_ROOT/.test_assets"

# Model definitions
QWEN3_5_NAME="qwen3_5"
QWEN3_5_GGUF_URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"
QWEN3_5_TOKENIZER_URL="https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json"
QWEN3_5_CONFIG_URL="https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json"

function download_if_missing() {
    local model_name=$1
    local file_name=$2
    local url=$3
    local target_dir="$ASSETS_DIR/$model_name"
    local target_path="$target_dir/$file_name"

    mkdir -p "$target_dir"

    if [ ! -f "$target_path" ]; then
        echo "Downloading $file_name for $model_name..."
        curl -L "$url" -o "$target_path"
    else
        echo "$file_name for $model_name already exists in cache."
    fi
}

function push_to_android() {
    local model_name=$1
    local device_id=$2
    local target_dir="/data/local/tmp/snowglobe/$model_name"

    # Try to find adb
    local adb_cmd="adb"
    if ! command -v adb &> /dev/null; then
        if [ -f "$HOME/Android/Sdk/platform-tools/adb" ]; then
            adb_cmd="$HOME/Android/Sdk/platform-tools/adb"
        elif [ -f "/opt/android-sdk/platform-tools/adb" ]; then
            adb_cmd="/opt/android-sdk/platform-tools/adb"
        else
            echo "adb not found in path or common locations. Skipping push."
            return 1
        fi
    fi

    echo "Checking for connected Android devices..."
    if [ -z "$device_id" ]; then
        device_id=$($adb_cmd devices | grep -v "List" | grep "device$" | head -n 1 | cut -f 1)
    fi

    if [ -n "$device_id" ]; then
        echo "Pushing $model_name assets to device $device_id using $adb_cmd..."
        $adb_cmd -s "$device_id" shell "mkdir -p $target_dir"
        $adb_cmd -s "$device_id" push "$ASSETS_DIR/$model_name/." "$target_dir/"
        echo "Assets pushed successfully to $target_dir"
    else
        echo "No Android device found. Skipping push."
    fi
}

# Default to Qwen 3.5
MODEL_NAME=$QWEN3_5_NAME

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --push) PUSH_TO_DEVICE=true ;;
        --device) DEVICE_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

case $MODEL_NAME in
    qwen3_5)
        download_if_missing "qwen3_5" "model.gguf" "$QWEN3_5_GGUF_URL"
        download_if_missing "qwen3_5" "tokenizer.json" "$QWEN3_5_TOKENIZER_URL"
        download_if_missing "qwen3_5" "config.json" "$QWEN3_5_CONFIG_URL"
        ;;
    qwen3)
        # Use locally generated PTE if it exists, otherwise would need URL
        mkdir -p "$ASSETS_DIR/qwen3"
        download_if_missing "qwen3" "tokenizer.json" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
        download_if_missing "qwen3" "config.json" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json"
        ;;
    qwen2_5_pte)
        # Use locally generated PTE
        mkdir -p "$ASSETS_DIR/qwen2_5_pte"
        if [ -f "qwen_qwen2.5_0.5b_instruct_int8.pte" ]; then
            cp "qwen_qwen2.5_0.5b_instruct_int8.pte" "$ASSETS_DIR/qwen2_5_pte/model.pte"
        fi
        download_if_missing "qwen2_5_pte" "tokenizer.json" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"
        download_if_missing "qwen2_5_pte" "config.json" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json"
        ;;
    *)
        echo "Model $MODEL_NAME not yet configured in this script."
        exit 1
        ;;
esac

if [ "$PUSH_TO_DEVICE" = true ]; then
    push_to_android "$MODEL_NAME" "$DEVICE_ID"
fi

echo "Asset sync complete."
