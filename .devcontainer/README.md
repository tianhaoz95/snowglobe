# Snowglobe Dev Container

This Dev Container provides a Linux (Ubuntu 22.04) environment for building the Snowglobe engine and running the Qualcomm model conversion script, which requires a Linux host.

## Prerequisites

- **Docker Desktop** or **OrbStack** installed and running on your host.
- **VS Code** with the **Dev Containers** extension.
- **Qualcomm QNN SDK**: Since the SDK is proprietary, you must have it on your host machine to mount it into the container.

## Getting Started

1.  Open the Snowglobe project in VS Code.
2.  VS Code should detect the `.devcontainer` folder and ask if you want to "Reopen in Container". Click **Reopen in Container**.
3.  If you have the QNN SDK at a specific path on your host (e.g., `~/qnn-sdk`), you should update `.devcontainer/devcontainer.json` to mount it:
    ```json
    "mounts": [
        "source=${localEnv:HOME}/qnn-sdk,target=/opt/qnn-sdk,type=bind,consistency=cached"
    ]
    ```
4.  Once the container is started, you can run the Qualcomm conversion:
    ```bash
    python3 converter/convert_qwen3_to_pte.py --backend qualcomm
    ```

## Building ExecuTorch Qualcomm Backend

If you need to rebuild the Qualcomm backend Python interface within the container:

```bash
cd third_party/executorch
./backends/qualcomm/scripts/build.sh
```

This will compile the necessary C++ libraries and copy them to the appropriate location for the Python conversion script to use.

## Features

- **Linux (Ubuntu 22.04)**: Essential for Qualcomm AOT model conversion.
- **Node.js & Gemini CLI**: The `@google/gemini-cli` is installed globally.
- **Rust & Flutter**: Pre-configured environment for cross-platform engine and UI development.
- **Android NDK**: Pre-installed for ExecuTorch Android builds.

## Environment Variables

The following environment variables are pre-configured in the container:
- `ANDROID_NDK_ROOT`: `/opt/android-ndk`
- `QNN_SDK_ROOT`: `/opt/qnn-sdk` (assumes you mount it here)
