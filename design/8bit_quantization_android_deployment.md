# Report: 8-bit Quantization Support and Android Deployment Plan

## 1. Overview
This report outlines the strategy for implementing 8-bit quantization for the Qwen3 LLM and deploying it to Android devices using the ExecuTorch framework within the Snowglobe project.

## 2. 8-bit Quantization Strategy

### 2.1 Technology Selection
We will utilize **Dynamic Symmetric Quantization** provided by the PyTorch `torch.ao` and ExecuTorch `XNNPACK` backend.

*   **Quantizer**: `executorch.backends.xnnpack.quantizer.xnnpack_quantizer.XNNPACKQuantizer`
*   **Mode**: Dynamic Quantization. This is preferred for LLMs as it quantizes weights offline but computes activations dynamically at runtime, avoiding the need for a representative calibration dataset while still providing significant speedups and memory savings.
*   **Granularity**: Per-channel quantization for weights to maintain high accuracy.

### 2.2 Expected Impact
*   **Memory Footprint**: For the Qwen3-0.6B model, the weight size will reduce from ~2.4GB (FP32) or ~1.2GB (FP16) to approximately **0.65GB**.
*   **Inference Speed**: XNNPACK is highly optimized for 8-bit integer operations on ARM CPUs, which should lead to improved tokens-per-second on Android devices.
*   **Disk Storage**: Significant reduction in `.pte` file size, facilitating easier distribution and faster loading.

## 3. Implementation Plan

### 3.1 Converter Script Enhancements (`converter/convert_qwen3_to_pte.py`)
1.  Add a `--quantize` CLI argument.
2.  Incorporate the following quantization workflow:
    *   Export the model to an `ExportedProgram` using `torch.export`.
    *   Initialize `XNNPACKQuantizer` with `get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)`.
    *   Apply `prepare_pt2e` and `convert_pt2e` from `torch.ao.quantization`.
    *   Lower the quantized model to Edge IR via `to_edge`.
    *   Delegate to the XNNPACK backend using `XnnpackPartitioner`.

### 3.2 Verification
*   Execute the modified converter with `--backend xnnpack --quantize`.
*   Validate the generated `qwen3_0.6b_xnnpack_q8.pte` for basic functional correctness using a next-token prediction test.

## 4. Android Deployment Workflow

### 4.1 Binary Preparation
The existing `scripts/build_executorch_android.sh` will be used to compile the ExecuTorch runtime and XNNPACK backend for `arm64-v8a` and `x86_64` architectures.
The resulting static libraries will be linked into the Rust engine via the `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` environment variable.

### 4.2 Model Deployment
1.  **Transfer**: Use `adb push` to move the quantized `.pte` file to the Android device's local storage (e.g., `/data/local/tmp/snowglobe/`).
2.  **App Configuration**: Update the `demo` Flutter application to point to the new model path.

### 4.3 Execution & Benchmarking
*   Launch the application on the target Android device (e.g., CPH2551).
*   Monitor logs for successful XNNPACK delegation.
*   Measure latency and peak memory usage during inference.

## 5. Conclusion
Implementing 8-bit dynamic quantization is a critical step for making Snowglobe viable on mobile hardware. The proposed plan leverages standard PyTorch 2.0 quantization APIs and the specialized XNNPACK backend to achieve a balance between implementation simplicity and runtime performance.
