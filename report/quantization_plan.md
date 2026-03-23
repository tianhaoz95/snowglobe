# Snowglobe: 8-bit Quantization and Android Deployment Plan

## Objective
Support 8-bit quantization for Qwen3 models using ExecuTorch and deploy the quantized model to Android for high-performance on-device inference.

## Current Status
- `convert_qwen3_to_pte.py` exports models in FP32 or FP16 (implicitly if model is cast) to `.pte`.
- Backends supported: `mps` (Mac), `xnnpack` (CPU/Android/iOS), `portable`.
- Android builds for ExecuTorch are already available via `scripts/build_executorch_android.sh`.

## Plan for 8-bit Quantization

### 1. Identify Quantization Strategy
For LLMs, the most effective balance between accuracy and performance is usually **Weight-Only 8-bit Quantization**.
- **Pros**: Significant memory reduction (up to 4x), lower disk footprint, faster loading.
- **Cons**: Some accuracy degradation compared to FP16, though minimal for 8-bit weights.
- **Implementation**: We will use `XNNPACKQuantizer` from ExecuTorch for Android/XNNPACK targets.

### 2. Update `convert_qwen3_to_pte.py`
- Add a `--quantize` argument to the script.
- Integration:
  - Load the model in FP32.
  - Apply `XNNPACKQuantizer` to quantize weights to `qint8` or `uint8`.
  - Export to Edge IR using `to_edge`.
  - Use `XnnpackPartitioner` for final backend delegation.

### 3. Verification & Metrics
- Verify the `.pte` file size. A 0.6B model in FP32 is ~2.4GB. With 8-bit quantization, it should be ~0.6-0.7GB.
- Test the quantized model for basic sanity (e.g., predicted next token consistency).

## Android Deployment Strategy

### 1. Build ExecuTorch for Android
- Ensure `scripts/build_executorch_android.sh` runs successfully.
- Libraries will be placed in `tmp/executorch-android/`.

### 2. Deploy Model to Device
- Use `adb push` to move the quantized `.pte` file to `/data/local/tmp/snowglobe/`.
- Update the Flutter demo code or configuration to use the quantized model path.

### 3. Run Benchmark/Demo
- Run the Flutter app with `flutter run --release`.
- Measure latency (tokens per second) and memory usage.

## Timeline
1. **Plan & Setup**: [Current Step]
2. **Quantization Implementation**: Update `convert_qwen3_to_pte.py`.
3. **Android Build & Push**: Verify binaries and push model.
4. **Final Report**: Document performance results.
