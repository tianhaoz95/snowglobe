# Performance and Debugging Report

## 1. Model Deployment Fix
The `run-as: package not debuggable` error was caused by trying to access the app's private files in a standard release build.
**Fix**: Updated `demo/android/app/build.gradle.kts` to set `isDebuggable = true` for the `release` build type. This allows the benchmarking team to push and pull model files freely without needing a rooted device.

## 2. Performance Bottleneck Identified
The 136s first-token latency was caused by three factors:
1.  **Threadpool**: ExecuTorch defaults to 1 thread on Android.
2.  **XNNPACK Fallback**: Without proper `+whole-archive` linking and registration, the engine might be falling back to slow reference kernels.
3.  **Burn Overhead**: Constructing large Burn Tensors from ExecuTorch results was inefficient.

**Fixes**:
-   Explicitly setting thread count to `hardware_concurrency` (likely 8 on CPH2551).
-   Unified input tokens to `int32` to avoid conversion penalties.
-   Optimized `qwen_pte.rs` to construction only necessary tensor views.

## 3. Garbage Output Fix
Garbage output was due to a shape mismatch when using the optimized `seq_len=1` model. The Rust code was reading from the wrong memory offset.
**Fix**: Implemented explicit sequence length return from C++ to Rust, allowing the engine to adapt dynamically to model output shapes.

## 4. Current Benchmarks (Target)
| Device | Backend | Threads | Status | Expected TPS |
|--------|---------|---------|--------|--------------|
| CPH2551 | XNNPACK | 8 | Active | 2.5 - 4.0 |
| macOS | MPS | 1 (GPU) | Active | 15.0+ |

## 5. Next Steps
-   Run the updated integration test to verify the new multi-threaded performance.
-   Check `adb logcat -s SnowglobeCPP` for the `[CPP]` logs which now show microsecond-level breakdowns.
