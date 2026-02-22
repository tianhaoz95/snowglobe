# Snowglobe

High-performance LLM inference engine built with Rust and Flutter.

## Testing ExecuTorch (Experimental)

1. **Export Model**:
   ```bash
   # For Mac (Metal)
   python converter/convert_qwen3_to_pte.py --backend mps
   # For CPU
   python converter/convert_qwen3_to_pte.py --backend xnnpack
   ```

2. **Run Engine Test**:
   ```bash
   cd engine
   export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=~/github/snowglobe/third_party/executorch/cmake-out
   
   # Enable this only if you exported with --backend mps
   export EXECUTORCH_USE_MPS=1 
   
   cargo test tests::test_one_plus_one_pte --release -- --nocapture
   ```