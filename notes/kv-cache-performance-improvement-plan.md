# Snowglobe Development Plan

## Performance Optimizations

### KV Cache Efficiency
- **Current Issue**: The KV cache implementation currently uses `Tensor::cat` on every generation step to append new keys and values. This leads to frequent memory reallocations and tensor copies, which increases overhead as the sequence length grows.
- **Proposed Improvement**: 
    - Implement a pre-allocated KV cache buffer with a fixed maximum context window.
    - Use slice-update (in-place modification if supported by the backend) to insert new tokens into the cache.
    - This will improve inference latency and reduce memory fragmentation.

## Feature Roadmap
- [ ] Implement pre-allocated KV cache.
- [ ] Add support for more Qwen 2.5 model sizes.
- [ ] Improve GPU memory management (e.g., paging).
- [ ] Add benchmark suite for different backends.
