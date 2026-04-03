# Supporting Gemma 4 in Snowglobe

Gemma 4 is a family of multimodal models from Google DeepMind, released in April 2026. This report outlines the technical requirements and implementation strategy for supporting Gemma 4 in the Snowglobe inference engine.

## 1. Gemma 4 Architecture Details

Gemma 4 introduces several key architectural advancements:

| Feature | Description | Snowglobe Status |
| :--- | :--- | :--- |
| **Hybrid Attention** | Interleaves local sliding-window attention (512-1024 tokens) with global full-context attention. | Not implemented in Burn core. |
| **p-RoPE** | Proportional Rotary Positional Embeddings for stability at 256K context. | Requires update to `rope.rs`. |
| **PLE** | Per-Layer Embeddings feeding a residual signal into every decoder layer. | Not implemented. |
| **MoE (26B model)** | 128 fine-grained experts with top-8 routing. | MoE not currently in Burn core. |
| **Thinking Mode** | Native support for `<|channel|>thought` reasoning delimiters. | Requires tokenizer/prompt updates. |
| **Multimodal** | Native Text, Image, and Audio processing. | Snowglobe API is currently text-only. |

## 2. Integration Strategy

### Phase 1: llama.cpp Backend (Immediate)

Since `llama.cpp` provides day-zero support for Gemma 4 (including vision and audio), this is the fastest path to support on-device inference in Snowglobe.

#### Implementation Steps:
1. **Update llama.cpp**: Sync `third_party/llama.cpp` to the latest version.
2. **Update llama-cpp-rs**: Ensure the Rust wrapper is compatible with the latest GGUF format for Gemma 4.
3. **Prompt Template Update**: Modify `engine/src/lib.rs` to detect Gemma 4 models and use the correct tokens.

**Gemma 4 Prompt Template:**
```text
<|turn|>
<|role|>system<|role|>
You are a helpful assistant.
<|turn|>
<|role|>user<|role|>
Hello!
<|turn|>
<|role|>assistant<|role|>
<|channel|>thought
Thinking process...
<|channel|>
Hello! How can I help you today?
```

### Phase 2: Native Burn Implementation (Medium Term)

Implementing Gemma 4 natively in Burn will allow for better optimization and backend flexibility.

#### Code Snippet: Proportional RoPE (p-RoPE)
We need to update `rope.rs` to support scaling factors for the global layers.

```rust
// engine/src/rope.rs

pub fn create_sin_cos_cache_proportional<B: Backend>(
    head_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    scaling_factor: f64, // New for p-RoPE
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Scaling logic for global layers
    let scaled_theta = rope_theta * scaling_factor;
    // ... existing create_sin_cos_cache logic ...
}
```

#### Code Snippet: Per-Layer Embeddings (PLE)
PLE requires an additional embedding table that is added as a residual to each layer.

```rust
// engine/src/model/gemma4.rs

pub struct Gemma4Block<B: Backend> {
    self_attn: Gemma4Attention<B>,
    mlp: Gemma4MLP<B>,
    ple_projection: Linear<B>, // PLE residual signal
    // ...
}

impl<B: Backend> Gemma4Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>, ple_signal: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x + self.ple_projection.forward(ple_signal);
        // ... standard transformer forward ...
    }
}
```

### Phase 3: Multimodal Support (Long Term)

To fully utilize Gemma 4's capabilities, the Snowglobe API needs to be extended to accept image and audio inputs.

**Proposed API Change:**
```rust
pub enum Modality {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<u8>),
}

pub struct ChatCompletionRequestPart {
    pub modality: Modality,
}
```

## 3. Model Conversion

Gemma 4 models from Hugging Face must be converted into formats supported by Snowglobe's backends:

- **GGUF (for llama.cpp):** Use the standard `llama.cpp` converter.
  ```bash
  python third_party/llama.cpp/convert_hf_to_gguf.py path/to/gemma4-hf --outfile model.gguf
  ```
- **Safetensors (for Burn):** A custom script is needed to map HF weights to the Snowglobe Gemma 4 architecture.
  - Specifically, `layer_scalar` and `ple_projection` weights must be handled.
- **PTE (for ExecuTorch):** Requires a new exporter in the `converter/` directory.

## 4. Recommended Roadmap

1.  **Week 1**: Bump `llama.cpp` and verify GGUF inference for Gemma 4 2B/4B models.
2.  **Week 2**: Update the Flutter Demo app to support the Gemma 4 prompt template and "thinking" mode display.
3.  **Week 3-4**: Begin implementation of `p-RoPE` and `Hybrid Attention` in the native Burn engine.
4.  **Week 5+**: Research integration of `mmproj` (multimodal projector) from `llama.cpp` into the Snowglobe bridge.
