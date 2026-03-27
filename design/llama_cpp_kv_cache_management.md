# Llama.cpp KV Cache Management in Engine

This report details how the KV cache is managed for the `llama.cpp` backend within the `engine` crate.

## 1. Context Initialization
The core component managing the KV cache in the `llama.cpp` backend is the `LlamaContext`. The context is created lazily when the first forward pass is executed for a session. It is stored inside the `EngineSession::state` as a `SafeLlamaContext` wrapper, which allows it to be shared safely across threads.

During initialization (`LlamaCppRunner::forward_all`), the context is created with specific parameters:
```rust
let ctx_params = LlamaContextParams::default()
    .with_n_ctx(std::num::NonZeroU32::new(4096))
    .with_n_seq_max(2); // Support at least 2 sequences for speculative decoding
```
The maximum number of tokens in the context (`n_ctx`) is set to 4096. More importantly, `n_seq_max` is set to 2. This explicitly reserves space to track two separate sequences within the single KV cache, which is vital for speculative decoding.

## 2. Token Batching and Sequence IDs
During generation, new tokens are added to a `LlamaBatch` before being decoded by the model. Each token added to the batch requires an associated sequence ID (`seq_id`).

In `LlamaCppRunner::forward_all`, the `seq_id` is determined by the session's state:
```rust
let seq_id = if session.is_speculative { 1 } else { 0 };
batch.add(llama_cpp_2::token::LlamaToken::new(token as i32), pos, &[seq_id], true)
```
- **Normal Generation:** Operates on sequence 0 (`seq_id = 0`). The KV cache is updated continuously as new tokens are confirmed and processed.
- **Speculative Verification:** Operates on sequence 1 (`seq_id = 1`). This allows the model to process draft tokens without polluting the main KV cache sequence (sequence 0).

## 3. Speculative Decoding and KV Cache Management
Speculative decoding uses a smaller "draft" model to guess upcoming tokens and a larger "target" model (the `llama.cpp` runner) to verify them. This requires managing the KV cache carefully so the draft tokens don't permanently corrupt the target model's state if they are rejected.

The `SpeculativeRunner` orchestrates this by using sequence 1 for verification.

### Preparation Phase
Before the target model evaluates the draft tokens, the KV cache state from the verified sequence (sequence 0) is duplicated into the speculative sequence (sequence 1).
```rust
fn prepare_speculative_verification(&self, session: &mut EngineSession) -> Result<(), String> {
    // ...
    ctx.copy_kv_cache_seq(0, 1, None, None)?;
    // ...
}
```

### Verification Phase
The draft tokens are then forwarded through the model with `session.is_speculative = true`, causing them to be evaluated against sequence 1. If the tokens are rejected, sequence 0 remains unaffected.

### Cleanup Phase
After verification, the speculative sequence (sequence 1) is cleared to free up memory and prepare for the next speculative step.
```rust
fn cleanup_speculative_verification(&self, session: &mut EngineSession) -> Result<(), String> {
    // ...
    ctx.clear_kv_cache_seq(Some(1), None, None)?;
    // ...
}
```

### Commit Phase
Once tokens are accepted, the `SpeculativeRunner` appends the accepted tokens to the final sequence and runs a forward pass with `session.is_speculative = false` (sequence 0). This permanently commits the accepted tokens to the main KV cache.

## Summary
The `llama.cpp` backend efficiently manages the KV cache by using multiple sequences within a single context. Sequence 0 is used for confirmed tokens, while Sequence 1 is utilized as a temporary scratchpad for speculative decoding. This enables seamless verification of draft tokens without needing to repeatedly clear or restore the main sequence's state manually.
