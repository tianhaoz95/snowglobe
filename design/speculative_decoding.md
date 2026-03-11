# Speculative Decoding in Snowglobe

Speculative decoding is a technique that accelerates LLM inference by using a small, fast "draft" model to predict multiple tokens ahead of time, which are then verified in parallel by a larger, slower "target" model.

## Core Concepts

1.  **Draft Model**: A fast model (e.g., Qwen 2.5 0.5B) that generates $K$ tokens.
2.  **Target Model**: A larger, more accurate model (e.g., Qwen 2.5 7B via LlamaCpp) that verifies these tokens in one batch.
3.  **Verification Phase**: The target model evaluates the sequence $[T_{start}, T_{draft1}, T_{draft2}, ..., T_{draftK}]$.
    -   If the target model agrees with a draft token, that token is accepted.
    -   If the target model disagrees, the first disagreement's prediction is used, and subsequent draft tokens are discarded.

## Integration Strategy

The integration into Snowglobe can be achieved without major changes to the top-level architecture.

### 1. Enhancing `ModelRunner`

The current `ModelRunner::forward` returns only the last token's logits. For speculative decoding, we need the target model to return logits for *all* tokens in the batch.

**Proposed Update:**

```rust
pub trait ModelRunner: Send {
    /// Perform a forward pass, returning logits for all new tokens.
    /// The last element in the outer Vec corresponds to the logits for the next token.
    fn forward_all(&self, session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String>;

    /// Default implementation that returns the last token's logits for backwards compatibility.
    fn forward(&self, session: &mut EngineSession) -> Result<Vec<f32>, String> {
        let all_logits = self.forward_all(session)?;
        Ok(all_logits.into_iter().last().ok_or("No logits returned")?)
    }
}
```

-   **Burn (Qwen)**: `Model::forward` already returns the whole tensor; updating is straightforward.
-   **LlamaCpp**: Supports getting logits for any token in the batch via `get_logits_ith`.
-   **ExecuTorch (QwenPte)**: Returns logits for the entire fixed-length buffer (128); can easily extract any number of logits.

### 2. Implementing `SpeculativeRunner`

A `SpeculativeRunner` struct will implement `ModelRunner` and wrap both draft and target models.

```rust
pub struct SpeculativeRunner {
    draft: Box<dyn ModelRunner>,
    target: Box<dyn ModelRunner>,
    k: usize, // Lookahead window size
}

impl ModelRunner for SpeculativeRunner {
    fn forward_all(&self, session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String> {
        // ... Speculative orchestration logic ...
    }
}
```

### 3. Session State Management

`EngineSession` holds a `state` which contains the KV cache. `SpeculativeRunner` must manage two separate states (one for draft, one for target).

**Proposed Session State for Speculative Decoding:**

```rust
struct SpeculativeSessionState {
    draft_state: Box<dyn Any + Send + Sync>,
    target_state: Box<dyn Any + Send + Sync>,
    draft_offset: usize,
    target_offset: usize,
}
```

The `SpeculativeRunner` will swap these states into the `session` object before calling `forward` on the draft or target models.

### 4. The Speculative Loop

Inside `SpeculativeRunner::forward_all`:
1.  **Drafting**: Use `draft.forward` $K$ times to predict $K$ tokens. Append them to `session.tokens`.
2.  **Target Verification**: Run `target.forward_all` with all $K+1$ new tokens (including the last confirmed token).
3.  **Comparison**: Compare draft tokens with target predictions.
    -   Use a sampling or greedy strategy.
    -   Calculate how many draft tokens are accepted ($J \le K$).
4.  **Correction**:
    -   Keep $J$ tokens in `session.tokens`.
    -   Discard tokens from $J+1$ to $K$ in `session.tokens`.
    -   Add the $(J+1)$-th token (the one predicted by target model after the first disagreement).
5.  **State Sync**:
    -   Update `target_offset` by $J+1$.
    -   Roll back `draft_offset` to match the current target state and re-sync.

## Performance Considerations

-   **Batch Efficiency**: Speculative decoding is most effective when the cost of one large-batch target forward pass is significantly less than $K$ individual target forward passes.
-   **Device Placement**: On mobile devices, both models might compete for GPU resources. Placing the draft model on CPU (NdArray) and the target model on GPU (WGPU) could be a valid strategy to balance load.
-   **Context Length**: Both models must be able to handle the full context length of the session.

## Example Configuration

-   **Draft Model**: Qwen 2.5 0.5B (Burn, CPU/NdArray)
-   **Target Model**: Qwen 2.5 7B (LlamaCpp/GGUF, GPU/Vulkan)
-   **K (Lookahead)**: 4-6 tokens.

## Next Steps

1.  Update the `ModelRunner` trait to support `forward_all`.
2.  Implement `forward_all` for `Qwen`, `LlamaCppRunner`, and `QwenPte`.
3.  Implement the `SpeculativeRunner` orchestrator.
4.  Add a `Speculative` variant to `EngineVariant` in `engine/src/model/mod.rs`.
5.  Expose speculative configuration via `InitConfig`.
