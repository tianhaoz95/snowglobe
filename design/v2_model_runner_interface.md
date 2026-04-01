# Design Document: ModelRunner Interface V2

## Status: Proposed
## Author: Gemini CLI
## Date: 2026-03-31

### 1. Objective
Redesign the `ModelRunner` interface to support modern LLM inference features including **Chunked Prefill**, **Speculative Decoding**, and **Paged KV Cache (SGLang style)**. The new interface decouples the execution logic from the request state, enabling efficient memory management and hardware-specific optimizations.

### 2. Proposed Interface

```rust
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

/// Unique identifier for a logical user request or agent stream.
pub type RequestId = u64;

/// Defines the hardware execution intent. 
pub enum ExecutionMode {
    /// Initial prompt processing. Backends prioritize high matrix-throughput.
    Prefill,
    /// Generating a single token. Backends prioritize low matrix-vector latency.
    Decode,
    /// Verifying N draft tokens. Backend must apply causal + lookahead masking
    /// and should NOT finalize KV updates until verified.
    Verify { draft_len: usize },
}

/// The state of a logical request. The ModelRunner modifies the data inside.
pub struct EngineSession {
    pub request_id: RequestId,
    /// Total tokens currently processed and stored in the hardware KV cache.
    pub current_kv_len: usize,
    /// Backend-specific, opaque state pointer (e.g., llama_context*, Burn Tensors).
    pub backend_state: Option<Box<dyn Any + Send + Sync>>,
    pub metadata: HashMap<String, String>,
}

/// A non-owning view over returned logits to minimize allocations.
pub struct LogitView<'a> {
    pub data: &'a [f32],
    pub shape: (usize, usize), 
}

/// Hardware capabilities report.
pub struct BackendInfo {
    pub name: String,
    pub max_sequence_length: usize,
    pub max_batch_size: usize,
}

pub trait ModelRunner: Send + Sync {
    /// 1. Lifecycle: Initialization
    fn load(path: &Path, config: &serde_json::Value) -> Result<Box<Self>, String>
    where
        Self: Sized;

    /// 2. Core Execution
    fn execute(
        &self,
        session: &mut EngineSession,
        input_tokens: &[u32],
        mode: ExecutionMode,
    ) -> Result<LogitView, String>;

    /// 3. KV Cache State Management
    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String>;

    fn fork_state(&self, session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String>;

    /// 4. Introspection
    fn get_backend_info(&self) -> BackendInfo;
}
```

### 3. Viability Analysis

| Feature | Viability | Technical Justification |
| :--- | :--- | :--- |
| **ExecutionMode** | **High** | Essential for selecting optimal kernels (GEMM for prefill vs GEMV for decode). Matches vLLM/SGLang architectures. |
| **Any State** | **High** | Allows the `ModelRunner` to remain stateless regarding specific requests, facilitating easier swap-in/swap-out for paged attention. |
| **LogitView** | **Medium-High** | Drastically reduces GC pressure and allocations. Requires careful lifetime management to ensure the runner's internal buffer is not reused while the view is active. |
| **fork_state** | **Medium** | Powerful for multi-turn/prefix caching. Implementation difficulty varies by backend (Burn requires tensor cloning; llama.cpp requires context duplication). |

### 4. Implementation Plan

#### Phase 1: Core Types & Trait (Week 1)
- [ ] Implement new types in `engine/src/model/runner.rs`.
- [ ] Define the `ModelRunner` V2 trait.
- [ ] Add `EngineSession` management utilities (e.g., `SessionManager`).

#### Phase 2: Burn Backend Migration (Week 1-2)
- [ ] Update `engine/src/model/qwen.rs` to support `ExecutionMode`.
- [ ] Move `KVCache<B>` into the `EngineSession::backend_state`.
- [ ] Implement `truncate_cache` by slicing Burn tensors.
- [ ] Implement `fork_state` by cloning tensors (Copy-on-Write).

#### Phase 3: ExecuTorch & llama.cpp (Week 2)
- [ ] Update `qwen_pte.rs` and `llama_cpp.rs` runners.
- [ ] Map `ExecutionMode::Verify` to `llama_cpp`'s batching API.
- [ ] Implement `truncate_cache` using `llama_kv_cache_seq_rm`.

#### Phase 4: Integration & Orchestration (Week 3)
- [ ] Update `engine/src/lib.rs` to expose the session-based API internally.
- [ ] Refactor `create_chat_completion` to use `ModelRunner::execute` while maintaining the public `CreateChatCompletionRequest` signature.
- [ ] Implement internal **Prefix Caching** by hashing message histories to reuse `EngineSession`s.
- [ ] Refactor `flutter_rust_bridge` calls in `demo/rust/src/api.rs`.

### 5. API Compatibility & Orchestration
The top-level OpenAI-compatible API (e.g., `CreateChatCompletionRequest`, `CreateChatCompletionResponse`) remains **unchanged**. The `ModelRunner` V2 is an internal architectural improvement.

- **Stateless to Stateful Mapping**: The orchestration layer in `engine/src/lib.rs` will map incoming stateless OpenAI requests to internal `EngineSession` objects.
- **Automated Mode Switching**: The streaming loop will automatically handle the transition from `ExecutionMode::Prefill` (for the initial prompt) to `ExecutionMode::Decode` (for subsequent tokens).
- **Prefix Caching**: By utilizing `truncate_cache` and `fork_state`, the engine can reuse KV caches for shared prefixes (e.g., system prompts) across different requests without changing the user-facing API.

### 6. Testing Strategy
To ensure the robustness of the new `ModelRunner` interface across platforms, the following verification steps are required:

- **Burn Backend**: 
  - Must pass unit and integration tests on the **Linux host** (CPU/NdArray).
  - Verify `ExecutionMode` switching and `truncate_cache` accuracy.
- **llama.cpp Backend**:
  - Must pass unit tests on the **Linux host**.
  - Must pass full integration tests (using `scripts/run_chat_test.sh`) on a **connected Android device** to verify hardware acceleration (Vulkan/OpenCL) and memory stability with the new session-based architecture.
- **Regression Testing**:
  - Ensure existing OpenAI-compatible API tests pass without modification.
  - Verify that **Prefix Caching** correctly identifies and reuses `EngineSession`s for identical message headers.

### 7. Open Questions
1. **LogitView Lifetime**: Should `LogitView` be tied to the `ModelRunner` or the `EngineSession`? (Current recommendation: the Runner, assuming it has a stable output buffer).
2. **Metadata usage**: What standard keys should we support in `EngineSession::metadata` (e.g., `"priority"`, `"user_id"`)?
