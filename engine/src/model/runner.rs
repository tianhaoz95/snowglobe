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

impl EngineSession {
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            current_kv_len: 0,
            backend_state: None,
            metadata: HashMap::new(),
        }
    }
}

/// A non-owning view over returned logits to minimize allocations.
pub struct LogitView {
    pub data: Vec<f32>,
    pub shape: (usize, usize), 
}

/// Hardware capabilities report.
pub struct BackendInfo {
    pub name: String,
    pub max_sequence_length: usize,
    pub max_batch_size: usize,
}

pub trait ModelRunner: Send {
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

    /// Returns the name of the model.
    fn model_name(&self) -> String {
        "unknown".to_string()
    }
}
