pub mod qwen;
pub mod qwen_pte;
pub mod runner;
pub mod llama_cpp;
pub mod speculative;

use burn::module::Module;
use burn::tensor::{Int, Tensor, backend::Backend};
use std::sync::Arc;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

/// Key-Value cache for the transformer layers.
#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub key: Tensor<B, 4>,
    pub value: Tensor<B, 4>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    Burn,
    ExecuTorch,
    LlamaCpp,
}

#[derive(Debug, Clone)]
pub struct InitConfig {
    pub vocab_shards: usize,
    pub max_gen_len: usize,
    pub use_executorch: bool, // Deprecated, keep for backwards compat for now or replace entirely
    pub backend: BackendType,
    pub speculate_tokens: usize, // 0 means disabled
}

pub enum EngineVariant {
    Burn(Box<dyn runner::ModelRunner>),
    ExecuTorch(Box<dyn runner::ModelRunner>),
    LlamaCpp(Box<dyn runner::ModelRunner>),
    Speculative(Box<dyn runner::ModelRunner>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub param_count: usize,
    pub model_size_bytes: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub runner: String,
    pub backend: String,
}

pub struct LoadedModel<B: Backend> {
    pub model: Arc<Mutex<EngineVariant>>,
    pub tokenizer: Tokenizer,
    pub config: QwenConfig,
    pub device: B::Device,
    pub init_config: InitConfig,
}

/// Trait defining the base API for LLM models.
pub trait Model<B: Backend>: Module<B> {
    type Config;

    /// Initialize the model from a configuration.
    fn init(config: &Self::Config, device: &B::Device) -> Self;

    /// Forward pass through the model.
    fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        cache: Option<Vec<Option<KVCache<B>>>>,
        offset: usize,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>);
}

pub use qwen::{Qwen, QwenAttentionRecord, QwenBlockRecord, QwenConfig, QwenMLPRecord, QwenRecord};
pub use qwen_pte::QwenPte;
