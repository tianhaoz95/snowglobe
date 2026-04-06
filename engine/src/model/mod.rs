pub mod qwen;
pub mod runner;
#[cfg(feature = "llamacpp")]
pub mod llama_cpp;
pub mod speculative;
pub mod litert;

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
    LlamaCpp,
    LiteRT,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HardwareTarget {
    Auto,
    Cpu,
    Gpu,
    Npu,
}

#[derive(Debug, Clone)]
pub struct InitConfig {
    pub vocab_shards: usize,
    pub max_gen_len: usize,
    pub backend: BackendType,
    pub hardware: HardwareTarget,
    pub speculate_tokens: usize, // 0 means disabled
}

pub enum EngineVariant {
    Burn(Box<dyn runner::ModelRunner>),
    #[cfg(feature = "llamacpp")]
    LlamaCpp(Box<dyn runner::ModelRunner>),
    LiteRT(Box<dyn runner::ModelRunner>),
    Speculative(Box<dyn runner::ModelRunner>),
}

impl EngineVariant {
    pub fn backend_name(&self) -> String {
        match self {
            EngineVariant::Burn(m) => m.get_backend_info().name,
            #[cfg(feature = "llamacpp")]
            EngineVariant::LlamaCpp(m) => m.get_backend_info().name,
            EngineVariant::LiteRT(m) => m.get_backend_info().name,
            EngineVariant::Speculative(m) => m.get_backend_info().name,
        }
    }

    pub fn model_name(&self) -> String {
        match self {
            EngineVariant::Burn(m) => m.model_name(),
            #[cfg(feature = "llamacpp")]
            EngineVariant::LlamaCpp(m) => m.model_name(),
            EngineVariant::LiteRT(m) => m.model_name(),
            EngineVariant::Speculative(m) => m.model_name(),
        }
    }

    pub fn update_cache(&self, tokens: &[u32]) {
        match self {
            EngineVariant::Burn(m) => m.update_cache(tokens),
            #[cfg(feature = "llamacpp")]
            EngineVariant::LlamaCpp(m) => m.update_cache(tokens),
            EngineVariant::LiteRT(m) => m.update_cache(tokens),
            EngineVariant::Speculative(m) => m.update_cache(tokens),
        }
    }

    pub fn truncate_cache(&self, session: &mut runner::EngineSession, len: usize) -> Result<(), String> {
        match self {
            EngineVariant::Burn(m) => m.truncate_cache(session, len),
            #[cfg(feature = "llamacpp")]
            EngineVariant::LlamaCpp(m) => m.truncate_cache(session, len),
            EngineVariant::LiteRT(m) => m.truncate_cache(session, len),
            EngineVariant::Speculative(m) => m.truncate_cache(session, len),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub name: String,
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
