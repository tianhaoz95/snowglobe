pub mod qwen;

use burn::module::Module;
use burn::tensor::{Int, Tensor, backend::Backend};

/// Key-Value cache for the transformer layers.
#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub key: Tensor<B, 4>,
    pub value: Tensor<B, 4>,
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

pub use qwen::{
    Qwen, QwenAttentionRecord, QwenBlockRecord, QwenConfig, QwenMLPRecord, QwenRecord,
};
