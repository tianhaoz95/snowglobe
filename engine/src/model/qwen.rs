use burn::{
    module::{Module, Param},
    nn::{EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Int, Tensor, backend::Backend},
};
// Closing brace for the `burn` import block
use serde::{Deserialize, Serialize};

use super::{KVCache, Model};
use crate::layer::large_vocab::{
    LargeVocabEmbedding, LargeVocabLinear, VocabEmbedding, VocabLinear,
};
use crate::rope::{apply_rotary_pos_emb, create_sin_cos_cache};

/// Configuration for the Qwen model.
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize, // For Grouped Query Attention (GQA)
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,   // For the feed-forward layer
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_true")]
    pub use_cache: bool,
    #[serde(alias = "tie_word_embeddings", default = "default_true")]
    pub tied_word_embeddings: bool,
    #[serde(alias = "attention_bias", default = "default_true")]
    pub qkv_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String, // "silu" for SwiGLU

    // General transformer settings
    #[serde(default)]
    pub dropout: f32,
    #[serde(default = "default_vocab_shards")]
    pub vocab_shards: usize,

    // Qwen3 / Advanced features
    pub model_type: Option<String>,
    pub head_dim: Option<usize>,
    pub use_qk_norm: Option<bool>,
}

fn default_vocab_size() -> usize { 151936 }
fn default_hidden_size() -> usize { 896 }
fn default_num_hidden_layers() -> usize { 24 }
fn default_num_attention_heads() -> usize { 14 }
fn default_num_key_value_heads() -> usize { 2 }
fn default_intermediate_size() -> usize { 4864 }
fn default_rope_theta() -> f64 { 1000000.0 }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_hidden_act() -> String { "silu".to_string() }

fn default_true() -> bool {
    true
}

fn default_vocab_shards() -> usize {
    1
}

impl QwenConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen<B> {
        let vocab_size = self.vocab_size;
        let hidden_size = self.hidden_size;
        let num_hidden_layers = self.num_hidden_layers;
        let num_attention_heads = self.num_attention_heads;
        let num_key_value_heads = self.num_key_value_heads;
        let intermediate_size = self.intermediate_size;
        let max_position_embeddings = self.max_position_embeddings;
        let rms_norm_eps = self.rms_norm_eps;
        let dropout = self.dropout;
        let qkv_bias = self.qkv_bias;
        let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);

        let use_qk_norm = self.use_qk_norm.unwrap_or_else(|| {
            if let Some(model_type) = &self.model_type {
                model_type == "qwen3"
            } else {
                false
            }
        });

        let embedding = if self.vocab_shards > 1 {
            VocabEmbedding::Sharded(LargeVocabEmbedding::init(
                vocab_size,
                hidden_size,
                self.vocab_shards,
                device,
            ))
        } else {
            VocabEmbedding::Normal(EmbeddingConfig::new(vocab_size, hidden_size).init(device))
        };

        let rms_norm = RmsNormConfig::new(hidden_size)
            .with_epsilon(rms_norm_eps)
            .init(device);

        let layers = (0..num_hidden_layers)
            .map(|_| {
                QwenBlockConfig::new(
                    hidden_size,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    intermediate_size,
                    max_position_embeddings,
                    rms_norm_eps,
                    dropout,
                    qkv_bias,
                    use_qk_norm,
                )
                .init(device)
            })
            .collect();

        let linear_output = if self.tied_word_embeddings {
            match &embedding {
                VocabEmbedding::Sharded(e) => {
                    VocabLinear::Sharded(LargeVocabLinear::from_embedding(e))
                }
                VocabEmbedding::Normal(e) => VocabLinear::Normal(Linear {
                    weight: Param::from_tensor(e.weight.clone().val().transpose()),
                    bias: None,
                }),
            }
        } else if self.vocab_shards > 1 {
            VocabLinear::Sharded(LargeVocabLinear::init(
                hidden_size,
                vocab_size,
                self.vocab_shards,
                device,
            ))
        } else {
            VocabLinear::Normal(LinearConfig::new(hidden_size, vocab_size).init(device))
        };

        Qwen {
            embedding,
            layers,
            rms_norm,
            linear_output,
        }
    }
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            use_cache: true,            // Typically true for inference
            tied_word_embeddings: true, // Check Qwen config for this
            qkv_bias: true,
            hidden_act: "silu".to_string(),
            dropout: 0.0,
            vocab_shards: 1,
            model_type: None,
            head_dim: None,
            use_qk_norm: None,
        }
    }
}

/// The Qwen model.
#[derive(Debug, Module)]
pub struct Qwen<B: Backend> {
    pub embedding: VocabEmbedding<B>,
    pub layers: Vec<QwenBlock<B>>,
    pub rms_norm: RmsNorm<B>,
    pub linear_output: VocabLinear<B>,
}

impl<B: Backend> Model<B> for Qwen<B> {
    type Config = QwenConfig;

    fn init(config: &Self::Config, device: &B::Device) -> Self {
        config.init(device)
    }

    fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        cache: Option<Vec<Option<KVCache<B>>>>,
        offset: usize,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let mut x = self.embedding.forward(input); // Output will be [batch_size, seq_len, hidden_size]

        let mut next_cache = Vec::new();
        let cache = cache.unwrap_or_else(|| vec![None; self.layers.len()]);

        for (layer, layer_cache) in self.layers.iter().zip(cache) {
            let (out, new_cache) = layer.forward(x, layer_cache, offset);
            x = out;
            next_cache.push(new_cache);
        }

        x = self.rms_norm.forward(x);

        let logits = self.linear_output.forward(x);

        (logits, next_cache)
    }
}

use crate::model::runner::{EngineSession, ModelRunner};
use burn::tensor::{Shape, TensorData};

impl<B: Backend> ModelRunner for Qwen<B> {
    fn load(_path: &std::path::Path, _config: &crate::InitConfig) -> Result<Box<Self>, String> {
        Err("load not implemented for Qwen directly".to_string())
    }

    fn forward(
        &self,
        session: &mut EngineSession,
    ) -> Result<Vec<f32>, String> {
        let device = self.rms_norm.gamma.val().device();
        let start = session.offset;
        let num_new = session.tokens.len() - start;

        if num_new == 0 {
            return Err("No new tokens".to_string());
        }

        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(
                session.tokens[start..]
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                Shape::new([1, num_new]),
            ),
            &device,
        );

        let cache = session.state.take().and_then(|s| {
            s.downcast::<Vec<KVCache<B>>>().ok().map(|b| *b)
        }).map(|c| c.into_iter().map(Some).collect());

        let (output, new_cache) = Model::forward(self, input_tensor, cache, session.offset);

        session.state = Some(Box::new(new_cache));
        session.offset += num_new;

        let [_, seq_len, vocab_size] = output.dims();
        let next_token_logits = output
            .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        next_token_logits.to_data().into_vec::<f32>().map_err(|e| format!("{:?}", e))
    }
}

// Configuration for a single Qwen block (transformer layer).
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenBlockConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub dropout: f32,
    pub qkv_bias: bool,
    pub use_qk_norm: bool,
}

impl QwenBlockConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        rms_norm_eps: f64,
        dropout: f32,
        qkv_bias: bool,
        use_qk_norm: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            max_position_embeddings,
            rms_norm_eps,
            dropout,
            qkv_bias,
            use_qk_norm,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenBlock<B> {
        let hidden_size = self.hidden_size;
        let num_attention_heads = self.num_attention_heads;
        let num_key_value_heads = self.num_key_value_heads;
        let head_dim = self.head_dim;
        let intermediate_size = self.intermediate_size;
        let max_position_embeddings = self.max_position_embeddings;
        let rms_norm_eps = self.rms_norm_eps;
        let dropout = self.dropout;
        let qkv_bias = self.qkv_bias;
        let use_qk_norm = self.use_qk_norm;

        let self_attn_norm = RmsNormConfig::new(hidden_size)
            .with_epsilon(rms_norm_eps)
            .init(device);
        let self_attn = QwenAttentionConfig::new(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            QwenConfig::default().rope_theta, // Use default for now
            max_position_embeddings,
            dropout,
            qkv_bias,
            use_qk_norm,
        )
        .init(device);

        let mlp_norm = RmsNormConfig::new(hidden_size)
            .with_epsilon(rms_norm_eps)
            .init(device);
        let mlp = QwenMLPConfig::new(hidden_size, intermediate_size, dropout).init(device);

        QwenBlock {
            self_attn_norm,
            self_attn,
            mlp_norm,
            mlp,
        }
    }
}

/// A single Qwen block (transformer layer).
#[derive(Debug, Module)]
pub struct QwenBlock<B: Backend> {
    self_attn_norm: RmsNorm<B>,
    self_attn: QwenAttention<B>,
    mlp_norm: RmsNorm<B>,
    mlp: QwenMLP<B>,
}

impl<B: Backend> QwenBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: Option<KVCache<B>>,
        offset: usize,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let hidden_states = self.self_attn_norm.forward(input.clone());
        let (self_attn_output, new_cache) = self.self_attn.forward(
            hidden_states.clone(),
            hidden_states.clone(),
            hidden_states,
            None,
            cache,
            offset,
        );

        let hidden_states = input + self_attn_output;

        let mlp_output = self
            .mlp
            .forward(self.mlp_norm.forward(hidden_states.clone()));
        (hidden_states + mlp_output, new_cache)
    }
}

/// Configuration for the Qwen attention mechanism.
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenAttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub dropout: f32,
    pub qkv_bias: bool,
    pub use_qk_norm: bool,
}

impl QwenAttentionConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        dropout: f32,
        qkv_bias: bool,
        use_qk_norm: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            max_position_embeddings,
            dropout,
            qkv_bias,
            use_qk_norm,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenAttention<B> {
        let hidden_size = self.hidden_size;
        let num_attention_heads = self.num_attention_heads;
        let num_key_value_heads = self.num_key_value_heads;
        let head_dim = self.head_dim;

        let q_proj = LinearConfig::new(hidden_size, num_attention_heads * head_dim)
            .with_bias(self.qkv_bias)
            .init(device);
        let k_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
            .with_bias(self.qkv_bias)
            .init(device);
        let v_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
            .with_bias(self.qkv_bias)
            .init(device);
        let o_proj = LinearConfig::new(num_attention_heads * head_dim, hidden_size)
            .with_bias(false) // Qwen doesn't use bias for output projection
            .init(device);

        let (q_norm, k_norm) = if self.use_qk_norm {
            (
                Some(RmsNormConfig::new(head_dim).init(device)),
                Some(RmsNormConfig::new(head_dim).init(device)),
            )
        } else {
            (None, None)
        };

        let (sin_cached, cos_cached) = create_sin_cos_cache(
            head_dim,
            self.max_position_embeddings,
            self.rope_theta,
            device,
        );

        QwenAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            dropout: self.dropout,
            sin_cached,
            cos_cached,
        }
    }
}

/// The Qwen attention mechanism.
#[derive(Debug, Module)]
pub struct QwenAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    dropout: f32,
    sin_cached: Tensor<B, 4>,
    cos_cached: Tensor<B, 4>,
}

impl<B: Backend> QwenAttention<B> {
    pub fn forward(
        &self,
        query: Tensor<B, 3>, // [batch_size, seq_len, hidden_size]
        key: Tensor<B, 3>,   // [batch_size, seq_len, hidden_size]
        value: Tensor<B, 3>, // [batch_size, seq_len, hidden_size]
        _mask: Option<Tensor<B, 3>>,
        cache: Option<KVCache<B>>,
        offset: usize,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let [batch_size, seq_len, _hidden_size] = query.dims();

        let q = self.q_proj.forward(query); // [batch_size, seq_len, num_attention_heads * head_dim]
        let k = self.k_proj.forward(key); // [batch_size, seq_len, num_key_value_heads * head_dim]
        let v = self.v_proj.forward(value); // [batch_size, seq_len, num_key_value_heads * head_dim]

        let mut q_reshaped = q
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_attention_heads, seq_len, head_dim]
        let mut k_reshaped = k
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_key_value_heads, seq_len, head_dim]
        let v_reshaped = v
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2); // [batch_size, num_key_value_heads, seq_len, head_dim]

        // Apply QK-Norm if present
        if let Some(q_norm) = &self.q_norm {
            q_reshaped = q_norm.forward(q_reshaped);
        }
        if let Some(k_norm) = &self.k_norm {
            k_reshaped = k_norm.forward(k_reshaped);
        }

        // Apply RoPE
        let (q_rotated, k_rotated) = apply_rotary_pos_emb(
            q_reshaped,
            k_reshaped,
            &self.sin_cached,
            &self.cos_cached,
            offset,
        );

        let (k_final, v_final) = if let Some(c) = cache {
            (
                Tensor::cat(vec![c.key, k_rotated], 2),
                Tensor::cat(vec![c.value, v_reshaped], 2),
            )
        } else {
            (k_rotated, v_reshaped)
        };

        let new_cache = KVCache {
            key: k_final.clone(),
            value: v_final.clone(),
        };

        // Grouped Query Attention (GQA)
        let [_batch_size, _num_heads, seq_len_k, _head_dim] = k_final.dims();

        let k_gqa = if self.num_key_value_heads < self.num_attention_heads {
            // Repeat the K heads
            let num_reps = self.num_attention_heads / self.num_key_value_heads;
            // k_rotated.repeat(&[1, num_reps, 1, 1])
            k_final
                .reshape([
                    batch_size,
                    self.num_key_value_heads,
                    1,
                    seq_len_k,
                    self.head_dim,
                ])
                .repeat(&[1, 1, num_reps, 1, 1])
                .reshape([
                    batch_size,
                    self.num_attention_heads,
                    seq_len_k,
                    self.head_dim,
                ])
        } else {
            k_final
        };
        let v_gqa = if self.num_key_value_heads < self.num_attention_heads {
            // Repeat the V heads
            let num_reps = self.num_attention_heads / self.num_key_value_heads;
            v_final
                .reshape([
                    batch_size,
                    self.num_key_value_heads,
                    1,
                    seq_len_k,
                    self.head_dim,
                ])
                .repeat(&[1, 1, num_reps, 1, 1])
                .reshape([
                    batch_size,
                    self.num_attention_heads,
                    seq_len_k,
                    self.head_dim,
                ])
        } else {
            v_final
        };

        // Scaled Dot-Product Attention
        let scores = q_rotated.matmul(k_gqa.swap_dims(2, 3)); // [batch_size, num_attention_heads, seq_len, seq_len_k]
        let mut scores = scores.div_scalar(f64::sqrt(self.head_dim as f64));

        // Causal mask
        let [_batch_size, _num_heads, seq_len_q, seq_len_k] = scores.dims();
        if seq_len_q > 1 {
            let causal_mask = Tensor::<B, 2>::ones([seq_len_q, seq_len_k], &scores.device())
                .tril(offset as i64)
                .bool()
                .reshape([1, 1, seq_len_q, seq_len_k]);
            scores = scores.mask_fill(causal_mask.equal_elem(false), f64::NEG_INFINITY);
        }

        // Apply mask (if provided)
        // if let Some(mask) = _mask {
        //     scores = scores.mask_fill(mask, f64::NEG_INFINITY);
        // }

        let attn_weights =
            burn::tensor::activation::softmax(scores.clone(), scores.dims().len() - 1);
        let attn_output = attn_weights.matmul(v_gqa); // [batch_size, num_attention_heads, seq_len, head_dim]

        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]); // [batch_size, seq_len, hidden_size]

        (self.o_proj.forward(attn_output), new_cache)
    }
}

/// Configuration for the Qwen MLP (feed-forward network).
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenMLPConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub dropout: f32,
}

impl QwenMLPConfig {
    pub fn new(hidden_size: usize, intermediate_size: usize, dropout: f32) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            dropout,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenMLP<B> {
        let gate_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(self.intermediate_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        QwenMLP {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

/// The Qwen MLP (feed-forward network).
#[derive(Debug, Module)]
pub struct QwenMLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> QwenMLP<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_proj.forward(input);

        let activated_gate = burn::tensor::activation::silu(gate);
        let intermediate = activated_gate.mul(up);
        self.down_proj.forward(intermediate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_qwen3_config_init() {
        let device = <NdArray as Backend>::Device::default();
        let config = QwenConfig {
            vocab_size: 151936,
            hidden_size: 1024,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 3072,
            rope_theta: 1000000.0,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            use_cache: true,
            tied_word_embeddings: true,
            qkv_bias: false,
            hidden_act: "silu".to_string(),
            dropout: 0.0,
            vocab_shards: 1,
            model_type: Some("qwen3".to_string()),
            head_dim: Some(128),
            use_qk_norm: Some(true),
        };

        let model: Qwen<NdArray> = config.init(&device);
        assert_eq!(model.layers.len(), 2);

        // Check attention params
        let layer0 = &model.layers[0];
        assert_eq!(layer0.self_attn.head_dim, 128);
        assert!(layer0.self_attn.q_norm.is_some());
        assert!(layer0.self_attn.k_norm.is_some());
    }
}
