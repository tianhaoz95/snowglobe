use burn::{
    module::{Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        RmsNorm, RmsNormConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
}; // Closing brace for the `burn` import block
use serde::{Deserialize, Serialize};

use crate::rope::{apply_rotary_pos_emb, create_sin_cos_cache};

/// Configuration for the Qwen model.
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // For Grouped Query Attention (GQA)
    pub intermediate_size: usize, // For the feed-forward layer
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tied_word_embeddings: bool,
    pub qkv_bias: bool,
    pub hidden_act: String, // "silu" for SwiGLU

    // General transformer settings
    pub dropout: f32,
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

        let embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
        let rms_norm = RmsNormConfig::new(hidden_size).with_epsilon(rms_norm_eps).init(device);

        let layers = (0..num_hidden_layers)
            .map(|_| {
                QwenBlockConfig::new(
                    hidden_size,
                    num_attention_heads,
                    num_key_value_heads,
                    intermediate_size,
                    max_position_embeddings,
                    rms_norm_eps,
                    dropout,
                    qkv_bias,
                )
                .init(device)
            })
            .collect();

        let linear_output = if self.tied_word_embeddings {
            Linear {
                weight: Param::from_tensor(embedding.weight.clone().val().transpose()),
                bias: None,
            }
        } else {
            LinearConfig::new(hidden_size, vocab_size).init(device)
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
            use_cache: true,      // Typically true for inference
            tied_word_embeddings: true, // Check Qwen config for this
            qkv_bias: true,
            hidden_act: "silu".to_string(),
            dropout: 0.0,
        }
    }
}

/// The Qwen model.
#[derive(Debug, Module)]
pub struct Qwen<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<QwenBlock<B>>,
    pub rms_norm: RmsNorm<B>,
    pub linear_output: Linear<B>,
}

impl<B: Backend> Qwen<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input); // Output will be [batch_size, seq_len, hidden_size]

        // QwenBlock expects 3D tensor
        for layer in &self.layers {
            x = layer.forward(x);
        }

        self.linear_output.forward(self.rms_norm.forward(x))
    }
}

/// Configuration for a single Qwen block (transformer layer).
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenBlockConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub dropout: f32,
    pub qkv_bias: bool,
}

impl QwenBlockConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        rms_norm_eps: f64,
        dropout: f32,
        qkv_bias: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            max_position_embeddings,
            rms_norm_eps,
            dropout,
            qkv_bias,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenBlock<B> {
        let hidden_size = self.hidden_size;
        let num_attention_heads = self.num_attention_heads;
        let num_key_value_heads = self.num_key_value_heads;
        let intermediate_size = self.intermediate_size;
        let max_position_embeddings = self.max_position_embeddings;
        let rms_norm_eps = self.rms_norm_eps;
        let dropout = self.dropout;
        let qkv_bias = self.qkv_bias;

        let self_attn_norm = RmsNormConfig::new(hidden_size).with_epsilon(rms_norm_eps).init(device);
        let self_attn = QwenAttentionConfig::new(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            QwenConfig::default().rope_theta, // Use default for now
            max_position_embeddings,
            dropout,
            qkv_bias,
        )
        .init(device);

        let mlp_norm = RmsNormConfig::new(hidden_size).with_epsilon(rms_norm_eps).init(device);
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
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.self_attn_norm.forward(input.clone());
        let self_attn_output = self
            .self_attn
            .forward(hidden_states.clone(), hidden_states.clone(), hidden_states, None);

        let hidden_states = input + self_attn_output;

        let mlp_output = self.mlp.forward(self.mlp_norm.forward(hidden_states.clone()));
        hidden_states + mlp_output
    }
}

/// Configuration for the Qwen attention mechanism.
#[derive(Debug, Clone, Serialize, Deserialize, Module)]
pub struct QwenAttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub dropout: f32,
    pub qkv_bias: bool,
}

impl QwenAttentionConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        dropout: f32,
        qkv_bias: bool,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            max_position_embeddings,
            dropout,
            qkv_bias,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenAttention<B> {
        let hidden_size = self.hidden_size;
        let num_attention_heads = self.num_attention_heads;
        let num_key_value_heads = self.num_key_value_heads;
        let head_dim = hidden_size / num_attention_heads;

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
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = query.dims();

        let q = self.q_proj.forward(query); // [batch_size, seq_len, num_attention_heads * head_dim]
        let k = self.k_proj.forward(key);   // [batch_size, seq_len, num_key_value_heads * head_dim]
        let v = self.v_proj.forward(value); // [batch_size, seq_len, num_key_value_heads * head_dim]

        let q_reshaped = q
            .reshape([
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_attention_heads, seq_len, head_dim]
        let k_reshaped = k
            .reshape([
                batch_size,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_key_value_heads, seq_len, head_dim]
        let v_reshaped = v
            .reshape([
                batch_size,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_key_value_heads, seq_len, head_dim]

        // Apply RoPE
        let (q_rotated, k_rotated) = apply_rotary_pos_emb(
            q_reshaped,
            k_reshaped,
            &self.sin_cached,
            &self.cos_cached,
            seq_len,
        );

        // Grouped Query Attention (GQA)
        let k_gqa = if self.num_key_value_heads < self.num_attention_heads {
            // Repeat the K heads
            let num_reps = self.num_attention_heads / self.num_key_value_heads;
            // k_rotated.repeat(&[1, num_reps, 1, 1])
            k_rotated.reshape([batch_size, self.num_key_value_heads, 1, seq_len, self.head_dim])
                .repeat(&[1, 1, num_reps, 1, 1])
                .reshape([batch_size, self.num_attention_heads, seq_len, self.head_dim])
        } else {
            k_rotated
        };
        let v_gqa = if self.num_key_value_heads < self.num_attention_heads {
            // Repeat the V heads
            let num_reps = self.num_attention_heads / self.num_key_value_heads;
            v_reshaped.repeat(&[1, num_reps, 1, 1])
        } else {
            v_reshaped
        };

        // Scaled Dot-Product Attention
        let scores = q_rotated.matmul(k_gqa.swap_dims(2, 3)); // [batch_size, num_attention_heads, seq_len, seq_len]
        let mut scores = scores.div_scalar(f64::sqrt(self.head_dim as f64));

        // Causal mask
        let [_batch_size, _num_heads, seq_len_q, seq_len_k] = scores.dims();
        let causal_mask = Tensor::<B, 2>::ones([seq_len_q, seq_len_k], &scores.device())
            .tril(0)
            .bool()
            .reshape([1, 1, seq_len_q, seq_len_k]);
        scores = scores.mask_fill(causal_mask.equal_elem(false), f64::NEG_INFINITY);


        // Apply mask (if provided)
        // if let Some(mask) = _mask {
        //     scores = scores.mask_fill(mask, f64::NEG_INFINITY);
        // }

        let attn_weights = burn::tensor::activation::softmax(
            scores.clone(), 
            scores.dims().len() - 1
        );
        let attn_output = attn_weights.matmul(v_gqa); // [batch_size, num_attention_heads, seq_len, head_dim]

        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]); // [batch_size, seq_len, hidden_size]

        self.o_proj.forward(attn_output)
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
        let gate_proj = LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let up_proj = LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let down_proj = LinearConfig::new(self.intermediate_size, self.hidden_size).init(device);

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