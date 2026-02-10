use burn_tensor::{backend::Backend, ops::*, Shape, Tensor, TensorData, Int, Element};

/// Applies rotary positional embeddings to query and key tensors.
pub fn apply_rotary_pos_emb<B: Backend>(
    query: Tensor<B, 4>, // [batch_size, num_heads, seq_len, head_dim]
    key: Tensor<B, 4>,   // [batch_size, num_heads, seq_len, head_dim]
    sin_cached: &Tensor<B, 4>,
    cos_cached: &Tensor<B, 4>,
    seq_len: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch_size, num_heads, _, head_dim] = query.dims();

    let query_rot = query.clone().reshape([batch_size, num_heads, seq_len, head_dim / 2, 2]);
    let query_left: Tensor<B, 4> = query_rot.clone().slice([0..batch_size, 0..num_heads, 0..seq_len, 0..head_dim/2, 0..1]).squeeze_dim(4);
    let query_right: Tensor<B, 4> = query_rot.slice([0..batch_size, 0..num_heads, 0..seq_len, 0..head_dim/2, 1..2]).squeeze_dim(4);

    let query_rotated: Tensor<B, 4> = Tensor::cat(vec![query_right.neg(), query_left], 4);
    let query_rotated: Tensor<B, 4> = query_rotated.reshape([batch_size, num_heads, seq_len, head_dim]);

    let key_rot = key.clone().reshape([batch_size, num_heads, seq_len, head_dim / 2, 2]);
    let key_left: Tensor<B, 4> = key_rot.clone().slice([0..batch_size, 0..num_heads, 0..seq_len, 0..head_dim/2, 0..1]).squeeze_dim(4);
    let key_right: Tensor<B, 4> = key_rot.slice([0..batch_size, 0..num_heads, 0..seq_len, 0..head_dim/2, 1..2]).squeeze_dim(4);

    let key_rotated: Tensor<B, 4> = Tensor::cat(vec![key_right.neg(), key_left], 4);
    let key_rotated: Tensor<B, 4> = key_rotated.reshape([batch_size, num_heads, seq_len, head_dim]);

    let cos = cos_cached.clone().slice([0..1, 0..1, 0..seq_len, 0..head_dim]).repeat(&query.dims());
    let sin = sin_cached.clone().slice([0..1, 0..1, 0..seq_len, 0..head_dim]).repeat(&query.dims());

    let q_out = query.mul(cos.clone()).add(query_rotated.mul(sin.clone()));
    let k_out = key.mul(cos).add(key_rotated.mul(sin));

    (q_out, k_out)
}

/// Creates cached sinusoidal and cosinusoidal values for Rotary Positional Embeddings.
pub fn create_sin_cos_cache<B: Backend>(
    dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / dim as f64) as f32)
        .collect();

    let inv_freq_expanded: Vec<f32> = inv_freq.into_iter().flat_map(|f| [f, f]).collect();
    let inv_freq_data = TensorData::new(inv_freq_expanded, Shape::new([1, dim]));
    let inv_freq = Tensor::<B, 2>::from_data(inv_freq_data, device)
        .reshape([1, dim]); // [1, dim]

    let t = Tensor::arange(0..(max_position_embeddings as i64), device)
        .float() // Convert to float type
        .reshape([max_position_embeddings, 1]); // [seq_len, 1]

    let freqs = t.matmul(inv_freq); // [seq_len, dim]

    let cos = freqs.clone().cos().reshape([1, 1, max_position_embeddings, dim]);
    let sin = freqs.sin().reshape([1, 1, max_position_embeddings, dim]);

    (sin, cos)
}