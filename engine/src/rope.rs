use burn_tensor::{Int, Shape, Tensor, TensorData, backend::Backend};

/// Applies rotary positional embeddings to query and key tensors.
pub fn apply_rotary_pos_emb<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    sin_cached: &Tensor<B, 4>,
    cos_cached: &Tensor<B, 4>,
    offset: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_batch_size, _num_heads, seq_len, head_dim] = query.dims();

    // 1. Implementation of "Rotate Half"
    let rotate_half = |x: Tensor<B, 4>| {
        let [b, h, s, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
        let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
        Tensor::cat(vec![x2.neg(), x1], 3)
    };

    let q_rotated = rotate_half(query.clone());
    let k_rotated = rotate_half(key.clone());

    // 2. Slice cache to current sequence length with offset
    let cos = cos_cached
        .clone()
        .slice([0..1, 0..1, offset..offset + seq_len, 0..head_dim]);
    let sin = sin_cached
        .clone()
        .slice([0..1, 0..1, offset..offset + seq_len, 0..head_dim]);

    // 3. formula: (x * cos) + (rotate(x) * sin)
    let q_out = query.mul(cos.clone()).add(q_rotated.mul(sin.clone()));
    let k_out = key.mul(cos).add(k_rotated.mul(sin));

    (q_out, k_out)
}

/// Creates cached sinusoidal and cosinusoidal values for Rotary Positional Embeddings (RoPE).
pub fn create_sin_cos_cache<B: Backend>(
    head_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // 1. Generate the inverse frequencies for HALF the head dimension
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (rope_theta.powf((2 * i) as f64 / head_dim as f64) as f32))
        .collect();

    // Convert to tensor [1, half_dim]
    let inv_freq_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(inv_freq, Shape::new([1, half_dim])), device);

    // 2. Create the position indices [max_seq, 1]
    let t = Tensor::<B, 1, Int>::arange(0..max_position_embeddings as i64, device)
        .float() // Convert i64 -> f32
        .reshape([max_position_embeddings, 1]);

    // 3. Calculate frequencies via outer product [max_seq, half_dim]
    let freqs = t.matmul(inv_freq_tensor);

    // 4. Concatenate frequencies with themselves [max_seq, head_dim]
    // This matches the Rotate Half strategy where both halves share the same frequencies.
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);

    // 5. Reshape to [1, 1, max_seq, head_dim] for broadcasting during attention
    let cos = emb
        .clone()
        .cos()
        .reshape([1, 1, max_position_embeddings, head_dim]);
    let sin = emb.sin().reshape([1, 1, max_position_embeddings, head_dim]);

    (sin, cos)
}
