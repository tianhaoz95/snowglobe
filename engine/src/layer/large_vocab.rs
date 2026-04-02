use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Int, Shape, Tensor, TensorData, backend::Backend},
};
use bytemuck;
use half::{bf16, f16};
use safetensors::{Dtype, tensor::TensorView};

pub const CHUNK_SIZE: usize = 32000;

#[derive(Debug, Module)]
pub enum VocabEmbedding<B: Backend> {
    Normal(Embedding<B>),
    Sharded(LargeVocabEmbedding<B>),
}

impl<B: Backend> VocabEmbedding<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        match self {
            Self::Normal(e) => e.forward(input),
            Self::Sharded(e) => e.forward(input),
        }
    }
}

#[derive(Debug, Module)]
pub enum VocabLinear<B: Backend> {
    Normal(Linear<B>),
    Sharded(LargeVocabLinear<B>),
}

impl<B: Backend> VocabLinear<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Normal(e) => input.matmul(e.weight.val().transpose().unsqueeze()),
            Self::Sharded(e) => e.forward(input),
        }
    }
}

fn load_tensor_2d_range<B: Backend>(
    tensor_view: &TensorView,
    range: std::ops::Range<usize>,
    hidden_size: usize,
    device: &B::Device,
    transpose: bool,
) -> Param<Tensor<B, 2>> {
    let data_bytes = tensor_view.data();
    let dtype = tensor_view.dtype();
    let element_size = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => panic!("Unsupported dtype"),
    };

    let start_byte = range.start * hidden_size * element_size;
    let end_byte = range.end * hidden_size * element_size;
    let slice_bytes = &data_bytes[start_byte..end_byte];
    let slice_shape = Shape::new([range.end - range.start, hidden_size]);

    let data: TensorData = match dtype {
        Dtype::F32 => TensorData::new(
            bytemuck::cast_slice::<u8, f32>(slice_bytes).to_vec(),
            slice_shape,
        ),
        Dtype::F16 => TensorData::new(
            bytemuck::cast_slice::<u8, f16>(slice_bytes).to_vec(),
            slice_shape,
        ),
        Dtype::BF16 => TensorData::new(
            bytemuck::cast_slice::<u8, bf16>(slice_bytes).to_vec(),
            slice_shape,
        ),
        _ => unreachable!(),
    };

    let tensor = Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), device);
    if transpose {
        Param::from_tensor(tensor.transpose())
    } else {
        Param::from_tensor(tensor)
    }
}

/// A split embedding layer for large vocabularies to stay within GPU binding limits.
#[derive(Debug, Module)]
pub struct LargeVocabEmbedding<B: Backend> {
    pub parts: Vec<Embedding<B>>,
    pub vocab_size: usize,
    pub chunk_size: usize,
}

impl<B: Backend> LargeVocabEmbedding<B> {
    pub fn init(
        vocab_size: usize,
        hidden_size: usize,
        num_chunks: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_size = (vocab_size + num_chunks - 1) / num_chunks;
        let mut parts = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(vocab_size);
            parts.push(EmbeddingConfig::new(end - start, hidden_size).init(device));
        }
        Self {
            parts,
            vocab_size,
            chunk_size,
        }
    }

    pub fn load_weights(
        record: &mut LargeVocabEmbeddingRecord<B>,
        view: &TensorView,
        hidden_size: usize,
        vocab_size: usize,
        device: &B::Device,
    ) {
        let num_chunks = record.parts.len();
        let chunk_size = (vocab_size + num_chunks - 1) / num_chunks;
        for (i, part_record) in record.parts.iter_mut().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(vocab_size);
            part_record.weight = load_tensor_2d_range(view, start..end, hidden_size, device, false);
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x: Option<Tensor<B, 3>> = None;

        for (i, part) in self.parts.iter().enumerate() {
            let start = (i * self.chunk_size) as i32;
            let end = ((i + 1) * self.chunk_size).min(self.vocab_size) as i32;
            let current_chunk_size = end - start;

            let mask = input
                .clone()
                .greater_equal_elem(start)
                .bool_and(input.clone().lower_equal_elem(end - 1));

            let clamped_input = (input.clone() - start).clamp(0, current_chunk_size - 1);
            let out = part.forward(clamped_input);
            let masked_out = out.mask_fill(mask.bool_not().unsqueeze_dim(2), 0.0);

            if let Some(prev_x) = x {
                x = Some(prev_x + masked_out);
            } else {
                x = Some(masked_out);
            }
        }
        x.expect("At least one chunk expected")
    }
}

/// A split linear layer for large vocabularies to stay within GPU binding limits.
#[derive(Debug, Module)]
pub struct LargeVocabLinear<B: Backend> {
    pub parts: Vec<Linear<B>>,
    pub vocab_size: usize,
    pub chunk_size: usize,
}

impl<B: Backend> LargeVocabLinear<B> {
    pub fn init(
        hidden_size: usize,
        vocab_size: usize,
        num_chunks: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_size = (vocab_size + num_chunks - 1) / num_chunks;
        let mut parts = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(vocab_size);
            parts.push(LinearConfig::new(hidden_size, end - start).init(device));
        }
        Self {
            parts,
            vocab_size,
            chunk_size,
        }
    }

    pub fn load_weights(
        record: &mut LargeVocabLinearRecord<B>,
        view: &TensorView,
        hidden_size: usize,
        vocab_size: usize,
        device: &B::Device,
        transpose: bool,
    ) {
        let num_chunks = record.parts.len();
        let chunk_size = (vocab_size + num_chunks - 1) / num_chunks;
        for (i, part_record) in record.parts.iter_mut().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(vocab_size);
            part_record.weight =
                load_tensor_2d_range(view, start..end, hidden_size, device, transpose);
        }
    }

    pub fn tie_weights(
        record: &mut LargeVocabLinearRecord<B>,
        embedding_record: &LargeVocabEmbeddingRecord<B>,
    ) {
        for (part_record, emb_part_record) in
            record.parts.iter_mut().zip(embedding_record.parts.iter())
        {
            part_record.weight = Param::from_tensor(emb_part_record.weight.val());
        }
    }

    pub fn from_embedding(embedding: &LargeVocabEmbedding<B>) -> Self {
        let parts = embedding
            .parts
            .iter()
            .map(|emb| Linear {
                weight: Param::from_tensor(emb.weight.clone().val()),
                bias: None,
            })
            .collect();
        Self {
            parts,
            vocab_size: embedding.vocab_size,
            chunk_size: embedding.chunk_size,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let outs: Vec<_> = self
            .parts
            .iter()
            .map(|part| input.clone().matmul(part.weight.val().transpose().unsqueeze()))
            .collect();
        Tensor::cat(outs, 2)
    }
}
