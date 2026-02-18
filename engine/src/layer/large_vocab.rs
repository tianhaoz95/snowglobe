use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Int, Shape, Tensor, TensorData, backend::Backend},
};
use safetensors::{Dtype, tensor::TensorView};
use bytemuck;
use half::{f16, bf16};

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
            Self::Normal(e) => e.forward(input),
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
    pub part0: Embedding<B>,
    pub part1: Embedding<B>,
    pub part2: Embedding<B>,
    pub part3: Embedding<B>,
    pub part4: Embedding<B>,
    pub vocab_size: usize,
}

impl<B: Backend> LargeVocabEmbedding<B> {
    pub fn init(vocab_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            part0: EmbeddingConfig::new(CHUNK_SIZE, hidden_size).init(device),
            part1: EmbeddingConfig::new(CHUNK_SIZE, hidden_size).init(device),
            part2: EmbeddingConfig::new(CHUNK_SIZE, hidden_size).init(device),
            part3: EmbeddingConfig::new(CHUNK_SIZE, hidden_size).init(device),
            part4: EmbeddingConfig::new(vocab_size - 4 * CHUNK_SIZE, hidden_size).init(device),
            vocab_size,
        }
    }

    pub fn load_weights(
        record: &mut LargeVocabEmbeddingRecord<B>,
        view: &TensorView,
        hidden_size: usize,
        vocab_size: usize,
        device: &B::Device,
    ) {
        record.part0.weight = load_tensor_2d_range(view, 0..CHUNK_SIZE, hidden_size, device, false);
        record.part1.weight =
            load_tensor_2d_range(view, CHUNK_SIZE..2 * CHUNK_SIZE, hidden_size, device, false);
        record.part2.weight = load_tensor_2d_range(
            view,
            2 * CHUNK_SIZE..3 * CHUNK_SIZE,
            hidden_size,
            device,
            false,
        );
        record.part3.weight = load_tensor_2d_range(
            view,
            3 * CHUNK_SIZE..4 * CHUNK_SIZE,
            hidden_size,
            device,
            false,
        );
        record.part4.weight = load_tensor_2d_range(
            view,
            4 * CHUNK_SIZE..vocab_size,
            hidden_size,
            device,
            false,
        );
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mask0 = input.clone().lower_equal_elem(CHUNK_SIZE as i32 - 1);
        let mask1 = input.clone().greater_equal_elem(CHUNK_SIZE as i32)
            .bool_and(input.clone().lower_equal_elem(2 * CHUNK_SIZE as i32 - 1));
        let mask2 = input.clone().greater_equal_elem(2 * CHUNK_SIZE as i32)
            .bool_and(input.clone().lower_equal_elem(3 * CHUNK_SIZE as i32 - 1));
        let mask3 = input.clone().greater_equal_elem(3 * CHUNK_SIZE as i32)
            .bool_and(input.clone().lower_equal_elem(4 * CHUNK_SIZE as i32 - 1));
        let mask4 = input.clone().greater_equal_elem(4 * CHUNK_SIZE as i32);

        let x0 = self.part0.forward(input.clone().clamp(0, CHUNK_SIZE as i32 - 1));
        let x1 = self.part1.forward((input.clone() - CHUNK_SIZE as i32).clamp(0, CHUNK_SIZE as i32 - 1));
        let x2 = self.part2.forward((input.clone() - 2 * CHUNK_SIZE as i32).clamp(0, CHUNK_SIZE as i32 - 1));
        let x3 = self.part3.forward((input.clone() - 3 * CHUNK_SIZE as i32).clamp(0, CHUNK_SIZE as i32 - 1));
        let x4 = self.part4.forward((input.clone() - 4 * CHUNK_SIZE as i32).clamp(0, (self.vocab_size - 4 * CHUNK_SIZE) as i32 - 1));

        let mut x = x0.mask_fill(mask0.bool_not().unsqueeze_dim(2), 0.0);
        x = x + x1.mask_fill(mask1.bool_not().unsqueeze_dim(2), 0.0);
        x = x + x2.mask_fill(mask2.bool_not().unsqueeze_dim(2), 0.0);
        x = x + x3.mask_fill(mask3.bool_not().unsqueeze_dim(2), 0.0);
        x = x + x4.mask_fill(mask4.bool_not().unsqueeze_dim(2), 0.0);
        x
    }
}

/// A split linear layer for large vocabularies to stay within GPU binding limits.
#[derive(Debug, Module)]
pub struct LargeVocabLinear<B: Backend> {
    pub part0: Linear<B>,
    pub part1: Linear<B>,
    pub part2: Linear<B>,
    pub part3: Linear<B>,
    pub part4: Linear<B>,
}

impl<B: Backend> LargeVocabLinear<B> {
    pub fn init(hidden_size: usize, vocab_size: usize, device: &B::Device) -> Self {
        Self {
            part0: LinearConfig::new(hidden_size, CHUNK_SIZE).init(device),
            part1: LinearConfig::new(hidden_size, CHUNK_SIZE).init(device),
            part2: LinearConfig::new(hidden_size, CHUNK_SIZE).init(device),
            part3: LinearConfig::new(hidden_size, CHUNK_SIZE).init(device),
            part4: LinearConfig::new(hidden_size, vocab_size - 4 * CHUNK_SIZE).init(device),
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
        record.part0.weight = load_tensor_2d_range(view, 0..CHUNK_SIZE, hidden_size, device, transpose);
        record.part1.weight = load_tensor_2d_range(
            view,
            CHUNK_SIZE..2 * CHUNK_SIZE,
            hidden_size,
            device,
            transpose,
        );
        record.part2.weight = load_tensor_2d_range(
            view,
            2 * CHUNK_SIZE..3 * CHUNK_SIZE,
            hidden_size,
            device,
            transpose,
        );
        record.part3.weight = load_tensor_2d_range(
            view,
            3 * CHUNK_SIZE..4 * CHUNK_SIZE,
            hidden_size,
            device,
            transpose,
        );
        record.part4.weight = load_tensor_2d_range(
            view,
            4 * CHUNK_SIZE..vocab_size,
            hidden_size,
            device,
            transpose,
        );
    }

    pub fn tie_weights(
        record: &mut LargeVocabLinearRecord<B>,
        embedding_record: &LargeVocabEmbeddingRecord<B>,
    ) {
        record.part0.weight = Param::from_tensor(embedding_record.part0.weight.val().transpose());
        record.part1.weight = Param::from_tensor(embedding_record.part1.weight.val().transpose());
        record.part2.weight = Param::from_tensor(embedding_record.part2.weight.val().transpose());
        record.part3.weight = Param::from_tensor(embedding_record.part3.weight.val().transpose());
        record.part4.weight = Param::from_tensor(embedding_record.part4.weight.val().transpose());
    }

    pub fn from_embedding(embedding: &LargeVocabEmbedding<B>) -> Self {
        Self {
            part0: Linear {
                weight: Param::from_tensor(embedding.part0.weight.clone().val().transpose()),
                bias: None,
            },
            part1: Linear {
                weight: Param::from_tensor(embedding.part1.weight.clone().val().transpose()),
                bias: None,
            },
            part2: Linear {
                weight: Param::from_tensor(embedding.part2.weight.clone().val().transpose()),
                bias: None,
            },
            part3: Linear {
                weight: Param::from_tensor(embedding.part3.weight.clone().val().transpose()),
                bias: None,
            },
            part4: Linear {
                weight: Param::from_tensor(embedding.part4.weight.clone().val().transpose()),
                bias: None,
            },
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let out0 = self.part0.forward(input.clone());
        let out1 = self.part1.forward(input.clone());
        let out2 = self.part2.forward(input.clone());
        let out3 = self.part3.forward(input.clone());
        let out4 = self.part4.forward(input);

        Tensor::cat(vec![out0, out1, out2, out3, out4], 2)
    }
}
