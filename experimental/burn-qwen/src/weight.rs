use burn::{
    module::Param,
    tensor::{backend::Backend, Shape, Tensor, TensorData},
};
use indicatif::{ProgressBar, ProgressStyle};
use half::{bf16, f16};
use safetensors::{
    tensor::{SafeTensors, TensorView},
    Dtype,
};
use std::collections::HashMap;

use crate::model::{QwenConfig, QwenRecord};

fn load_tensor_2d<B: Backend>(
    tensors: &mut HashMap<String, TensorView>,
    key: &str,
    device: &B::Device,
    transpose: bool,
) -> Param<Tensor<B, 2>> {
    let tensor_view = tensors
        .remove(key)
        .unwrap_or_else(|| panic!("Could not find tensor: {}", key));
    let shape = Shape::from(tensor_view.shape());
    let data_bytes = tensor_view.data();

    let data: TensorData = match tensor_view.dtype() {
        Dtype::F32 => TensorData::new(bytemuck::cast_slice::<u8, f32>(data_bytes).to_vec(), shape.clone()),
        Dtype::F16 => TensorData::new(bytemuck::cast_slice::<u8, f16>(data_bytes).to_vec(), shape.clone()),
        Dtype::BF16 => TensorData::new(bytemuck::cast_slice::<u8, bf16>(data_bytes).to_vec(), shape.clone()),
        dtype => panic!("Unsupported dtype {dtype:?}"),
    };

    let tensor: Tensor<B, 2> =
        Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), device).reshape(shape);

    if transpose {
        Param::from_tensor(tensor.transpose())
    } else {
        Param::from_tensor(tensor)
    }
}

fn load_tensor_1d<B: Backend>(
    tensors: &mut HashMap<String, TensorView>,
    key: &str,
    device: &B::Device,
) -> Param<Tensor<B, 1>> {
    let tensor_view = tensors
        .remove(key)
        .unwrap_or_else(|| panic!("Could not find tensor: {}", key));
    let shape = Shape::from(tensor_view.shape());
    let data_bytes = tensor_view.data();

    let data: TensorData = match tensor_view.dtype() {
        Dtype::F32 => TensorData::new(bytemuck::cast_slice::<u8, f32>(data_bytes).to_vec(), shape.clone()),
        Dtype::F16 => TensorData::new(bytemuck::cast_slice::<u8, f16>(data_bytes).to_vec(), shape.clone()),
        Dtype::BF16 => TensorData::new(bytemuck::cast_slice::<u8, bf16>(data_bytes).to_vec(), shape.clone()),
        dtype => panic!("Unsupported dtype {dtype:?}"),
    };

    let tensor: Tensor<B, 1> =
        Tensor::<B, 1>::from_data(data.convert::<B::FloatElem>(), device).reshape(shape);
    Param::from_tensor(tensor)
}

pub fn load_qwen_record<B: Backend>(
    config: &QwenConfig,
    safetensors: &SafeTensors,
    mut record: QwenRecord<B>,
    device: &B::Device,
) -> QwenRecord<B> {
    let mut tensors: HashMap<String, TensorView> = safetensors
        .tensors()
        .into_iter()
        .map(|(k, v)| (k.clone(), v))
        .collect();

    let pb = ProgressBar::new(safetensors.tensors().len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    record.embedding.weight = load_tensor_2d(&mut tensors, "model.embed_tokens.weight", device, false);
    pb.inc(1);

    for (i, layer) in record.layers.iter_mut().enumerate() {
        let layer_path = format!("model.layers.{}", i);

        layer.self_attn_norm.gamma = load_tensor_1d(
            &mut tensors,
            &format!("{}.input_layernorm.weight", layer_path),
            device,
        );
        pb.inc(1);
        layer.mlp_norm.gamma = load_tensor_1d(
            &mut tensors,
            &format!("{}.post_attention_layernorm.weight", layer_path),
            device,
        );
        pb.inc(1);

        layer.self_attn.q_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.self_attn.q_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
        if config.qkv_bias {
            layer.self_attn.q_proj.bias = Some(load_tensor_1d(
                &mut tensors,
                &format!("{}.self_attn.q_proj.bias", layer_path),
                device,
            ));
            pb.inc(1);
        }

        layer.self_attn.k_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.self_attn.k_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
        if config.qkv_bias {
            layer.self_attn.k_proj.bias = Some(load_tensor_1d(
                &mut tensors,
                &format!("{}.self_attn.k_proj.bias", layer_path),
                device,
            ));
            pb.inc(1);
        }

        layer.self_attn.v_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.self_attn.v_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
        if config.qkv_bias {
            layer.self_attn.v_proj.bias = Some(load_tensor_1d(
                &mut tensors,
                &format!("{}.self_attn.v_proj.bias", layer_path),
                device,
            ));
            pb.inc(1);
        }

        layer.self_attn.o_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.self_attn.o_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);

        layer.mlp.gate_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.mlp.gate_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
        layer.mlp.up_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.mlp.up_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
        layer.mlp.down_proj.weight = load_tensor_2d(
            &mut tensors,
            &format!("{}.mlp.down_proj.weight", layer_path),
            device,
            true,
        );
        pb.inc(1);
    }

    record.rms_norm.gamma = load_tensor_1d(&mut tensors, "model.norm.weight", device);
    pb.inc(1);

    pb.finish();
    println!("Model weights loaded.");

    record
}
