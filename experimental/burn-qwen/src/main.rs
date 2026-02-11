mod model;
mod rope;
mod weight; // Add this line to import the weight module

use crate::model::QwenConfig;
use burn_autodiff::Autodiff;
use burn::backend::NdArray;
use burn::tensor::backend::Backend; // Import the Backend trait
use crate::model::Qwen; // Import Qwen
use crate::weight::load_qwen_record; // Import load_qwen_record
use burn::prelude::*;

use hf_hub::api::sync::Api;
use safetensors::SafeTensors;
use std::fs;

fn main() {
    type Backend = NdArray<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let config = QwenConfig::default();
    let device = <Backend as burn::tensor::backend::Backend>::Device::Cpu; // Access Device directly

    let mut model: Qwen<Backend> = config.init(&device);

    println!("Qwen model initialized: {:?}", model);

    // --- Safetensors loading ---
    let api = Api::new().unwrap();
    let repo_id = "Qwen/Qwen2.5-0.5B-Instruct".to_string();
    let model_file_name = "model.safetensors".to_string();

    let model_path = api.model(repo_id).get(&model_file_name).unwrap();

    println!("Model downloaded to: {:?}", model_path);

    let buffer = fs::read(&model_path).unwrap();
    let safetensors = SafeTensors::deserialize(&buffer).unwrap();

    let safetensors_keys: Vec<String> = safetensors.tensors().iter().map(|(key, _)| key.clone()).collect();
    println!("Safetensors loaded with keys: {:?}", safetensors_keys);

    let record = model.clone().into_record();
    let model_with_weights = crate::weight::load_qwen_record(&config, &safetensors, record, &device);
    model = model.load_record(model_with_weights);
}
