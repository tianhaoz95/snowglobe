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
use burn::tensor::Int;
use tokenizers::Tokenizer;


use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use safetensors::SafeTensors;
use std::fs;

fn main() {
    type Backend = NdArray<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    let config = QwenConfig::default();
    let device = <Backend as burn::tensor::backend::Backend>::Device::Cpu; // Access Device directly

    println!("Initializing Qwen model...");
    let mut model: Qwen<Backend> = config.init(&device);



    // --- Safetensors loading ---
    let api = Api::new().unwrap();
    let repo_id = "Qwen/Qwen2.5-0.5B-Instruct".to_string();
    let model_file_name = "model.safetensors".to_string();
    let tokenizer_file_name = "tokenizer.json".to_string();

    let repo = api.repo(Repo::with_revision(
        repo_id.clone(),
        RepoType::Model,
        "main".to_string(),
    ));

    println!("Downloading model...");
    let model_path = repo.get(&model_file_name).unwrap();

    println!("Model downloaded to: {:?}", model_path);

    println!("Loading safetensors...");
    let buffer = fs::read(&model_path).unwrap();
    let safetensors = SafeTensors::deserialize(&buffer).unwrap();



    println!("Loading model weights...");
    let record = model.clone().into_record();
    let model_with_weights = crate::weight::load_qwen_record(&config, &safetensors, record, &device);
    model = model.load_record(model_with_weights);

    println!("Starting inference...");

    // Download and load tokenizer
    println!("Downloading tokenizer...");
    let tokenizer_path = repo.get(&tokenizer_file_name).unwrap();
    println!("Tokenizer downloaded to: {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // 1. Create a dummy input tensor
    let device = &model.devices()[0]; // Get the device from the model
    let input: Tensor<Backend, 2, Int> = burn::tensor::Tensor::from_data([[1, 2, 3]], device);

    // 2. Run inference
    let output = model.forward(input);

    // 3. Convert output to token IDs
    let top_token_ids = output.argmax(2); // shape [batch_size, seq_len]
    let flat_token_ids: Vec<u32> = top_token_ids
        .into_data() // Convert to TensorData
        .into_vec() // Get the underlying vector (Vec<i64>)
        .into_iter()
        .flatten() // Flatten Vec<Vec<i64>> to Vec<i64>
        .map(|x: i64| x as u32) // Convert i64 to u32
        .collect();

    // 4. Decode token IDs to string
    let decoded_string = tokenizer.decode(&flat_token_ids, true).unwrap();

    // 5. Print the decoded string
    println!("Inference output: {}", decoded_string);

}
