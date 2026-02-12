mod model;
mod rope;
mod weight; // Add this line to import the weight module

use crate::model::QwenConfig;
use burn::backend::NdArray;
use crate::model::Qwen;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use tokenizers::Tokenizer;


use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use safetensors::SafeTensors;
use std::fs;
use std::io::{self, Write};

fn main() {
    type Backend = NdArray<f32>;

    let config = QwenConfig::default();
    let device = <Backend as burn::tensor::backend::Backend>::Device::Cpu;

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

    // 1. Create a meaningful input tensor
    let device = &model.devices()[0]; // Get the device from the model
    let input_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    println!("Encoding input: '{}'", input_text);
    let encoding = tokenizer.encode(input_text, true).unwrap();
    let mut token_ids = encoding.get_ids().to_vec(); // Vec<u32>

    println!("Generating response...");

    for _ in 0..16 { // Max generation length
        let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(token_ids.clone().into_iter().map(|x| x as i64).collect(), Shape::new([1, token_ids.len()])),
            device
        );

        let output = model.forward(input_tensor);

        // Get logits for the last token
        let next_token_logits = output.slice([0..1, (token_ids.len() - 1)..(token_ids.len()), 0..config.vocab_size]).reshape([1, config.vocab_size]); // shape [1, vocab_size]
        let next_token_id = next_token_logits.argmax(1).into_data().into_vec::<i64>().unwrap()[0] as u32;

        if next_token_id == tokenizer.token_to_id("<|im_end|>").unwrap() {
            break;
        }

        token_ids.push(next_token_id);

        let decoded = tokenizer.decode(&[next_token_id], true).unwrap();
        print!("{}", decoded);
        io::stdout().flush().unwrap(); // To print token by token
    }

    println!();

}
