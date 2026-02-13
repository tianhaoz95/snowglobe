#![recursion_limit = "256"]

mod model;
mod rope;
mod weight; // Add this line to import the weight module

use crate::model::QwenConfig;
use burn::backend::Wgpu;
use crate::model::Qwen;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use tokenizers::Tokenizer;
use burn::backend::wgpu::WgpuDevice;


use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use safetensors::SafeTensors;
use std::fs;
use std::io::{self, Write};

fn main() {
    type Backend = Wgpu<f32, i32>;

    let config = QwenConfig::default();
    let device = WgpuDevice::DefaultDevice;

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

    let im_start_id = tokenizer.token_to_id("<|im_start|>").expect("Missing <|im_start|>");
    let im_end_id = tokenizer.token_to_id("<|im_end|>").expect("Missing <|im_end|>");
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n

    let mut token_ids = Vec::new();
    let system_text = "You are a helpful assistant.";
    let system_tokens = tokenizer.encode(system_text, false).unwrap().get_ids().to_vec();
    token_ids.push(im_start_id);
    token_ids.extend(tokenizer.encode("system", false).unwrap().get_ids());
    token_ids.push(newline_id);
    token_ids.extend(system_tokens);
    token_ids.push(im_end_id);
    token_ids.push(newline_id);

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        let user_tokens = tokenizer.encode(user_input, false).unwrap().get_ids().to_vec();

        token_ids.push(im_start_id);
        token_ids.extend(tokenizer.encode("user", false).unwrap().get_ids());
        token_ids.push(newline_id);
        token_ids.extend(user_tokens);
        token_ids.push(im_end_id);
        token_ids.push(newline_id);
        token_ids.push(im_start_id);
        token_ids.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
        token_ids.push(newline_id);

        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut assistant_response = Vec::new();
        for _ in 0..1024 { // Generate up to 1024 tokens
            let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
                TensorData::new(
                    token_ids.clone().into_iter().map(|x| x as i32).collect(),
                    Shape::new([1, token_ids.len()])
                ),
                &device
            );

            let output = model.forward(input_tensor);

            let next_token_logits = output.slice([0..1, (token_ids.len() - 1)..(token_ids.len()), 0..config.vocab_size]).reshape([1, config.vocab_size]);
            let next_token_id = next_token_logits.argmax(1).into_data().into_vec::<i32>().unwrap()[0] as u32;

            if next_token_id == im_end_id {
                break;
            }

            token_ids.push(next_token_id);
            assistant_response.push(next_token_id);

            let decoded = tokenizer.decode(&[next_token_id], true).unwrap();
            print!("{}", decoded);
            io::stdout().flush().unwrap();
        }
        println!();
        token_ids.push(im_end_id);
        token_ids.push(newline_id);
    }
}
