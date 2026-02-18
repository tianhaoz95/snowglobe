#![recursion_limit = "256"]

pub mod model;
pub mod rope;
pub mod weight;

use crate::model::{Qwen, QwenConfig};
use crate::weight::load_qwen_record;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use dashmap::DashMap;
use futures_util::StreamExt;
use parking_lot::Mutex;
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;
use once_cell::sync::OnceCell;

#[cfg(feature = "high_perf")]
pub mod backend_setup {
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use once_cell::sync::OnceCell;

    // GPU Types: Wgpu takes <Float, Int>
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;
    
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub type GraphicsApi = burn_wgpu::graphics::Metal;
    #[cfg(target_os = "android")]
    pub type GraphicsApi = burn_wgpu::graphics::Vulkan;
    #[cfg(not(any(target_os = "ios", target_os = "macos", target_os = "android")))]
    pub type GraphicsApi = burn_wgpu::graphics::AutoGraphicsApi;

    pub static GLOBAL_DEVICE: OnceCell<Device> = OnceCell::new();
}

#[cfg(not(feature = "high_perf"))]
pub mod backend_setup {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use once_cell::sync::OnceCell;
    
    // CPU Types: NdArray in 0.20.1 usually only takes <Float>
    pub type Backend = NdArray<f32, i32>;
    pub type Device = NdArrayDevice;

    pub static GLOBAL_DEVICE: OnceCell<Device> = OnceCell::new();
}

pub use backend_setup::Backend;
pub use backend_setup::Device;
pub use backend_setup::GLOBAL_DEVICE;

static GLOBAL_MODEL: OnceCell<Mutex<Qwen<Backend>>> = OnceCell::new();
static GLOBAL_TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();
static GLOBAL_CONFIG: OnceCell<QwenConfig> = OnceCell::new();
static SESSIONS: OnceCell<DashMap<String, Vec<u32>>> = OnceCell::new();

pub fn check_backend() -> String {
    #[cfg(feature = "high_perf")]
    return "🚀 USING GPU (WGPU/VULKAN)".to_string();
    
    #[cfg(not(feature = "high_perf"))]
    return "💻 USING CPU (NDARRAY)".to_string();
}

pub async fn init(cache_dir: String) -> String {
    let config = QwenConfig::default();

    // 1. Initialize Device based on Feature
    #[cfg(feature = "high_perf")]
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    #[cfg(not(feature = "high_perf"))]
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // 2. Setup GPU only if High Perf is enabled
    #[cfg(feature = "high_perf")]
    {
        let _setup = burn_wgpu::init_setup_async::<backend_setup::GraphicsApi>(
            &device,
            burn_wgpu::RuntimeOptions::default(),
        ).await;
    }

    let mut model: Qwen<Backend> = config.init(&device);

    let model_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors";
    let tokenizer_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";

    let model_path = Path::new(&cache_dir).join("model.safetensors");
    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");

    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        return format!("Permission error: {}", e);
    }

    // For debugging, delete the model if it already exists
    // if model_path.exists() {
    //     std::fs::remove_file(&model_path).unwrap();
    // }

    if !model_path.exists() {
        let response = reqwest::get(model_url).await;

        match response {
            Ok(res) => {
                let mut stream = res.bytes_stream();
                let mut file = File::create(&model_path).await.unwrap();
                while let Some(item) = stream.next().await {
                    file.write_all(&item.unwrap()).await.unwrap();
                }
            }
            Err(e) => {
                // Return a string error back to Flutter instead of panicking
                return format!("Download failed: {}", e);
            }
        }        
    }

    if !tokenizer_path.exists() {
        let response = reqwest::get(tokenizer_url).await;

        match response {
            Ok(res) => {
                let mut stream = res.bytes_stream();
                let mut file = File::create(&tokenizer_path).await.unwrap();
                while let Some(item) = stream.next().await {
                    file.write_all(&item.unwrap()).await.unwrap();
                }
            }
            Err(e) => {
                // Return a string error back to Flutter instead of panicking
                return format!("Download failed: {}", e);
            }
        }
    }

    let file = std::fs::File::open(&model_path).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
    let safetensors = SafeTensors::deserialize(&mmap).unwrap();

    let record = model.clone().into_record();
    let model_with_weights = load_qwen_record(&config, &safetensors, record, &device);
    model = model.load_record(model_with_weights);

    // CRITICAL: Ensure the 'mmap' and 'safetensors' variables are dropped or
    // go out of scope here to free up that ~1-2GB of RAM.
    drop(safetensors);
    drop(mmap);

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    GLOBAL_MODEL.set(Mutex::new(model)).unwrap();
    GLOBAL_TOKENIZER.set(tokenizer).unwrap();
    GLOBAL_CONFIG.set(config).unwrap();
    GLOBAL_DEVICE.set(device).unwrap();
    SESSIONS.set(DashMap::new()).unwrap();
    return "Success".to_string();
}

pub fn init_session() -> String {
    let session_id = Uuid::new_v4().to_string();
    let tokenizer = GLOBAL_TOKENIZER.get().unwrap();
    let mut token_ids = Vec::new();
    let im_start_id = tokenizer
        .token_to_id("<|im_start|>")
        .expect("Missing <|im_start|>");
    let im_end_id = tokenizer
        .token_to_id("<|im_end|>")
        .expect("Missing <|im_end|>");
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n
    let system_text = "You are a helpful assistant.";
    let system_tokens = tokenizer
        .encode(system_text, false)
        .unwrap()
        .get_ids()
        .to_vec();
    token_ids.push(im_start_id);
    token_ids.extend(tokenizer.encode("system", false).unwrap().get_ids());
    token_ids.push(newline_id);
    token_ids.extend(system_tokens);
    token_ids.push(im_end_id);
    token_ids.push(newline_id);
    SESSIONS
        .get()
        .unwrap()
        .insert(session_id.clone(), token_ids);
    session_id
}

pub fn generate(name: String) -> String {
    format!("Hello, {name} :)")
}

pub fn generate_response(session_id: &str, prompt: &str) -> String {
    let tokenizer = GLOBAL_TOKENIZER.get().unwrap();
    let config = GLOBAL_CONFIG.get().unwrap();
    let device = GLOBAL_DEVICE.get().unwrap();
    let model = GLOBAL_MODEL.get().unwrap().lock();

    let mut token_ids = SESSIONS.get().unwrap().get_mut(session_id).unwrap();

    let im_start_id = tokenizer
        .token_to_id("<|im_start|>")
        .expect("Missing <|im_start|>");
    let im_end_id = tokenizer
        .token_to_id("<|im_end|>")
        .expect("Missing <|im_end|>");
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n

    let user_tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();

    token_ids.push(im_start_id);
    token_ids.extend(tokenizer.encode("user", false).unwrap().get_ids());
    token_ids.push(newline_id);
    token_ids.extend(user_tokens);
    token_ids.push(im_end_id);
    token_ids.push(newline_id);
    token_ids.push(im_start_id);
    token_ids.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
    token_ids.push(newline_id);

    let mut assistant_response_ids = Vec::new();
    for _ in 0..8 {
        // Generate up to 1024 tokens
        let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                token_ids.clone().into_iter().map(|x| x as i32).collect(),
                Shape::new([1, token_ids.len()]),
            ),
            device,
        );

        let output = model.forward(input_tensor);

        let next_token_logits = output
            .slice([
                0..1,
                (token_ids.len() - 1)..(token_ids.len()),
                0..config.vocab_size,
            ])
            .reshape([1, config.vocab_size]);
        // Burn 0.20.1 preferred way to sync GPU data to CPU
        let next_token_id = next_token_logits
            .argmax(1)
            .to_data() // Forces a GPU sync and returns TensorData
            .into_vec::<i32>()
            .expect("GPU Sync Failed")[0] as u32;

        if next_token_id == im_end_id {
            break;
        }

        token_ids.push(next_token_id);
        assistant_response_ids.push(next_token_id);
    }
    token_ids.push(im_end_id);
    token_ids.push(newline_id);

    tokenizer.decode(&assistant_response_ids, true).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let name = "snowglobe".to_string();
        let result = generate(name);
        assert_eq!(result, "Hello, snowglobe :)");
    }

    #[tokio::test]
    async fn test_one_plus_one() {
        let cache_dir = "./tmp/testing";
        tokio::fs::create_dir_all(cache_dir).await.unwrap();
        init(cache_dir.to_string()).await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";
        let response = generate_response(&session_id, prompt);

        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert_eq!(response.trim(), "2");
    }
}
