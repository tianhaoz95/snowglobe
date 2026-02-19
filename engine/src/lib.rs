#![recursion_limit = "256"]

pub mod model;
pub mod layer;
pub mod rope;
pub mod weight;

use crate::model::{KVCache, Qwen, QwenConfig};
use crate::layer::large_vocab::CHUNK_SIZE;
use crate::weight::load_qwen_record;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use dashmap::DashMap;
use futures_util::StreamExt;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

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

pub struct SessionState {
    pub tokens: Vec<u32>,
    pub cache: Option<Vec<KVCache<Backend>>>,
    pub offset: usize,
}

static SESSIONS: OnceCell<DashMap<String, SessionState>> = OnceCell::new();

pub fn check_backend() -> String {
    #[cfg(feature = "high_perf")]
    return "🚀 USING GPU (WGPU/VULKAN)".to_string();

    #[cfg(not(feature = "high_perf"))]
    return "💻 USING CPU (NDARRAY)".to_string();
}

async fn download_file(url: &str, path: &Path) -> Result<(), String> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Download failed: {}", e))?;
    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }
    let mut stream = response.bytes_stream();
    let mut file = File::create(path)
        .await
        .map_err(|e| format!("File creation failed: {}", e))?;
    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| format!("Stream error: {}", e))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
    }
    Ok(())
}

pub async fn download_model(cache_dir: String) -> String {
    let model_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors";
    let tokenizer_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";

    let model_path = Path::new(&cache_dir).join("model.safetensors");
    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");

    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        return format!("Permission error: {}", e);
    }

    if !model_path.exists() {
        if let Err(e) = download_file(model_url, &model_path).await {
            return e;
        }
    } else {
        // Simple check to ensure file is at least somewhat reasonably sized.
        if let Ok(meta) = std::fs::metadata(&model_path) {
            if meta.len() < 900 * 1024 * 1024 { // ~942M expected
                let _ = std::fs::remove_file(&model_path);
                if let Err(e) = download_file(model_url, &model_path).await {
                    return e;
                }
            }
        }
    }

    if !tokenizer_path.exists() {
        if let Err(e) = download_file(tokenizer_url, &tokenizer_path).await {
            return e;
        }
    }

    "Success".to_string()
}

static GPU_SETUP: once_cell::sync::OnceCell<()> = once_cell::sync::OnceCell::new();

pub async fn init(cache_dir: String, vocab_shards: usize) -> String {
    let mut config = QwenConfig::default();
    config.vocab_shards = if vocab_shards == 0 {
        (config.vocab_size + CHUNK_SIZE - 1) / CHUNK_SIZE
    } else {
        vocab_shards
    };

    // 1. Initialize Device based on Feature
    #[cfg(feature = "high_perf")]
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    #[cfg(not(feature = "high_perf"))]
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // 2. Setup GPU only if High Perf is enabled
    #[cfg(feature = "high_perf")]
    {
        if GPU_SETUP.get().is_none() {
            let _setup = burn_wgpu::init_setup_async::<backend_setup::GraphicsApi>(
                &device,
                burn_wgpu::RuntimeOptions::default(),
            )
            .await;
            let _ = GPU_SETUP.set(());
        }
    }

    let mut model: Qwen<Backend> = config.init(&device);

    let model_path = Path::new(&cache_dir).join("model.safetensors");
    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");

    if !model_path.exists() {
        return "Model file missing. Please call download_model first.".to_string();
    }

    if !tokenizer_path.exists() {
        return "Tokenizer file missing. Please call download_model first.".to_string();
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
    drop(file);

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    let _ = GLOBAL_MODEL.set(Mutex::new(model));
    let _ = GLOBAL_TOKENIZER.set(tokenizer);
    let _ = GLOBAL_CONFIG.set(config);
    let _ = GLOBAL_DEVICE.set(device);
    let _ = SESSIONS.set(DashMap::new());
    return "Success".to_string();
}

pub fn init_session() -> String {
    let session_id = Uuid::new_v4().to_string();
    let tokenizer = GLOBAL_TOKENIZER.get().expect("Global tokenizer not initialized. Call init first.");
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

    let state = SessionState {
        tokens: token_ids,
        cache: None,
        offset: 0,
    };

    SESSIONS
        .get()
        .expect("SESSIONS not initialized. Call init first.")
        .insert(session_id.clone(), state);
    session_id
}

pub fn generate(name: String) -> String {
    format!("Hello, {name} :)")
}

pub trait StreamSink<T> {
    fn add(&self, value: T) -> bool;
}

impl<T, S: StreamSink<T>> StreamSink<T> for &S {
    fn add(&self, value: T) -> bool {
        (*self).add(value)
    }
}

pub fn generate_response<S>(session_id: &str, prompt: &str, sink: S) -> Result<(), String>
where
    S: StreamSink<String>,
{
    let tokenizer = GLOBAL_TOKENIZER.get().expect("Global tokenizer not initialized.");
    let config = GLOBAL_CONFIG.get().expect("Global config not initialized.");
    let device = GLOBAL_DEVICE.get().expect("Global device not initialized.");
    let model = GLOBAL_MODEL.get().expect("Global model not initialized.").lock();

    let mut session_state = SESSIONS.get().expect("SESSIONS not initialized.").get_mut(session_id).unwrap();

    let im_start_id = tokenizer
        .token_to_id("<|im_start|>")
        .expect("Missing <|im_start|>");
    let im_end_id = tokenizer
        .token_to_id("<|im_end|>")
        .expect("Missing <|im_end|>");
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n

    let user_tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();

    session_state.tokens.push(im_start_id);
    session_state.tokens.extend(tokenizer.encode("user", false).unwrap().get_ids());
    session_state.tokens.push(newline_id);
    session_state.tokens.extend(user_tokens);
    session_state.tokens.push(im_end_id);
    session_state.tokens.push(newline_id);
    session_state.tokens.push(im_start_id);
    session_state.tokens.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
    session_state.tokens.push(newline_id);

    for _ in 0..256 {
        // 1. Process all pending tokens
        let num_pending = session_state.tokens.len() - session_state.offset;
        if num_pending == 0 {
             break;
        }

        let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                session_state.tokens[session_state.offset..].iter().map(|&x| x as i32).collect::<Vec<_>>(),
                Shape::new([1, num_pending]),
            ),
            device,
        );

        let (output, new_cache) = model.forward(
            input_tensor,
            session_state.cache.clone().map(|c| c.into_iter().map(Some).collect()),
            session_state.offset
        );

        session_state.cache = Some(new_cache);
        session_state.offset += num_pending;

        let next_token_logits = output
            .slice([
                0..1,
                (num_pending - 1)..(num_pending),
                0..config.vocab_size,
            ])
            .reshape([1, config.vocab_size]);

        let next_token_id = next_token_logits
            .argmax(1)
            .to_data()
            .into_vec::<i32>()
            .expect("GPU Sync Failed")[0] as u32;

        if next_token_id == im_end_id {
            break;
        }

        session_state.tokens.push(next_token_id);

        let new_text = tokenizer.decode(&[next_token_id], true).unwrap_or_default();
        if !new_text.is_empty() {
            if !sink.add(new_text) {
                break;
            }
        }
    }
    session_state.tokens.push(im_end_id);
    session_state.tokens.push(newline_id);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_turn() {
        let cache_dir = setup_test().await;
        init(cache_dir, 1).await;
        let session_id = init_session();
        
        // Turn 1
        let prompt1 = "My name is Alice. Remember that.";
        let sink1 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt1, &sink1).unwrap();
        
        // Turn 2
        let prompt2 = "What is my name?";
        let sink2 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt2, &sink2).unwrap();
        
        let response2 = sink2.0.lock().clone();
        println!("Prompt 2: {}", prompt2);
        println!("Response 2: {}", response2);
        
        assert!(response2.contains("Alice"));
    }

    #[test]
    fn it_works() {
        let name = "snowglobe".to_string();
        let result = generate(name);
        assert_eq!(result, "Hello, snowglobe :)");
    }

    struct TestSink(Mutex<String>);
    impl StreamSink<String> for TestSink {
        fn add(&self, value: String) -> bool {
            self.0.lock().push_str(&value);
            true
        }
    }

    async fn setup_test() -> String {
        let cache_dir = "./tmp/testing";
        tokio::fs::create_dir_all(cache_dir).await.unwrap();
        download_model(cache_dir.to_string()).await;
        cache_dir.to_string()
    }

    #[tokio::test]
    async fn test_one_plus_one() {
        let cache_dir = setup_test().await;
        init(cache_dir, 1).await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert_eq!(response.trim(), "2");
    }

    #[tokio::test]
    async fn test_sharded_one_plus_one() {
        let cache_dir = setup_test().await;
        init(cache_dir, 0).await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert_eq!(response.trim(), "2");
    }

    #[tokio::test]
    async fn test_multi_sharded_one_plus_one() {
        let cache_dir = setup_test().await;
        init(cache_dir, 10).await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert_eq!(response.trim(), "2");
    }
}
