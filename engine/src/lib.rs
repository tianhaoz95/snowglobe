#![recursion_limit = "256"]

pub mod adapter;
pub mod layer;
pub mod model;
pub mod rope;
pub mod utils;
pub mod weight;

pub use utils::downloader::{download_model, download_qwen2_5_0_5b_instruct, download_qwen3_0_6b};

use crate::layer::large_vocab::CHUNK_SIZE;
pub use crate::model::{
    InitConfig, KVCache, LoadedModel, Qwen, QwenConfig, QwenPte, QwenVariant,
};
use crate::weight::load_qwen_record;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;
use uuid::Uuid;

/// Generates a completion using an ExecuTorch .pte model.
///
/// Note: This function requires the ExecuTorch C++ static libraries to be present in the
/// path specified by the `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` environment variable during build.
/// To avoid ABI mismatches (like [abi:ne200100] on macOS), ensure that ExecuTorch is built
/// with the same SDK and compiler flags as the Rust engine.
pub fn experimental_completion_with_pte(pte_path: &str, prompt: &str) -> Result<String, String> {
    let model = QwenPte::<Backend>::new(pte_path)?;
    model.generate(prompt, 16)
}

#[cfg(feature = "high_perf")]
pub mod backend_setup {
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;

    // GPU Types: Wgpu takes <Float, Int>
    pub type Backend = Wgpu<f32, i32>;
    pub type Device = WgpuDevice;

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub type GraphicsApi = burn_wgpu::graphics::Metal;
    #[cfg(target_os = "android")]
    pub type GraphicsApi = burn_wgpu::graphics::Vulkan;
    #[cfg(not(any(target_os = "ios", target_os = "macos", target_os = "android")))]
    pub type GraphicsApi = burn_wgpu::graphics::AutoGraphicsApi;
}

#[cfg(not(feature = "high_perf"))]
pub mod backend_setup {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    // CPU Types: NdArray in 0.20.1 usually only takes <Float>
    pub type Backend = NdArray<f32, i32>;
    pub type Device = NdArrayDevice;
}

pub use backend_setup::Backend;
pub use backend_setup::Device;

pub static GLOBAL_MODEL: OnceCell<LoadedModel<Backend>> = OnceCell::new();

pub struct SessionState {
    pub tokens: Vec<u32>,
    pub cache: Option<Vec<KVCache<Backend>>>,
    pub offset: usize,
}

pub static SESSIONS: OnceCell<DashMap<String, SessionState>> = OnceCell::new();

pub fn check_backend() -> String {
    #[cfg(feature = "high_perf")]
    return "🚀 USING GPU (WGPU/VULKAN)".to_string();

    #[cfg(not(feature = "high_perf"))]
    return "💻 USING CPU (NDARRAY)".to_string();
}

static GPU_SETUP: once_cell::sync::OnceCell<()> = once_cell::sync::OnceCell::new();

pub async fn init(cache_dir: String, init_config: InitConfig) -> String {
    let device = init_platform(&init_config).await;
    init_model(cache_dir, init_config, device).await
}

async fn init_platform(_init_config: &InitConfig) -> Device {
    // 1. Initialize Device based on Feature
    #[cfg(feature = "high_perf")]
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    #[cfg(not(feature = "high_perf"))]
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // 2. Setup GPU only if High Perf is enabled and NOT using ExecuTorch
    #[cfg(feature = "high_perf")]
    {
        if !_init_config.use_executorch && GPU_SETUP.get().is_none() {
            let _setup = burn_wgpu::init_setup_async::<backend_setup::GraphicsApi>(
                &device,
                burn_wgpu::RuntimeOptions::default(),
            )
            .await;
            let _ = GPU_SETUP.set(());
        }
    }

    device
}

async fn init_model(cache_dir: String, init_config: InitConfig, device: Device) -> String {
    let config_path = Path::new(&cache_dir).join("config.json");
    let mut config = if config_path.exists() {
        let config_str = std::fs::read_to_string(config_path).expect("Failed to read config.json");
        serde_json::from_str::<QwenConfig>(&config_str).expect("Failed to parse config.json")
    } else {
        QwenConfig::default()
    };

    if init_config.vocab_shards != 0 {
        config.vocab_shards = init_config.vocab_shards;
    } else if config.vocab_shards == 0 {
        config.vocab_shards = (config.vocab_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }

    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");
    if !tokenizer_path.exists() {
        return "Tokenizer file missing. Please call download_model first.".to_string();
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    if init_config.use_executorch {
        let pte_path = Path::new(&cache_dir).join("model.pte");
        if !pte_path.exists() {
            return "PTE model file missing (model.pte).".to_string();
        }
        let model = QwenPte::<Backend>::new(pte_path.to_str().unwrap())
            .expect("Failed to load PTE model");

        let _ = GLOBAL_MODEL.set(LoadedModel {
            model: Mutex::new(QwenVariant::ExecuTorch(model)),
            tokenizer,
            config,
            device,
            init_config,
        });
    } else {
        let mut model: Qwen<Backend> = config.init(&device);
        let model_path = Path::new(&cache_dir).join("model.safetensors");

        if !model_path.exists() {
            return "Model file missing. Please call download_model first.".to_string();
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

        let _ = GLOBAL_MODEL.set(LoadedModel {
            model: Mutex::new(QwenVariant::Burn(model)),
            tokenizer,
            config,
            device,
            init_config,
        });
    }

    let _ = SESSIONS.set(DashMap::new());
    return "Success".to_string();
}

pub fn init_session() -> String {
    let session_id = Uuid::new_v4().to_string();
    let loaded_model = GLOBAL_MODEL
        .get()
        .expect("Global model not initialized. Call init first.");
    let tokenizer = &loaded_model.tokenizer;
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
    let loaded_model = GLOBAL_MODEL
        .get()
        .expect("Global model not initialized.");
    let tokenizer = &loaded_model.tokenizer;
    let model_config = &loaded_model.config;
    let init_config = &loaded_model.init_config;
    let device = &loaded_model.device;
    let model = loaded_model.model.lock();

    let mut session_state = SESSIONS
        .get()
        .expect("SESSIONS not initialized.")
        .get_mut(session_id)
        .unwrap();

    let im_start_id = tokenizer
        .token_to_id("<|im_start|>")
        .expect("Missing <|im_start|>");
    let im_end_id = tokenizer
        .token_to_id("<|im_end|>")
        .expect("Missing <|im_end|>");
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n

    let user_tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();

    session_state.tokens.push(im_start_id);
    session_state
        .tokens
        .extend(tokenizer.encode("user", false).unwrap().get_ids());
    session_state.tokens.push(newline_id);
    session_state.tokens.extend(user_tokens);
    session_state.tokens.push(im_end_id);
    session_state.tokens.push(newline_id);
    session_state.tokens.push(im_start_id);
    session_state
        .tokens
        .extend(tokenizer.encode("assistant", false).unwrap().get_ids());
    session_state.tokens.push(newline_id);

    for _ in 0..init_config.max_gen_len {
        // 1. Process pending tokens
        // For ExecuTorch, we always pass the full sequence (0..len)
        // For Burn, we only pass new tokens (offset..len)
        let start_idx = if init_config.use_executorch { 0 } else { session_state.offset };
        let num_to_process = session_state.tokens.len() - start_idx;
        let num_new = session_state.tokens.len() - session_state.offset;

        if num_new == 0 {
            break;
        }

        let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                session_state.tokens[start_idx..]
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                Shape::new([1, num_to_process]),
            ),
            device,
        );

        let (output, new_cache) = model.forward(
            input_tensor,
            session_state
                .cache
                .clone()
                .map(|c| c.into_iter().map(Some).collect()),
            session_state.offset,
        );

        session_state.cache = Some(new_cache);
        session_state.offset += num_new;

        let next_token_logits = output
            .slice([
                0..1,
                (num_to_process - 1)..(num_to_process),
                0..model_config.vocab_size,
            ])
            .reshape([1, model_config.vocab_size]);

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
    async fn test_kv_cache_latency_comparison() {
        let cache_dir = setup_test("./tmp/testing", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;

        let loaded_model = GLOBAL_MODEL.get().unwrap();
        let tokenizer = &loaded_model.tokenizer;
        let model = loaded_model.model.lock();
        let device = &loaded_model.device;

        // Create a moderately long prompt to make the difference noticeable
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(60);
        let tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
        let n = tokens.len();
        println!("Benchmark sequence length: {} tokens", n);

        // 1. Prefill to get initial cache
        let input_prefill: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                tokens.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                Shape::new([1, n]),
            ),
            device,
        );
        let (_, cache) = model.forward(input_prefill, None, 0);

        // 2. Measure "With KV Cache" (Generating 1 token)
        let next_token_id = 1234u32;
        let input_cached: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(vec![next_token_id as i32], Shape::new([1, 1])),
            device,
        );

        // Warmup
        for _ in 0..3 {
            let _ = model.forward(
                input_cached.clone(),
                Some(cache.iter().cloned().map(Some).collect()),
                n,
            );
        }

        let start_cached = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = model.forward(
                input_cached.clone(),
                Some(cache.iter().cloned().map(Some).collect()),
                n,
            );
        }
        let duration_cached = start_cached.elapsed() / iterations;

        // 3. Measure "Without KV Cache" (Processing N+1 tokens from scratch)
        let mut tokens_full = tokens.clone();
        tokens_full.push(next_token_id);
        let input_full: Tensor<Backend, 2, Int> = Tensor::from_data(
            TensorData::new(
                tokens_full.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                Shape::new([1, n + 1]),
            ),
            device,
        );

        // Warmup
        for _ in 0..3 {
            let _ = model.forward(input_full.clone(), None, 0);
        }

        let start_full = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(input_full.clone(), None, 0);
        }
        let duration_full = start_full.elapsed() / iterations;

        let speedup = duration_full.as_secs_f64() / duration_cached.as_secs_f64();
        println!("Latency WITH KV Cache:    {:?}", duration_cached);
        println!("Latency WITHOUT KV Cache: {:?}", duration_full);
        println!("Measured Speedup:         {:.2}x", speedup);

        // With N tokens, the "Without KV Cache" forward pass should be significantly slower.
        // We expect at least 2x speedup for this sequence length.
        assert!(
            speedup >= 2.0,
            "KV Cache speedup was only {:.2}x, expected >= 2.0x",
            speedup
        );
    }

    #[tokio::test]
    async fn test_multi_turn() {
        let cache_dir = setup_test("./tmp/testing", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;
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

    struct TestSink(Mutex<String>);
    impl StreamSink<String> for TestSink {
        fn add(&self, value: String) -> bool {
            self.0.lock().push_str(&value);
            true
        }
    }

    async fn setup_test(cache_dir: &str, model: &str) -> String {
        tokio::fs::create_dir_all(cache_dir).await.unwrap();
        if model == "qwen3" {
            download_qwen3_0_6b(cache_dir.to_string()).await;
        } else {
            download_qwen2_5_0_5b_instruct(cache_dir.to_string()).await;
        }
        cache_dir.to_string()
    }

    #[tokio::test]
    async fn test_one_plus_one_pte() {
        // 1. Ensure PTE exists at root and prepare cache dir
        let pte_path = "../qwen3_0.6b.pte";
        assert!(std::path::Path::new(pte_path).exists());

        let cache_dir = setup_test("./tmp/testing_pte", "qwen3").await;
        
        // 2. Link or copy the pte file to the cache dir as expected by init_model
        let target_pte = Path::new(&cache_dir).join("model.pte");
        if !target_pte.exists() {
            std::fs::copy(pte_path, target_pte).expect("Failed to copy PTE to cache dir");
        }

        // 3. Initialize engine with ExecuTorch enabled
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: true,
            },
        )
        .await;

        let session_id = init_session();
        let prompt = "what is the capital of China? /no_think";
        let sink = TestSink(Mutex::new(String::new()));
        
        // 4. Verify the integrated generate_response works with ExecuTorch
        let result = generate_response(&session_id, prompt, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert!(response.to_lowercase().contains("beijing"));
    }

    #[tokio::test]
    async fn test_one_plus_one_qwen2() {
        let cache_dir = setup_test("./tmp/testing", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;
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
    async fn test_one_plus_one_qwen3() {
        let cache_dir = setup_test("./tmp/testing_qwen3", "qwen3").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;
        let session_id = init_session();
        let prompt = "what is the capital of china? only answer the city name /no_think";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert!(response.to_lowercase().contains("beijing"));
    }

    #[tokio::test]
    async fn test_sharded_one_plus_one() {
        let cache_dir = setup_test("./tmp/testing", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 0,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;
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
        let cache_dir = setup_test("./tmp/testing", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 10,
                max_gen_len: 256,
                use_executorch: false,
            },
        )
        .await;
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
