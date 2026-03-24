#![recursion_limit = "256"]

pub mod adapter;
pub mod layer;
pub mod model;
pub mod rope;
pub mod utils;
pub mod weight;

pub use utils::downloader::{download_model, download_qwen2_5_0_5b_instruct, download_qwen3_0_6b, download_qwen3_5_0_8b, download_qwen_gguf};

use crate::layer::large_vocab::CHUNK_SIZE;
pub use crate::model::{
    InitConfig, KVCache, LoadedModel, Qwen, QwenConfig, QwenPte, EngineVariant, BackendType
};
pub use crate::model::runner::{EngineSession, ModelRunner};
use crate::weight::load_qwen_record;
use burn::prelude::*;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;
use uuid::Uuid;

use std::sync::Arc;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse, ChatCompletionResponseMessage,
    ChatChoice, ChatChoiceStream, ChatCompletionStreamResponseDelta,
    ChatCompletionMessageToolCalls, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    Role, FinishReason,
    FunctionCallStream,
    ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessageContentPart,
    FunctionCall,
};
use std::pin::Pin;
use futures_util::{Stream, StreamExt};
use tool_parser::{ToolParser, QwenParser};

pub enum ChatCompletionOutput {
    Single(CreateChatCompletionResponse),
    Stream(Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse, String>> + Send>>),
}

fn extract_system_content(content: &ChatCompletionRequestSystemMessageContent) -> String {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestSystemMessageContent::Array(a) => a.iter().filter_map(|part| {
            use async_openai::types::chat::ChatCompletionRequestSystemMessageContentPart::*;
            match part {
                Text(t) => Some(t.text.clone()),
            }
        }).collect::<Vec<_>>().join("\n"),
    }
}

fn extract_user_content(content: &ChatCompletionRequestUserMessageContent) -> String {
    match content {
        ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestUserMessageContent::Array(a) => a.iter().filter_map(|part| {
            match part {
                ChatCompletionRequestUserMessageContentPart::Text(t) => Some(t.text.clone()),
                _ => None,
            }
        }).collect::<Vec<_>>().join("\n"),
    }
}

fn extract_assistant_content(content: &ChatCompletionRequestAssistantMessageContent) -> String {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestAssistantMessageContent::Array(a) => a.iter().filter_map(|part| {
            use async_openai::types::chat::ChatCompletionRequestAssistantMessageContentPart::*;
            match part {
                Text(t) => Some(t.text.clone()),
                _ => None,
            }
        }).collect::<Vec<_>>().join("\n"),
    }
}

fn extract_tool_content(content: &ChatCompletionRequestToolMessageContent) -> String {
    match content {
        ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestToolMessageContent::Array(a) => a.iter().filter_map(|part| {
            use async_openai::types::chat::ChatCompletionRequestToolMessageContentPart::*;
            match part {
                Text(t) => Some(t.text.clone()),
            }
        }).collect::<Vec<_>>().join("\n"),
    }
}

fn forward_model(model_arc: &Arc<Mutex<EngineVariant>>, session_state: &mut EngineSession) -> Result<Vec<f32>, String> {
    let model = model_arc.lock();
    match &*model {
        EngineVariant::Burn(m) => m.forward(session_state).map_err(|e| e.to_string()),
        EngineVariant::ExecuTorch(m) => m.forward(session_state).map_err(|e| e.to_string()),
        EngineVariant::LlamaCpp(m) => m.forward(session_state).map_err(|e| e.to_string()),
        EngineVariant::Speculative(m) => m.forward(session_state).map_err(|e| e.to_string()),
    }
}

pub async fn create_chat_completion(
    request: CreateChatCompletionRequest,
) -> Result<ChatCompletionOutput, String> {
    let global_lock = GLOBAL_MODEL.read();
    let loaded_model = match &*global_lock {
        Some(m) => m,
        None => return Err("Error: Global model not initialized. Call init first.".to_string()),
    };
    let tokenizer = &loaded_model.tokenizer;
    let init_config = &loaded_model.init_config;

    let im_start_id = match tokenizer.token_to_id("<|im_start|>") {
        Some(id) => id,
        None => return Err("Error: Missing <|im_start|> in tokenizer".to_string()),
    };
    let im_end_id = match tokenizer.token_to_id("<|im_end|>") {
        Some(id) => id,
        None => return Err("Error: Missing <|im_end|> in tokenizer".to_string()),
    };
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198);

    // Format prompt from messages
    let mut token_ids = Vec::new();
    for msg in &request.messages {
        let (role, content) = match msg {
            ChatCompletionRequestMessage::System(m) => ("system", extract_system_content(&m.content)),
            ChatCompletionRequestMessage::User(m) => ("user", extract_user_content(&m.content)),
            ChatCompletionRequestMessage::Assistant(m) => {
                let content = m.content.as_ref().map(extract_assistant_content).unwrap_or_default();
                ("assistant", content)
            },
            ChatCompletionRequestMessage::Tool(m) => ("tool", extract_tool_content(&m.content)),
            _ => ("unknown", "".to_string()),
        };

        token_ids.push(im_start_id);
        token_ids.extend(tokenizer.encode(role, false).unwrap().get_ids());
        token_ids.push(newline_id);
        token_ids.extend(tokenizer.encode(content.as_str(), false).unwrap().get_ids());
        token_ids.push(im_end_id);
        token_ids.push(newline_id);
    }

    // Prepare assistant turn
    token_ids.push(im_start_id);
    token_ids.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
    token_ids.push(newline_id);

    let session_state = EngineSession {
        tokens: token_ids,
        state: None,
        offset: 0,
        last_accepted_count: 0,
        is_speculative: false,
    };

    #[allow(deprecated)]
    let generation_limit = request.max_completion_tokens.or(request.max_tokens).unwrap_or(init_config.max_gen_len as u32);
    let is_streaming_request = request.stream.unwrap_or(false);

    let model_arc = loaded_model.model.clone();
    let tokenizer_clone = tokenizer.clone();
    let im_end_id = im_end_id;

    let output_stream = async_stream::try_stream! {
        let mut session_state = session_state;
        let mut full_text = String::new();
        
        for _ in 0..generation_limit {
            let num_tokens_before = session_state.tokens.len();
            let logits = forward_model(&model_arc, &mut session_state)?;

            let next_token_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| "Error: Logits are empty".to_string())?;

            if next_token_id == im_end_id {
                break;
            }

            session_state.tokens.push(next_token_id);
            let new_text = tokenizer_clone.decode(&session_state.tokens[num_tokens_before..], true).map_err(|e| e.to_string())?;
            
            if !new_text.is_empty() {
                full_text.push_str(&new_text);
                
                yield CreateChatCompletionStreamResponse {
                    id: Uuid::new_v4().to_string(),
                    choices: vec![ChatChoiceStream {
                        index: 0,
                        delta: ChatCompletionStreamResponseDelta {
                            content: Some(new_text),
                            role: Some(Role::Assistant),
                            tool_calls: None,
                            refusal: None,
                            #[allow(deprecated)]
                            function_call: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created: 0,
                    model: "snowglobe".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    #[allow(deprecated)]
                    system_fingerprint: None,
                    service_tier: None,
                    usage: None,
                };
            }
        }

        // After loop, check for tool calls in the full text
        let parser = QwenParser::new();
        let (_cleaned_text, tool_calls) = parser.parse_complete(&full_text).await.map_err(|e| e.to_string())?;
        if !tool_calls.is_empty() {
            let openai_tool_calls: Vec<ChatCompletionMessageToolCallChunk> = tool_calls.into_iter().enumerate().map(|(idx, tc)| {
                ChatCompletionMessageToolCallChunk {
                    index: idx as u32,
                    id: Some(format!("call_{}", idx)),
                    r#type: Some(async_openai::types::chat::FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: Some(tc.function.name),
                        arguments: Some(tc.function.arguments),
                    }),
                }
            }).collect();

            yield CreateChatCompletionStreamResponse {
                id: Uuid::new_v4().to_string(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: None,
                        role: None,
                        tool_calls: Some(openai_tool_calls),
                        refusal: None,
                        #[allow(deprecated)]
                        function_call: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: 0,
                model: "snowglobe".to_string(),
                object: "chat.completion.chunk".to_string(),
                #[allow(deprecated)]
                system_fingerprint: None,
                service_tier: None,
                usage: None,
            };
        }
    };

    if is_streaming_request {
        Ok(ChatCompletionOutput::Stream(Box::pin(output_stream)))
    } else {
        let mut pinned_stream = Box::pin(output_stream);
        let mut full_text = String::new();
        let mut final_tool_calls = None;
        let mut last_id = Uuid::new_v4().to_string();
        let mut last_model = "snowglobe".to_string();

        while let Some(chunk_res) = pinned_stream.next().await {
            let chunk = chunk_res?;
            last_id = chunk.id;
            last_model = chunk.model;
            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    full_text.push_str(&content);
                }
                if let Some(tool_calls) = choice.delta.tool_calls {
                    let mut calls = Vec::new();
                    for tc_chunk in tool_calls {
                        if let Some(func) = tc_chunk.function {
                            calls.push(ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                                id: tc_chunk.id.unwrap_or_else(|| format!("call_{}", calls.len())),
                                function: FunctionCall {
                                    name: func.name.unwrap_or_default(),
                                    arguments: func.arguments.unwrap_or_default(),
                                },
                            }));
                        }
                    }
                    if !calls.is_empty() {
                        final_tool_calls = Some(calls);
                    }
                }
            }
        }

        let parser = QwenParser::new();
        let (cleaned_text, _tool_calls_from_text) = parser.parse_complete(&full_text).await.map_err(|e| e.to_string())?;

        let response = CreateChatCompletionResponse {
            id: last_id,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    content: if final_tool_calls.is_some() { 
                        if cleaned_text.is_empty() { None } else { Some(cleaned_text) }
                    } else { Some(full_text) },
                    role: Role::Assistant,
                    tool_calls: final_tool_calls,
                    refusal: None,
                    annotations: None,
                    #[allow(deprecated)]
                    function_call: None,
                    audio: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            created: 0,
            model: last_model,
            object: "chat.completion".to_string(),
            #[allow(deprecated)]
            system_fingerprint: None,
            service_tier: None,
            usage: None,
        };

        Ok(ChatCompletionOutput::Single(response))
    }
}



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

pub static GLOBAL_MODEL: RwLock<Option<LoadedModel<Backend>>> = RwLock::new(None);

pub type SessionState = EngineSession;

pub static SESSIONS: RwLock<Option<DashMap<String, SessionState>>> = RwLock::new(None);

pub fn check_backend() -> String {
    #[cfg(feature = "high_perf")]
    {
        #[cfg(target_os = "android")]
        return "Vulkan GPU".to_string();
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        return "Metal GPU".to_string();
        #[cfg(not(any(target_os = "ios", target_os = "macos", target_os = "android")))]
        return "WGPU GPU".to_string();
    }

    #[cfg(not(feature = "high_perf"))]
    return "CPU (NdArray)".to_string();
}

pub fn get_model_info() -> Option<crate::model::ModelInfo> {
    GLOBAL_MODEL.read().as_ref().map(|m| {
        let mut info = m.config.get_model_info();
        info.runner = match &*m.model.lock() {
            EngineVariant::Burn(_) => "Burn".to_string(),
            EngineVariant::ExecuTorch(_) => "ExecuTorch".to_string(),
            EngineVariant::LlamaCpp(_) => "llama.cpp".to_string(),
            EngineVariant::Speculative(_) => "Speculative".to_string(),
        };
        info.backend = check_backend();
        info
    })
}

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
        let config_str = match std::fs::read_to_string(&config_path) {
            Ok(s) => s,
            Err(e) => return format!("Failed to read config.json: {}", e),
        };
        match serde_json::from_str::<QwenConfig>(&config_str) {
            Ok(c) => c,
            Err(_) => {
                // Try parsing nested text_config for Qwen 3.5 multimodal models
                let v: serde_json::Value = serde_json::from_str(&config_str).unwrap_or(serde_json::Value::Null);
                if let Some(text_config) = v.get("text_config") {
                    match serde_json::from_value::<QwenConfig>(text_config.clone()) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Warning: Failed to parse nested text_config: {}. Using default.", e);
                            QwenConfig::default()
                        }
                    }
                } else {
                    QwenConfig::default()
                }
            }
        }
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
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => return format!("Failed to load tokenizer.json: {}", e),
    };

    if init_config.use_executorch || init_config.backend == BackendType::ExecuTorch {
        let pte_path = Path::new(&cache_dir).join("model.pte");
        if !pte_path.exists() {
            return "PTE model file missing (model.pte).".to_string();
        }
        let pte_path_str = pte_path.to_str().unwrap_or("");
        let model = match QwenPte::<Backend>::new(pte_path_str) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load PTE model: {}", e),
        };

        let engine = if init_config.speculate_tokens > 0 {
            EngineVariant::Speculative(Box::new(crate::model::speculative::SpeculativeRunner::new(
                Box::new(model),
                init_config.speculate_tokens,
            )))
        } else {
            EngineVariant::ExecuTorch(Box::new(model))
        };

        let _ = GLOBAL_MODEL.write().insert(LoadedModel {
            model: Arc::new(Mutex::new(engine)),
            tokenizer,
            config,
            device,
            init_config,
        });
    } else if init_config.backend == BackendType::LlamaCpp {
        let gguf_path = Path::new(&cache_dir).join("model.gguf");
        if !gguf_path.exists() {
            return "GGUF model file missing (model.gguf).".to_string();
        }
        use crate::model::llama_cpp::LlamaCppRunner;
        let model = match LlamaCppRunner::load(&gguf_path, &init_config) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load LlamaCpp model: {}", e),
        };

        let engine = if init_config.speculate_tokens > 0 {
            EngineVariant::Speculative(Box::new(crate::model::speculative::SpeculativeRunner::new(
                model,
                init_config.speculate_tokens,
            )))
        } else {
            EngineVariant::LlamaCpp(model)
        };

        let _ = GLOBAL_MODEL.write().insert(LoadedModel {
            model: Arc::new(Mutex::new(engine)),
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

        let file = match std::fs::File::open(&model_path) {
            Ok(f) => f,
            Err(e) => return format!("Failed to open model.safetensors: {}", e),
        };
        let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
            Ok(m) => m,
            Err(e) => return format!("Failed to mmap model.safetensors: {}", e),
        };
        let safetensors = match SafeTensors::deserialize(&mmap) {
            Ok(s) => s,
            Err(e) => return format!("Failed to deserialize safetensors: {}", e),
        };

        let record = model.clone().into_record();
        let model_with_weights = load_qwen_record(&config, &safetensors, record, &device);
        model = model.load_record(model_with_weights);

        // CRITICAL: Ensure the 'mmap' and 'safetensors' variables are dropped or
        // go out of scope here to free up that ~1-2GB of RAM.
        drop(safetensors);
        drop(mmap);
        drop(file);

        let engine = if init_config.speculate_tokens > 0 {
            EngineVariant::Speculative(Box::new(crate::model::speculative::SpeculativeRunner::new(
                Box::new(model),
                init_config.speculate_tokens,
            )))
        } else {
            EngineVariant::Burn(Box::new(model))
        };

        let _ = GLOBAL_MODEL.write().insert(LoadedModel {
            model: Arc::new(Mutex::new(engine)),
            tokenizer,
            config,
            device,
            init_config,
        });
    }

    *SESSIONS.write() = Some(DashMap::new());
    return "Success".to_string();
}

pub fn init_session() -> String {
    let session_id = Uuid::new_v4().to_string();
    let global_lock = GLOBAL_MODEL.read();
    let loaded_model = match &*global_lock {
        Some(m) => m,
        None => return "Error: Global model not initialized. Call init first.".to_string(),
    };
    let tokenizer = &loaded_model.tokenizer;
    let mut token_ids = Vec::new();
    let im_start_id = match tokenizer.token_to_id("<|im_start|>") {
        Some(id) => id,
        None => return "Error: Missing <|im_start|> in tokenizer".to_string(),
    };
    let im_end_id = match tokenizer.token_to_id("<|im_end|>") {
        Some(id) => id,
        None => return "Error: Missing <|im_end|> in tokenizer".to_string(),
    };
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
        state: None,
        offset: 0,
        last_accepted_count: 0, is_speculative: false,
    };

    let sessions_lock = SESSIONS.read();
    let sessions = sessions_lock.as_ref().expect("SESSIONS not initialized. Call init first.");
    sessions.insert(session_id.clone(), state);
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

pub fn generate_response<S>(
    session_id: &str,
    prompt: &str,
    max_gen_len: u32,
    sink: S,
) -> Result<(), String>
where
    S: StreamSink<String>,
{
    let global_lock = GLOBAL_MODEL.read();
    let loaded_model = match &*global_lock {
        Some(m) => m,
        None => return Err("Error: Global model not initialized. Call init first.".to_string()),
    };
    let tokenizer = &loaded_model.tokenizer;
    let _model_config = &loaded_model.config;
    let init_config = &loaded_model.init_config;
    let _device = &loaded_model.device;
    let model = loaded_model.model.lock();

    let sessions_lock = SESSIONS.read();
    let sessions = match &*sessions_lock {
        Some(s) => s,
        None => return Err("Error: SESSIONS not initialized. Call init first.".to_string()),
    };

    let mut session_state = match sessions.get_mut(session_id) {
        Some(s) => s,
        None => return Err(format!("Error: Session {} not found", session_id)),
    };

    let im_start_id = match tokenizer.token_to_id("<|im_start|>") {
        Some(id) => id,
        None => return Err("Error: Missing <|im_start|> in tokenizer".to_string()),
    };
    let im_end_id = match tokenizer.token_to_id("<|im_end|>") {
        Some(id) => id,
        None => return Err("Error: Missing <|im_end|> in tokenizer".to_string()),
    };
    let newline_id = tokenizer.token_to_id("\n").unwrap_or(198); // Common ID for \n

    let user_tokens = match tokenizer.encode(prompt, false) {
        Ok(t) => t.get_ids().to_vec(),
        Err(e) => return Err(format!("Error encoding prompt: {}", e)),
    };

    session_state.tokens.push(im_start_id);
    match tokenizer.encode("user", false) {
        Ok(t) => session_state.tokens.extend(t.get_ids()),
        Err(e) => return Err(format!("Error encoding 'user' tag: {}", e)),
    };
    session_state.tokens.push(newline_id);
    session_state.tokens.extend(user_tokens);
    session_state.tokens.push(im_end_id);
    session_state.tokens.push(newline_id);
    session_state.tokens.push(im_start_id);
    match tokenizer.encode("assistant", false) {
        Ok(t) => session_state.tokens.extend(t.get_ids()),
        Err(e) => return Err(format!("Error encoding 'assistant' tag: {}", e)),
    };
    session_state.tokens.push(newline_id);

    let generation_limit = if max_gen_len > 0 {
        max_gen_len
    } else {
        init_config.max_gen_len as u32
    };

    for _ in 0..generation_limit {
        let num_tokens_before = session_state.tokens.len();
        let logits = match &*model {
            EngineVariant::Burn(m) => m.forward(&mut session_state)?,
            EngineVariant::ExecuTorch(m) => m.forward(&mut session_state)?,
            EngineVariant::LlamaCpp(m) => m.forward(&mut session_state)?,
            EngineVariant::Speculative(m) => m.forward(&mut session_state)?,
        };

        let next_token_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "Error: Logits are empty".to_string())?;

        if next_token_id == im_end_id {
            break;
        }

        session_state.tokens.push(next_token_id);

        let new_text = tokenizer.decode(&session_state.tokens[num_tokens_before..], true).unwrap_or_default();
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
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;

        let global_lock = GLOBAL_MODEL.read();
        let loaded_model = global_lock.as_ref().unwrap();
        let tokenizer = &loaded_model.tokenizer;
        let model = loaded_model.model.lock();

        // Create a moderately long prompt to make the difference noticeable
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(60);
        let tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
        let n = tokens.len();
        println!("Benchmark sequence length: {} tokens", n);

        // 1. Prefill to get initial cache
        let mut session_cached = EngineSession { tokens: tokens.clone(), offset: 0, state: None, last_accepted_count: 0, is_speculative: false };
        if let EngineVariant::Burn(m) = &*model {
            let _ = m.forward(&mut session_cached).unwrap();
        }

        // 2. Measure "With KV Cache" (Generating 1 token)
        let next_token_id = 1234u32;
        session_cached.tokens.push(next_token_id);

        let start_cached = std::time::Instant::now();
        let iterations = 10;
        
        if let EngineVariant::Burn(m) = &*model {
            for _ in 0..iterations {
                let _ = m.forward(&mut session_cached).unwrap();
                session_cached.tokens.push(1234);
            }
        }
        let duration_cached = start_cached.elapsed() / iterations;

        // 3. Measure "Without KV Cache" (Processing N+1 tokens from scratch)
        let mut tokens_full = tokens.clone();
        tokens_full.push(next_token_id);

        let start_full = std::time::Instant::now();
        if let EngineVariant::Burn(m) = &*model {
            for _ in 0..iterations {
                let mut fresh_session = EngineSession { tokens: tokens_full.clone(), offset: 0, state: None, last_accepted_count: 0, is_speculative: false };
                let _ = m.forward(&mut fresh_session).unwrap();
            }
        }
        let duration_full = start_full.elapsed() / iterations;

        let speedup = duration_full.as_secs_f64() / duration_cached.as_secs_f64();
        println!("Latency WITH KV Cache:    {:?}", duration_cached);
        println!("Latency WITHOUT KV Cache: {:?}", duration_full);
        println!("Measured Speedup:         {:.2}x", speedup);

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
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;
        let session_id = init_session();

        // Turn 1
        let prompt1 = "My name is Alice. Remember that.";
        let sink1 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt1, 256, &sink1).unwrap();

        // Turn 2
        let prompt2 = "What is my name?";
        let sink2 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt2, 256, &sink2).unwrap();

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
        let res = if model == "qwen3" {
            download_qwen3_0_6b(cache_dir.to_string()).await
        } else if model == "gguf" {
            download_qwen_gguf(cache_dir.to_string()).await
        } else {
            download_qwen2_5_0_5b_instruct(cache_dir.to_string()).await
        };
        if res != "Success" {
            panic!("Setup failed: {}", res);
        }
        cache_dir.to_string()
    }

    #[tokio::test]
    async fn test_one_plus_one_gguf() {
        let cache_dir = setup_test("./tmp/testing_gguf", "gguf").await;
        
        let init_res = init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
                backend: BackendType::LlamaCpp, speculate_tokens: 0,
            },
        )
        .await;
        assert_eq!(init_res, "Success");

        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";
        let sink = TestSink(Mutex::new(String::new()));
        
        let result = generate_response(&session_id, prompt, 256, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert!(response.trim().contains("2"));
    }

    #[tokio::test]
    async fn test_multi_turn_gguf() {
        let cache_dir = setup_test("./tmp/testing_gguf_multi", "gguf").await;
        
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
                backend: BackendType::LlamaCpp, speculate_tokens: 0,
            },
        )
        .await;

        let session_id = init_session();
        
        let prompt1 = "My name is Bob. Remember that.";
        let sink1 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt1, 256, &sink1).unwrap();

        let prompt2 = "What is my name?";
        let sink2 = TestSink(Mutex::new(String::new()));
        generate_response(&session_id, prompt2, 256, &sink2).unwrap();

        let response2 = sink2.0.lock().clone();
        println!("Prompt 2: {}", prompt2);
        println!("Response 2: {}", response2);

        assert!(response2.contains("Bob"));
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
                backend: BackendType::ExecuTorch, speculate_tokens: 0,
            },
        )
        .await;

        let session_id = init_session();
        let prompt = "what is the capital of China? /no_think";
        let sink = TestSink(Mutex::new(String::new()));
        
        // 4. Verify the integrated generate_response works with ExecuTorch
        let result = generate_response(&session_id, prompt, 256, &sink);

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
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, 256, &sink);

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
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;
        let session_id = init_session();
        let prompt = "what is the capital of china? only answer the city name /no_think";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, 256, &sink);

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
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;
        let session_id = init_session();
        let prompt = "what is 1+1? only answer with numbers";

        let sink = TestSink(Mutex::new(String::new()));
        let result = generate_response(&session_id, prompt, 256, &sink);

        assert!(result.is_ok());
        let response = sink.0.lock().clone();
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        assert_eq!(response.trim(), "2");
    }

        #[tokio::test]
        async fn test_speculative_decoding_gguf() {
            let cache_dir = setup_test("./tmp/testing_gguf_spec", "gguf").await;
            
            let init_res = init(
                cache_dir,
                InitConfig {
                    vocab_shards: 1,
                    max_gen_len: 128,
                    use_executorch: false,
                    backend: BackendType::LlamaCpp,
                    speculate_tokens: 4,
                },
            )
            .await;
            assert_eq!(init_res, "Success");
    
            let session_id = init_session();
            let prompt = "what is 1+1? only answer with numbers";
            let sink = TestSink(Mutex::new(String::new()));
            
            let result = generate_response(&session_id, prompt, 128, &sink);
    
            assert!(result.is_ok());
            let response = sink.0.lock().clone();
            println!("Prompt: {}", prompt);
            println!("Response: {}", response);
    
            assert!(response.trim().contains("2"));
            
            // Verify accepted count
            let sessions_lock = SESSIONS.read();
            let sessions = sessions_lock.as_ref().unwrap();
            let session = sessions.get(&session_id).unwrap();
            println!("Last accepted count: {}", session.last_accepted_count);
            // Since we use dummy pad tokens, it should be 1 (the bonus token)
            assert_eq!(session.last_accepted_count, 1);
        }

    #[tokio::test]
    async fn test_openai_api_interface() {
        let cache_dir = setup_test("./tmp/testing_openai", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;

        let request = CreateChatCompletionRequest {
            model: "snowglobe".to_string(),
            messages: vec![
                ChatCompletionRequestMessage::User(async_openai::types::chat::ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("what is 1+1? only answer with numbers".to_string()),
                    name: None,
                }),
            ],
            max_completion_tokens: Some(10),
            ..Default::default()
        };

        let response = create_chat_completion(request).await.unwrap();
        match response {
            ChatCompletionOutput::Single(res) => {
                let text = res.choices[0].message.content.as_ref().unwrap();
                println!("OpenAI Response: {}", text);
                assert!(text.contains("2"));
            }
            _ => panic!("Expected single response"),
        }
    }

    #[tokio::test]
    async fn test_openai_api_function_calling() {
        let cache_dir = setup_test("./tmp/testing_openai_func", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;

        // We'll mock a prompt that specifically asks for a tool call in the expected format if possible,
        // or we just check if the parser works when the model outputs it.
        // To make it deterministic for the test, we can try to use a very specific prompt.
        let prompt = "Call the function 'get_weather' with argument 'city' as 'London'. Use the format: <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n</tool_call>";
        
        let request = CreateChatCompletionRequest {
            model: "snowglobe".to_string(),
            messages: vec![
                ChatCompletionRequestMessage::User(async_openai::types::chat::ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                    name: None,
                }),
            ],
            max_completion_tokens: Some(100),
            ..Default::default()
        };

        let response = create_chat_completion(request).await.unwrap();
        match response {
            ChatCompletionOutput::Single(res) => {
                let tool_calls = res.choices[0].message.tool_calls.as_ref().expect("Expected tool calls");
                assert_eq!(tool_calls.len(), 1);
                match &tool_calls[0] {
                    ChatCompletionMessageToolCalls::Function(tc) => {
                        assert_eq!(tc.function.name, "get_weather");
                        assert!(tc.function.arguments.contains("London"));
                    }
                    _ => panic!("Expected function tool call"),
                }
            }
            _ => panic!("Expected single response"),
        }
    }

    #[tokio::test]
    async fn test_openai_api_streaming() {
        let cache_dir = setup_test("./tmp/testing_openai_stream", "qwen2.5").await;
        init(
            cache_dir,
            InitConfig {
                vocab_shards: 1,
                max_gen_len: 256,
                use_executorch: false,
                backend: BackendType::Burn, speculate_tokens: 0,
            },
        )
        .await;

        let request = CreateChatCompletionRequest {
            model: "snowglobe".to_string(),
            messages: vec![
                ChatCompletionRequestMessage::User(async_openai::types::chat::ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("what is 1+1? only answer with numbers".to_string()),
                    name: None,
                }),
            ],
            max_completion_tokens: Some(10),
            stream: Some(true),
            ..Default::default()
        };

        let response = create_chat_completion(request).await.unwrap();
        match response {
            ChatCompletionOutput::Stream(mut stream) => {
                let mut full_text = String::new();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    if let Some(content) = &chunk.choices[0].delta.content {
                        full_text.push_str(content);
                    }
                }
                println!("Streaming Response: {}", full_text);
                assert!(full_text.contains("2"));
            }
            _ => panic!("Expected streaming response"),
        }
    }

    #[tokio::test]
    async fn test_qwen_parser_direct() {
        let parser = QwenParser::new();
        let text = "Certainly! <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n</tool_call>";
        let (cleaned, tool_calls) = parser.parse_complete(text).await.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(cleaned.contains("Certainly!"));
    }
    }
