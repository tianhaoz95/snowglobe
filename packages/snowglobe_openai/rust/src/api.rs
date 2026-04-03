use snowglobe::{self, ChatCompletionOutput};
use crate::frb_generated::StreamSink;
use futures_util::StreamExt;
use async_openai::types::chat::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest};

pub enum BackendType {
    Burn,
    ExecuTorch,
    LlamaCpp,
}

impl From<BackendType> for snowglobe::BackendType {
    fn from(val: BackendType) -> Self {
        match val {
            BackendType::Burn => snowglobe::BackendType::Burn,
            BackendType::ExecuTorch => snowglobe::BackendType::ExecuTorch,
            BackendType::LlamaCpp => snowglobe::BackendType::LlamaCpp,
        }
    }
}

pub enum HardwareTarget {
    Auto,
    Cpu,
    Gpu,
    Npu,
}

impl From<HardwareTarget> for snowglobe::model::HardwareTarget {
    fn from(val: HardwareTarget) -> Self {
        match val {
            HardwareTarget::Auto => snowglobe::model::HardwareTarget::Auto,
            HardwareTarget::Cpu => snowglobe::model::HardwareTarget::Cpu,
            HardwareTarget::Gpu => snowglobe::model::HardwareTarget::Gpu,
            HardwareTarget::Npu => snowglobe::model::HardwareTarget::Npu,
        }
    }
}

pub struct InitConfig {
    pub vocab_shards: u32,
    pub max_gen_len: u32,
    pub use_executorch: bool,
    pub backend: BackendType,
    pub hardware: HardwareTarget,
    pub speculate_tokens: u32,
}

pub async fn init_engine(cache_dir: String, config: InitConfig) -> String {
    snowglobe::init(
        cache_dir,
        snowglobe::InitConfig {
            vocab_shards: config.vocab_shards as usize,
            max_gen_len: config.max_gen_len as usize,
            use_executorch: config.use_executorch,
            backend: config.backend.into(),
            hardware: config.hardware.into(),
            speculate_tokens: config.speculate_tokens as usize,
        },
    )
    .await
}

pub async fn chat_completion(request_json: String) -> Result<String, String> {
    let request: CreateChatCompletionRequest = serde_json::from_str(&request_json).map_err(|e| e.to_string())?;
    let output = snowglobe::create_chat_completion(request).await?;
    match output {
        ChatCompletionOutput::Single(res) => serde_json::to_string(&res).map_err(|e| e.to_string()),
        ChatCompletionOutput::Stream(_) => Err("Expected single response, got stream".to_string()),
    }
}

pub async fn chat_completion_stream(request_json: String, sink: StreamSink<String>) -> Result<(), String> {
    let request: CreateChatCompletionRequest = serde_json::from_str(&request_json).map_err(|e| e.to_string())?;
    let output = snowglobe::create_chat_completion(request).await?;
    match output {
        ChatCompletionOutput::Single(_) => Err("Expected stream, got single response".to_string()),
        ChatCompletionOutput::Stream(mut stream) => {
            while let Some(chunk_res) = stream.next().await {
                match chunk_res {
                    Ok(chunk) => {
                        let chunk_json = serde_json::to_string(&chunk).map_err(|e| e.to_string())?;
                        sink.add(chunk_json).map_err(|e| e.to_string())?;
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}

pub struct ModelInfo {
    pub name: String,
    pub param_count: u64,
    pub model_size_bytes: u64,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub runner: String,
    pub backend: String,
}

impl From<snowglobe::model::ModelInfo> for ModelInfo {
    fn from(info: snowglobe::model::ModelInfo) -> Self {
        Self {
            name: info.name,
            param_count: info.param_count as u64,
            model_size_bytes: info.model_size_bytes as u64,
            num_layers: info.num_layers as u32,
            hidden_size: info.hidden_size as u32,
            vocab_size: info.vocab_size as u32,
            runner: info.runner,
            backend: info.backend,
        }
    }
}

pub fn get_model_info() -> Option<ModelInfo> {
    snowglobe::get_model_info().map(Into::into)
}

pub fn check_backend() -> String {
    snowglobe::check_backend()
}

pub fn init_session() -> String {
    snowglobe::init_session()
}

pub fn get_last_accepted_count(session_id: String) -> u32 {
    let sessions_lock = snowglobe::SESSIONS.read();
    if let Some(sessions) = sessions_lock.as_ref() {
        if let Some(session) = sessions.get(&session_id) {
            return session.last_accepted_count as u32;
        }
    }
    0
}

pub async fn generate_response(
    session_id: String,
    prompt: String,
    max_gen_len: u32,
    sink: StreamSink<String>,
) {
    struct SinkWrapper(StreamSink<String>);
    impl snowglobe::StreamSink<String> for SinkWrapper {
        fn add(&self, value: String) -> bool {
            self.0.add(value).is_ok()
        }
    }

    if let Err(e) = snowglobe::generate_response(&session_id, &prompt, max_gen_len, SinkWrapper(sink)) {
        android_log(&format!("generate_response error: {}", e));
    }
}

#[cfg(target_os = "android")]
fn android_log(msg: &str) {
    use std::ffi::{CString, c_char};
    let tag = CString::new("SNOWGLOBE_RS").unwrap_or_default();
    let msg_c = CString::new(msg).unwrap_or_default();
    unsafe extern "C" {
        fn __android_log_write(prio: i32, tag: *const c_char, text: *const c_char) -> i32;
    }
    unsafe { __android_log_write(4, tag.as_ptr(), msg_c.as_ptr()); }
}

#[cfg(not(target_os = "android"))]
fn android_log(msg: &str) {
    eprintln!("[SNOWGLOBE_RS] {}", msg);
}
