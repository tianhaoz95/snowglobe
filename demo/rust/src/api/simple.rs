use crate::frb_generated::StreamSink;
use snowglobe::{self};

pub enum BackendType {
    Burn,
    LiteRT,
    LlamaCpp,
}

impl From<BackendType> for snowglobe::model::BackendType {
    fn from(val: BackendType) -> Self {
        match val {
            BackendType::Burn => snowglobe::model::BackendType::Burn,
            BackendType::LiteRT => snowglobe::model::BackendType::LiteRT,
            BackendType::LlamaCpp => snowglobe::model::BackendType::LlamaCpp,
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
            backend: config.backend.into(),
            hardware: config.hardware.into(),
            speculate_tokens: config.speculate_tokens as usize,
        },
    )
    .await
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

pub struct ModelInfo {
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

use async_openai::types::chat::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest};
use futures_util::StreamExt;

pub async fn generate_response(
    _session_id: String,
    prompt: String,
    max_gen_len: u32,
    sink: StreamSink<String>,
) {
    let request = CreateChatCompletionRequest {
        model: "snowglobe".to_string(),
        messages: vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(prompt),
                name: None,
            },
        )],
        max_completion_tokens: Some(max_gen_len),
        stream: Some(true),
        ..Default::default()
    };

    if let Ok(snowglobe::ChatCompletionOutput::Stream(mut stream)) = snowglobe::create_chat_completion(request).await {
        while let Some(chunk_res) = stream.next().await {
            if let Ok(chunk) = chunk_res {
                if let Some(content) = &chunk.choices[0].delta.content {
                    let _ = sink.add(content.clone());
                }
            }
        }
    }
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
