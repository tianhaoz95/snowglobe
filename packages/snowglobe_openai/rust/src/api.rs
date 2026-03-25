use snowglobe::{self, ChatCompletionOutput};
use crate::frb_generated::StreamSink;
use futures_util::StreamExt;
use async_openai::types::chat::CreateChatCompletionRequest;

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

pub struct InitConfig {
    pub vocab_shards: u32,
    pub max_gen_len: u32,
    pub use_executorch: bool,
    pub backend: BackendType,
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
