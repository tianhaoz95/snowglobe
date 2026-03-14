use crate::frb_generated::StreamSink;
use snowglobe::{self};

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
}

pub async fn init_engine(cache_dir: String, config: InitConfig) -> String {
    snowglobe::init(
        cache_dir,
        snowglobe::InitConfig {
            vocab_shards: config.vocab_shards as usize,
            max_gen_len: config.max_gen_len as usize,
            use_executorch: config.use_executorch,
            backend: config.backend.into(),
        },
    )
    .await
}

pub struct ModelInfo {
    pub param_count: u64,
    pub model_size_bytes: u64,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
}

impl From<snowglobe::model::ModelInfo> for ModelInfo {
    fn from(info: snowglobe::model::ModelInfo) -> Self {
        Self {
            param_count: info.param_count as u64,
            model_size_bytes: info.model_size_bytes as u64,
            num_layers: info.num_layers as u32,
            hidden_size: info.hidden_size as u32,
            vocab_size: info.vocab_size as u32,
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

struct FrbSink(StreamSink<String>);

impl snowglobe::StreamSink<String> for FrbSink {
    fn add(&self, value: String) -> bool {
        self.0.add(value).is_ok()
    }
}

pub fn generate_response(
    session_id: String,
    prompt: String,
    max_gen_len: u32,
    sink: StreamSink<String>,
) {
    let _ = snowglobe::generate_response(&session_id, &prompt, max_gen_len, FrbSink(sink));
}

pub fn experimental_completion_with_pte(pte_path: String, prompt: String) -> String {
    snowglobe::experimental_completion_with_pte(&pte_path, &prompt).unwrap_or_else(|e| e)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
