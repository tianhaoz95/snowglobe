use crate::frb_generated::StreamSink;
use snowglobe::{self};

pub async fn init_engine(cache_dir: String, vocab_shards: u32) -> String {
    snowglobe::init(cache_dir, vocab_shards as usize).await
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

pub fn generate_response(session_id: String, prompt: String, sink: StreamSink<String>) {
    let _ = snowglobe::generate_response(&session_id, &prompt, FrbSink(sink));
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
