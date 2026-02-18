use snowglobe::{self};
use flutter_rust_bridge::DartFnFuture;

pub async fn init_engine(cache_dir: String) -> String {
    snowglobe::init(cache_dir).await
}

pub fn check_backend() -> String {
    snowglobe::check_backend()
}

pub fn init_session() -> String {
    snowglobe::init_session()
}

pub fn generate_response(session_id: &str, prompt: &str) -> String {
    snowglobe::generate_response(session_id, prompt)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
