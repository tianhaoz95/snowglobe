use snowglobe::{self, generate};

pub fn greet(name: String) -> String {
    generate(name)
}

pub fn init_engine(cache_dir: String) {
    snowglobe::init(cache_dir);
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
