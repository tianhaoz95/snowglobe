use snowglobe::{self, generate};

pub fn greet(name: String) -> String {
    generate(name)
}

pub fn init_engine() {
    snowglobe::init();
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
