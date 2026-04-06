fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    
    // llama.cpp specific links
    if target.contains("android") {
        println!("cargo:rustc-link-lib=vulkan");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Compile LiteRT adapter
    cc::Build::new()
        .cpp(true)
        .file("src/adapter/litert_adapter.cpp")
        .include("src/adapter")
        // In a real environment, we'd add LiteRT SDK includes here
        .flag("-std=c++17")
        .compile("litert_adapter");
}
