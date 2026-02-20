fn main() {
    println!("cargo:rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    if let Ok(libs_dir) = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR") {
        // Core ExecuTorch Search Path
        println!("cargo:rustc-link-search=native={}", libs_dir);
        
        // MPS Backend (macOS)
        println!("cargo:rustc-link-search=native={}/backends/apple/mps", libs_dir);
        println!("cargo:rustc-link-lib=static:+whole-archive=mpsdelegate");
        
        // Kernels
        println!("cargo:rustc-link-search=native={}/kernels/portable", libs_dir);
        println!("cargo:rustc-link-lib=static:+whole-archive=portable_kernels");
        println!("cargo:rustc-link-lib=static:+whole-archive=portable_ops_lib");

        // Optimized Kernels (CPU)
        println!("cargo:rustc-link-search=native={}/kernels/optimized", libs_dir);
        println!("cargo:rustc-link-lib=static:+whole-archive=optimized_kernels");
        
        // Required Apple Frameworks
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
