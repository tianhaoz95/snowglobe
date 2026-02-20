fn main() {
    println!("cargo:rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    if let Ok(libs_dir) = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR") {
        let target = std::env::var("TARGET").unwrap_or_default();
        let libs_path = std::path::Path::new(&libs_dir);
        
        // If we are cross-compiling for Android, we might have architecture-specific subdirectories
        let abi_path = if target.contains("aarch64-linux-android") {
            Some(libs_path.join("arm64-v8a"))
        } else if target.contains("x86_64-linux-android") {
            Some(libs_path.join("x86_64"))
        } else if target.contains("armv7-linux-androideabi") {
            Some(libs_path.join("armeabi-v7a"))
        } else {
            None
        };

                let search_base = abi_path.as_deref().unwrap_or(libs_path);
        
                // --- Core ExecuTorch ---
                println!("cargo:rustc-link-search=native={}", search_base.display());
                println!("cargo:rustc-link-lib=static:+whole-archive=executorch");
                println!("cargo:rustc-link-lib=static:+whole-archive=executorch_core");
        
                // --- Extensions ---
                let module_path = search_base.join("extension/module");
                if module_path.exists() {
                    println!("cargo:rustc-link-search=native={}", module_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=extension_module");
                }
        
                let data_loader_path = search_base.join("extension/data_loader");
                if data_loader_path.exists() {
                    println!("cargo:rustc-link-search=native={}", data_loader_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=extension_data_loader");
                }
        
                let named_data_map_path = search_base.join("extension/named_data_map");
                if named_data_map_path.exists() {
                    println!("cargo:rustc-link-search=native={}", named_data_map_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=extension_named_data_map");
                }
        
                let flat_tensor_path = search_base.join("extension/flat_tensor");
                if flat_tensor_path.exists() {
                    println!("cargo:rustc-link-search=native={}", flat_tensor_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=extension_flat_tensor");
                }
        
                let threadpool_path = search_base.join("extension/threadpool");
                if threadpool_path.exists() {
                    println!("cargo:rustc-link-search=native={}", threadpool_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=extension_threadpool");
                }
        
                // --- Backends & Delegates ---
                
                // MPS (Apple)
                let mps_path = search_base.join("backends/apple/mps");
                if mps_path.exists() {
                    println!("cargo:rustc-link-search=native={}", mps_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=mpsdelegate");
                }
        
                // XNNPACK
                let xnnpack_path = search_base.join("backends/xnnpack");
                if xnnpack_path.exists() {
                    println!("cargo:warning=✅ XNNPACK PATH FOUND AT: {}", xnnpack_path.display());
                    println!("cargo:rustc-link-search=native={}", xnnpack_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack_backend");
                    
                    let xnn_third_party = xnnpack_path.join("third-party/XNNPACK");
                    if xnn_third_party.exists() {
                        println!("cargo:rustc-link-search=native={}", xnn_third_party.display());
                        println!("cargo:rustc-link-lib=static:+whole-archive=XNNPACK");
                        println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack-microkernels-prod");
                    }
        
                    let pthreadpool_path = xnnpack_path.join("third-party/pthreadpool");
                    if pthreadpool_path.exists() {
                        println!("cargo:rustc-link-search=native={}", pthreadpool_path.display());
                        println!("cargo:rustc-link-lib=static=pthreadpool");
                    }
        
                    let cpuinfo_path = xnnpack_path.join("third-party/cpuinfo");
                    if cpuinfo_path.exists() {
                        println!("cargo:rustc-link-search=native={}", cpuinfo_path.display());
                        println!("cargo:rustc-link-lib=static=cpuinfo");
                    }
                } else {
                    println!("cargo:warning=❌ XNNPACK PATH MISSING! Looked for: {}", xnnpack_path.display());
                }
        
                let kleidiai_path = search_base.join("kleidiai");
                if kleidiai_path.exists() {
                    println!("cargo:rustc-link-search=native={}", kleidiai_path.display());
                    println!("cargo:rustc-link-lib=static=kleidiai");
                }
        
                // --- Kernels ---
                let portable_path = search_base.join("kernels/portable");
                if portable_path.exists() {
                    println!("cargo:rustc-link-search=native={}", portable_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=portable_kernels");
                    println!("cargo:rustc-link-lib=static:+whole-archive=portable_ops_lib");
                    
                    let kernels_util_path = portable_path.join("cpu/util");
                    if kernels_util_path.exists() {
                        println!("cargo:rustc-link-search=native={}", kernels_util_path.display());
                        println!("cargo:rustc-link-lib=static=kernels_util_all_deps");
                    }
                }
        
                let optimized_path = search_base.join("kernels/optimized");
                if optimized_path.exists() {
                    println!("cargo:rustc-link-search=native={}", optimized_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=optimized_kernels");
                }
        
                // Required Apple Frameworks
                if target.contains("apple") {
                    println!("cargo:rustc-link-lib=framework=Foundation");
                    println!("cargo:rustc-link-lib=framework=Metal");
                    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
                    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
                }
        // Required Apple Frameworks
        if target.contains("apple") {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
        }

        // --- C++ Adapter Compilation ---
        let base_dir = libs_path.parent().unwrap_or(libs_path);
        let src_include = base_dir.join("src");
        let schema_include = libs_path.join("schema/include");
        let c10_include = base_dir.join("runtime/core/portable_type/c10");
        let mps_include = base_dir.join("backends/apple/mps");
        let pthreadpool_include = base_dir.join("backends/xnnpack/third-party/pthreadpool/include");
        
        cc::Build::new()
            .cpp(true)
            .file("src/adapter/executorch_adapter.cpp")
            .include("src/adapter")
            .include(&src_include)
            .include(&schema_include)
            .include(&c10_include)
            .include(&mps_include)
            .include(&pthreadpool_include)
            .define("C10_USING_CUSTOM_GENERATED_MACROS", None)
            .flag("-std=c++17")
            .compile("executorch_adapter");
    }
}
