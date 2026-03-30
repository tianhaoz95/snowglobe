fn main() {
    println!("cargo:rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    println!("cargo:rerun-if-changed=src/adapter/executorch_adapter.cpp");
    println!("cargo:rerun-if-changed=src/adapter/executorch_adapter.h");
    
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

    if let Ok(libs_dir) = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR") {
        let libs_path = std::path::Path::new(&libs_dir);

        // If we are cross-compiling for Android, we might have architecture-specific subdirectories
        let abi_path = if target.contains("aarch64-linux-android") {
            Some(libs_path.join("arm64-v8a"))
        } else if target.contains("x86_64-linux-android") {
            Some(libs_path.join("x86_64"))
        } else if target.contains("i686-linux-android") {
            Some(libs_path.join("x86"))
        } else if target.contains("armv7-linux-androideabi") {
            Some(libs_path.join("armeabi-v7a"))
        } else {
            None
        };

        let search_base = abi_path.as_deref().unwrap_or(libs_path);

        // --- Core ExecuTorch ---
        println!("cargo:rustc-link-search=native={}", search_base.display());
        println!("cargo:rustc-link-lib=static=executorch");
        println!("cargo:rustc-link-lib=static=executorch_core");

        // --- Extensions ---
        let module_path = search_base.join("extension/module");
        if module_path.exists() {
            println!("cargo:rustc-link-search=native={}", module_path.display());
            println!("cargo:rustc-link-lib=static=extension_module");
        } else if search_base.join("libextension_module.a").exists() {
            println!("cargo:rustc-link-lib=static=extension_module");
        }

        let data_loader_path = search_base.join("extension/data_loader");
        if data_loader_path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                data_loader_path.display()
            );
            println!("cargo:rustc-link-lib=static=extension_data_loader");
        } else if search_base.join("libextension_data_loader.a").exists() {
            println!("cargo:rustc-link-lib=static=extension_data_loader");
        }

        let named_data_map_path = search_base.join("extension/named_data_map");
        if named_data_map_path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                named_data_map_path.display()
            );
            println!("cargo:rustc-link-lib=static=extension_named_data_map");
        } else if search_base.join("libextension_named_data_map.a").exists() {
            println!("cargo:rustc-link-lib=static=extension_named_data_map");
        }

        let flat_tensor_path = search_base.join("extension/flat_tensor");
        if flat_tensor_path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                flat_tensor_path.display()
            );
            println!("cargo:rustc-link-lib=static=extension_flat_tensor");
        } else if search_base.join("libextension_flat_tensor.a").exists() {
            println!("cargo:rustc-link-lib=static=extension_flat_tensor");
        }

        let threadpool_path = search_base.join("extension/threadpool");
        if threadpool_path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                threadpool_path.display()
            );
            println!("cargo:rustc-link-lib=static=extension_threadpool");
        } else if search_base.join("libextension_threadpool.a").exists() {
            println!("cargo:rustc-link-lib=static=extension_threadpool");
        }

        // --- Backends & Delegates ---

        // MPS (Apple)
        let mps_path = search_base.join("backends/apple/mps");
        if mps_path.exists() {
            println!("cargo:rustc-link-search=native={}", mps_path.display());
            // Backends DO need whole-archive for static registration
            println!("cargo:rustc-link-lib=static:+whole-archive=mpsdelegate");
        }

        // XNNPACK
        let xnnpack_path = search_base.join("backends/xnnpack");
        if xnnpack_path.exists() || search_base.join("libxnnpack_backend.a").exists() {
            if xnnpack_path.exists() {
                println!("cargo:rustc-link-search=native={}", xnnpack_path.display());
            }
            println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack_backend");

            // Handle sub-libraries (might be in third-party/ or directly in search_base)
            if search_base.join("libXNNPACK.a").exists() {
                println!("cargo:rustc-link-lib=static:+whole-archive=XNNPACK");
                println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack-microkernels-prod");
            } else {
                let xnn_third_party = xnnpack_path.join("third-party/XNNPACK");
                if xnn_third_party.exists() {
                    println!("cargo:rustc-link-search=native={}", xnn_third_party.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=XNNPACK");
                    println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack-microkernels-prod");
                }
            }

            if search_base.join("libpthreadpool.a").exists() {
                println!("cargo:rustc-link-lib=static:+whole-archive=pthreadpool");
            } else {
                let pthreadpool_path = xnnpack_path.join("third-party/pthreadpool");
                if pthreadpool_path.exists() {
                    println!("cargo:rustc-link-search=native={}", pthreadpool_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=pthreadpool");
                }
            }

            if search_base.join("libcpuinfo.a").exists() {
                println!("cargo:rustc-link-lib=static:+whole-archive=cpuinfo");
            } else {
                let cpuinfo_path = xnnpack_path.join("third-party/cpuinfo");
                if cpuinfo_path.exists() {
                    println!("cargo:rustc-link-search=native={}", cpuinfo_path.display());
                    println!("cargo:rustc-link-lib=static:+whole-archive=cpuinfo");
                }
            }
        }

        let kleidiai_path = search_base.join("kleidiai");
        if kleidiai_path.exists() {
            println!("cargo:rustc-link-search=native={}", kleidiai_path.display());
            println!("cargo:rustc-link-lib=static=kleidiai");
        } else if search_base.join("libkleidiai.a").exists() {
            println!("cargo:rustc-link-lib=static=kleidiai");
        }

        // Qualcomm QNN
        let qualcomm_path = search_base.join("backends/qualcomm");
        if qualcomm_path.exists() || search_base.join("libqnn_executorch_backend.a").exists() || search_base.join("libqnn_executorch_backend.so").exists() {
            if qualcomm_path.exists() {
                println!("cargo:rustc-link-search=native={}", qualcomm_path.display());
            }
            if search_base.join("libqnn_executorch_backend.a").exists() || qualcomm_path.join("libqnn_executorch_backend.a").exists() {
                println!("cargo:rustc-link-lib=static:+whole-archive=qnn_executorch_backend");
            } else if search_base.join("libqnn_executorch_backend.so").exists() || qualcomm_path.join("libqnn_executorch_backend.so").exists() {
                println!("cargo:rustc-link-lib=dylib=qnn_executorch_backend");
            }
        }

        // --- Kernels ---
        let portable_path = search_base.join("kernels/portable");
        if portable_path.exists() || search_base.join("libportable_kernels.a").exists() {
            if portable_path.exists() {
                println!("cargo:rustc-link-search=native={}", portable_path.display());
            }
            // Kernels DO need whole-archive for static registration
            println!("cargo:rustc-link-lib=static:+whole-archive=portable_kernels");
            println!("cargo:rustc-link-lib=static:+whole-archive=portable_ops_lib");
            // Link quantized kernels if available
            if portable_path.join("libquantized_kernels.a").exists() || search_base.join("libquantized_kernels.a").exists() {
                println!("cargo:rustc-link-lib=static:+whole-archive=quantized_kernels");
                println!("cargo:rustc-link-lib=static:+whole-archive=quantized_ops_lib");
            }
        }

        let optimized_path = search_base.join("kernels/optimized");
        if optimized_path.exists() || search_base.join("liboptimized_kernels.a").exists() {
            if optimized_path.exists() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    optimized_path.display()
                );
            }
            println!("cargo:rustc-link-lib=static:+whole-archive=optimized_kernels");
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
        let cpuinfo_include = base_dir.join("backends/xnnpack/third-party/cpuinfo/include");

        // Try to find the actual ExecuTorch source for headers
        let project_root = std::env::current_dir().unwrap().parent().unwrap().to_path_buf();
        let et_root = project_root.join("third_party/executorch");
        
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("src/adapter/executorch_adapter.cpp")
            .include("src/adapter")
            .include(&src_include)
            .include(&schema_include)
            .include(&c10_include)
            .include(&mps_include)
            .include(&pthreadpool_include)
            .include(&cpuinfo_include)
            .include(&project_root.join("third_party")) // Allow <executorch/...>
            .include(&et_root)
            .include(&et_root.join("runtime/core/portable_type/c10")) // Allow <c10/...>
            .include(&et_root.join("backends/xnnpack/third-party/pthreadpool/include"))
            .include(&et_root.join("backends/xnnpack/third-party/cpuinfo/include"))
            .define("C10_USING_CUSTOM_GENERATED_MACROS", None)
            .flag("-std=c++17")
            .flag("-fno-aligned-allocation")
            .flag("-fno-exceptions");

        if target.contains("apple-darwin") {
            build.flag("-mmacosx-version-min=26.0");
        }

        build.compile("executorch_adapter");
        println!("cargo:rustc-cfg=has_executorch");
    }
}
