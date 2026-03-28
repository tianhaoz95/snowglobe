use crate::model::runner::{EngineSession, ModelRunner};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;
use once_cell::sync::OnceCell;

// Global LlamaBackend instance because llama_cpp_2 requires it to be a singleton-like,
// and calling LlamaBackend::init() more than once results in an error.
static LLAMA_BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

pub struct LlamaCppRunner {
    pub model: LlamaModel,
}

struct SafeLlamaContext(llama_cpp_2::context::LlamaContext<'static>);
unsafe impl Send for SafeLlamaContext {}
unsafe impl Sync for SafeLlamaContext {}

impl ModelRunner for LlamaCppRunner {
    fn load(path: &Path, config: &crate::InitConfig) -> Result<Box<Self>, String> {
        let backend = LLAMA_BACKEND.get_or_try_init(|| {
            LlamaBackend::init().map_err(|e| format!("Failed to init backend: {}", e))
        })?;
        
        let mut model_params = LlamaModelParams::default();

        match config.hardware {
            crate::model::HardwareTarget::Cpu => {
                model_params = model_params.with_n_gpu_layers(0);
            }
            crate::model::HardwareTarget::Gpu => {
                model_params = model_params.with_n_gpu_layers(99);
                // Optionally explicitly select a GPU device if multiple exist
            }
            crate::model::HardwareTarget::Npu => {
                model_params = model_params.with_n_gpu_layers(99);
                // Look for NPU device
                let dev_count = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
                for i in 0..dev_count {
                    let dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(i) };
                    let name_ptr = unsafe { llama_cpp_sys_2::ggml_backend_dev_name(dev) };
                    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) }.to_string_lossy();
                    if name.to_lowercase().contains("qnn") || name.to_lowercase().contains("npu") {
                        model_params = model_params.with_devices(&[i])
                            .map_err(|e| format!("Failed to set NPU device: {:?}", e))?;
                        break;
                    }
                }
            }
            crate::model::HardwareTarget::Auto => {
                model_params = model_params.with_n_gpu_layers(99);
            }
        }
        
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        Ok(Box::new(Self { model }))
    }

    fn forward_all(&self, session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String> {
        let backend = LLAMA_BACKEND.get().ok_or("LlamaBackend not initialized")?;

        if session.state.is_none() {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(4096))
                .with_n_seq_max(2); // Support at least 2 sequences for speculative decoding
            let ctx = self.model
                .new_context(backend, ctx_params)
                .map_err(|e| format!("Failed to create context: {}", e))?;
            
            // Unsafe lifetime extension because the model outlives the session
            let ctx_static: llama_cpp_2::context::LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            session.state = Some(Box::new(SafeLlamaContext(ctx_static)));
        }

        let ctx_wrapper = session.state.as_mut().unwrap().downcast_mut::<SafeLlamaContext>().unwrap();
        let ctx = &mut ctx_wrapper.0;

        let start = session.offset;
        let num_new = session.tokens.len() - start;
        if num_new == 0 {
            return Err("No new tokens".to_string());
        }

        let mut batch = LlamaBatch::new(num_new, 1);
        for (i, &token) in session.tokens[start..].iter().enumerate() {
            let pos = (start + i) as i32;
            let seq_id = if session.is_speculative { 1 } else { 0 };
            batch.add(llama_cpp_2::token::LlamaToken::new(token as i32), pos, &[seq_id], true)
                .map_err(|e| format!("Failed to add to batch: {}", e))?;
        }

        ctx.decode(&mut batch).map_err(|e| format!("Decode failed: {}", e))?;

        // Extract logits for all tokens
        let mut all_logits = Vec::with_capacity(num_new);
        for i in 0..num_new {
            let logits = ctx.get_logits_ith(i as i32);
            all_logits.push(logits.to_vec());
        }
        
        session.offset += num_new;
        
        Ok(all_logits)
    }

    fn forward(&self, session: &mut EngineSession) -> Result<Vec<f32>, String> {
        let all_logits = self.forward_all(session)?;
        Ok(all_logits.into_iter().last().ok_or("No logits returned")?)
    }

    fn prepare_speculative_verification(&self, session: &mut EngineSession) -> Result<(), String> {
        if let Some(state) = &mut session.state {
            let ctx_wrapper = state.downcast_mut::<SafeLlamaContext>().unwrap();
            let ctx = &mut ctx_wrapper.0;
            
            // Copy sequence 0 to sequence 1 for speculative verification
            ctx.copy_kv_cache_seq(0, 1, None, None)
                .map_err(|e| format!("Failed to copy speculative KV cache: {:?}", e))?;
        }
        Ok(())
    }

    fn cleanup_speculative_verification(&self, session: &mut EngineSession) -> Result<(), String> {
        if let Some(state) = &mut session.state {
            let ctx_wrapper = state.downcast_mut::<SafeLlamaContext>().unwrap();
            let ctx = &mut ctx_wrapper.0;
            
            // Clear sequence 1 after speculative verification
            ctx.clear_kv_cache_seq(Some(1), None, None)
                .map_err(|e| format!("Failed to clear speculative KV cache: {:?}", e))?;
        }
        Ok(())
    }

    fn backend_name(&self) -> String {
        unsafe {
            let model_ptr = self.model.as_ptr();
            // Try to get the device of the first tensor (which is usually on the main backend)
            // Or just check if any GPU layers are offloaded
            let n_gpu_layers = llama_cpp_sys_2::llama_n_layer(model_ptr);
            if n_gpu_layers > 0 {
                // In modern llama.cpp, we can check backend devices
                let dev_count = llama_cpp_sys_2::ggml_backend_dev_count();
                for i in 0..dev_count {
                    let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);
                    let name_ptr = llama_cpp_sys_2::ggml_backend_dev_name(dev);
                    let name = std::ffi::CStr::from_ptr(name_ptr).to_string_lossy();
                    // If we find QNN/NPU or Vulkan/Metal, return it
                    if name.to_lowercase().contains("qnn") || 
                       name.to_lowercase().contains("npu") || 
                       name.to_lowercase().contains("vulkan") || 
                       name.to_lowercase().contains("metal") {
                        return name.into_owned();
                    }
                }
            }
        }
        "CPU".to_string()
    }
}
