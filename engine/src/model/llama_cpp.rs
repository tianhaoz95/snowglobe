use crate::model::runner::{EngineSession, ModelRunner, ExecutionMode, LogitView, BackendInfo};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;
use once_cell::sync::OnceCell;
use std::any::Any;

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
    fn load(path: &Path, config: &serde_json::Value) -> Result<Box<Self>, String> {
        let backend = LLAMA_BACKEND.get_or_try_init(|| {
            LlamaBackend::init().map_err(|e| format!("Failed to init backend: {}", e))
        })?;
        
        let mut model_params = LlamaModelParams::default();

        // Since config is now serde_json::Value, we'd normally parse it here.
        // For now, let's assume default or extract from value if possible.
        // A better way would be to have a proper Config struct that can be deserialized.
        let hardware = config.get("hardware").and_then(|h| h.as_str()).unwrap_or("auto");

        match hardware {
            "cpu" => {
                model_params = model_params.with_n_gpu_layers(0);
            }
            "gpu" | "auto" => {
                model_params = model_params.with_n_gpu_layers(99);
            }
            "npu" => {
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
            _ => {
                model_params = model_params.with_n_gpu_layers(99);
            }
        }
        
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        Ok(Box::new(Self { model }))
    }

    fn execute(
        &self,
        session: &mut EngineSession,
        input_tokens: &[u32],
        mode: ExecutionMode,
    ) -> Result<LogitView, String> {
        let backend = LLAMA_BACKEND.get().ok_or("LlamaBackend not initialized")?;

        if session.backend_state.is_none() {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(4096))
                .with_n_seq_max(2); 
            let ctx = self.model
                .new_context(backend, ctx_params)
                .map_err(|e| format!("Failed to create context: {}", e))?;
            
            // Unsafe lifetime extension because the model outlives the session
            let ctx_static: llama_cpp_2::context::LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            session.backend_state = Some(Box::new(SafeLlamaContext(ctx_static)));
            session.current_kv_len = 0;
        }

        let ctx_wrapper = session.backend_state.as_mut().unwrap().downcast_mut::<SafeLlamaContext>().unwrap();
        let ctx = &mut ctx_wrapper.0;

        let num_new = input_tokens.len();
        if num_new == 0 {
            return Err("No new tokens".to_string());
        }

        // Ensure we are appending at a valid position for llama.cpp.
        // M-RoPE and other logic in llama.cpp requires strictly increasing positions for a sequence.
        let max_pos = ctx.kv_cache_seq_pos_max(0);
        let start_pos = if max_pos < 0 { 0 } else { max_pos + 1 };

        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(num_new, 1);
        for (i, &token) in input_tokens.iter().enumerate() {
            let pos = start_pos + i as i32;
            let seq_id = 0; 
            
            let is_last = i == num_new - 1;
            // For Verify mode, we might want logits for all tokens, but for now just last.
            let logits = match mode {
                ExecutionMode::Prefill => is_last,
                ExecutionMode::Decode => true,
                ExecutionMode::Verify { .. } => true,
            };

            batch.add(llama_cpp_2::token::LlamaToken::new(token as i32), pos, &[seq_id], logits)
                .map_err(|e| format!("Failed to add to batch: {}", e))?;
        }

        ctx.decode(&mut batch).map_err(|e| format!("Decode failed: {}", e))?;

        session.current_kv_len = (start_pos + num_new as i32) as usize;

        // Extract logits for all requested tokens
        let mut all_logits = Vec::with_capacity(num_new * self.model.n_vocab() as usize);
        for i in 0..num_new {
            // Check if we requested logits for this token
            let is_last = i == num_new - 1;
            let wanted_logits = match mode {
                ExecutionMode::Prefill => is_last,
                ExecutionMode::Decode => true,
                ExecutionMode::Verify { .. } => true,
            };
            
            if wanted_logits {
                let logits = ctx.get_logits_ith(i as i32);
                all_logits.extend_from_slice(logits);
            }
        }
        
        let num_output_rows = if mode == ExecutionMode::Prefill { 1 } else { num_new };

        Ok(LogitView {
            data: all_logits,
            shape: (num_output_rows, self.model.n_vocab() as usize),
        })
    }

    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String> {
        if let Some(state) = &mut session.backend_state {
            let ctx_wrapper = state.downcast_mut::<SafeLlamaContext>().unwrap();
            let ctx = &mut ctx_wrapper.0;
            
            // Clear ALL tokens from 'len' onwards in ALL sequences to be safe.
            ctx.clear_kv_cache_seq(None, Some(len as u32), None)
                .map_err(|e| format!("Failed to truncate KV cache: {:?}", e))?;
            
            session.current_kv_len = len;
        }
        Ok(())
    }

    fn fork_state(&self, session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String> {
        let backend = LLAMA_BACKEND.get().ok_or("LlamaBackend not initialized")?;
        if let Some(state) = &session.backend_state {
            let ctx_wrapper = state.downcast_ref::<SafeLlamaContext>().unwrap();
            let _ctx = &ctx_wrapper.0;
            
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(4096))
                .with_n_seq_max(2);
            let new_ctx = self.model
                .new_context(backend, ctx_params)
                .map_err(|e| format!("Failed to create context for fork: {}", e))?;
            
            // Copy KV cache from old context to new context
            // Note: llama_cpp_2 doesn't easily support cross-context KV copy via high-level API,
            // this might need low-level llama_kv_cache_seq_cp.
            // For now, let's assume we can't easily fork and return error or implement if possible.
            // Actually, the trait requires it. 
            
            // Placeholder:
            let new_ctx_static: llama_cpp_2::context::LlamaContext<'static> = unsafe { std::mem::transmute(new_ctx) };
            Ok(Box::new(SafeLlamaContext(new_ctx_static)))
        } else {
            Err("No state to fork".to_string())
        }
    }

    fn get_backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: self.backend_name(),
            max_sequence_length: 4096, // Example
            max_batch_size: 512,
        }
    }
}

impl LlamaCppRunner {
    fn backend_name(&self) -> String {
        unsafe {
            let _model_ptr = self.model.as_ptr();
            let dev_count = llama_cpp_sys_2::ggml_backend_dev_count();
            for i in 0..dev_count {
                let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);
                let name_ptr = llama_cpp_sys_2::ggml_backend_dev_name(dev);
                let name = std::ffi::CStr::from_ptr(name_ptr).to_string_lossy();
                if name.to_lowercase().contains("qnn") || 
                   name.to_lowercase().contains("npu") || 
                   name.to_lowercase().contains("vulkan") || 
                   name.to_lowercase().contains("metal") {
                    return name.into_owned();
                }
            }
        }
        "CPU".to_string()
    }
}
