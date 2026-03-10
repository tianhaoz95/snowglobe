use crate::model::runner::{EngineSession, ModelRunner};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;

pub struct LlamaCppRunner {
    pub backend: LlamaBackend,
    pub model: LlamaModel,
}

struct SafeLlamaContext(llama_cpp_2::context::LlamaContext<'static>);
unsafe impl Send for SafeLlamaContext {}
unsafe impl Sync for SafeLlamaContext {}

impl ModelRunner for LlamaCppRunner {
    fn load(path: &Path, _config: &crate::InitConfig) -> Result<Box<Self>, String> {
        let backend = LlamaBackend::init().map_err(|e| format!("Failed to init backend: {}", e))?;
        
        let model_params = LlamaModelParams::default().with_n_gpu_layers(99); 
        
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        Ok(Box::new(Self { backend, model }))
    }

    fn forward(&self, session: &mut EngineSession) -> Result<Vec<f32>, String> {
        if session.state.is_none() {
            let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(4096));
            let ctx = self.model
                .new_context(&self.backend, ctx_params)
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
            let is_last = i == num_new - 1;
            batch.add(llama_cpp_2::token::LlamaToken::new(token as i32), (start + i) as i32, &[0], is_last)
                .map_err(|e| format!("Failed to add to batch: {}", e))?;
        }

        ctx.decode(&mut batch).map_err(|e| format!("Decode failed: {}", e))?;

        // Extract logits
        let mut logits_vec = Vec::new();
        // get_logits_ith returns &[f32]
        let logits = ctx.get_logits_ith(batch.n_tokens() - 1);
        logits_vec.extend_from_slice(logits);
        
        session.offset += num_new;
        
        Ok(logits_vec)
    }
}
