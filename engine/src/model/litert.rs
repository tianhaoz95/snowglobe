use std::path::Path;
use crate::model::runner::{ModelRunner, EngineSession, ExecutionMode, LogitView, BackendInfo};
use std::ffi::CString;

unsafe extern "C" {
    fn litert_model_load(model_path: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
    fn litert_model_destroy(model: *mut std::ffi::c_void);
    fn litert_model_prefill(
        model: *mut std::ffi::c_void,
        input_ids: *const i32,
        length: usize,
        output_logits: *mut f32,
    ) -> i32;
    fn litert_model_decode(
        model: *mut std::ffi::c_void,
        input_id: i32,
        pos: i32,
        output_logits: *mut f32,
    ) -> i32;
    fn litert_model_reset_state(model: *mut std::ffi::c_void) -> i32;
}

pub struct LiteRTRunner {
    model_ptr: *mut std::ffi::c_void,
    vocab_size: usize,
}

unsafe impl Send for LiteRTRunner {}
unsafe impl Sync for LiteRTRunner {}

impl LiteRTRunner {
    pub fn new(model_path: &Path, vocab_size: usize) -> Result<Self, String> {
        let path_str = model_path.to_str().ok_or("Invalid path")?;
        let c_path = CString::new(path_str).map_err(|e| e.to_string())?;
        let ptr = unsafe { litert_model_load(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err("Failed to load LiteRT model".to_string());
        }
        Ok(Self {
            model_ptr: ptr,
            vocab_size,
        })
    }
}

impl Drop for LiteRTRunner {
    fn drop(&mut self) {
        unsafe { litert_model_destroy(self.model_ptr) };
    }
}

impl ModelRunner for LiteRTRunner {
    fn load(path: &Path, _config: &serde_json::Value) -> Result<Box<Self>, String> {
        let vocab_size = _config["vocab_size"].as_u64().unwrap_or(256000) as usize;
        Ok(Box::new(Self::new(path, vocab_size)?))
    }

    fn execute(
        &self,
        session: &mut EngineSession,
        input_tokens: &[u32],
        mode: ExecutionMode,
    ) -> Result<LogitView, String> {
        let mut logits = vec![0.0f32; self.vocab_size];
        
        let res = match mode {
            ExecutionMode::Prefill => {
                let input_ids: Vec<i32> = input_tokens.iter().map(|&t| t as i32).collect();
                unsafe {
                    litert_model_prefill(
                        self.model_ptr,
                        input_ids.as_ptr(),
                        input_ids.len(),
                        logits.as_mut_ptr(),
                    )
                }
            }
            ExecutionMode::Decode => {
                let last_token = *input_tokens.last().ok_or("No input tokens")? as i32;
                let pos = session.tokens.len() as i32;
                unsafe {
                    litert_model_decode(
                        self.model_ptr,
                        last_token,
                        pos,
                        logits.as_mut_ptr(),
                    )
                }
            }
            ExecutionMode::Verify { draft_len } => {
                // For LiteRT, we can implement Verify as a sequence of decodes for now
                // Or if the model supports it, a parallel verify signature.
                // Using fallback to Decode loop logic in lib.rs if possible, 
                // but here we must satisfy the trait.
                let mut all_logits = Vec::with_capacity((draft_len + 1) * self.vocab_size);
                let mut current_pos = session.tokens.len();
                
                for &token in input_tokens {
                    let mut step_logits = vec![0.0f32; self.vocab_size];
                    let step_res = unsafe {
                        litert_model_decode(
                            self.model_ptr,
                            token as i32,
                            current_pos as i32,
                            step_logits.as_mut_ptr(),
                        )
                    };
                    if step_res != 0 {
                        return Err(format!("LiteRT decode failed during verify at pos {}", current_pos));
                    }
                    all_logits.extend(step_logits);
                    current_pos += 1;
                }
                
                return Ok(LogitView {
                    data: all_logits,
                    shape: (input_tokens.len(), self.vocab_size),
                });
            }
        };

        if res != 0 {
            return Err(format!("LiteRT execution failed with code {}", res));
        }

        Ok(LogitView {
            data: logits,
            shape: (1, self.vocab_size),
        })
    }

    fn truncate_cache(&self, _session: &mut EngineSession, _len: usize) -> Result<(), String> {
        // LiteRT models with stateful variables handle this via 'pos' input in next decode.
        // Some models might need a reset if they don't support arbitrary 'pos'.
        Ok(())
    }

    fn fork_state(&self, _session: &EngineSession) -> Result<Box<dyn std::any::Any + Send + Sync>, String> {
        Err("Forking state not supported for LiteRT".to_string())
    }

    fn get_backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "LiteRT (Real)".to_string(),
            max_sequence_length: 2048,
            max_batch_size: 1,
        }
    }

    fn model_name(&self) -> String {
        "Gemma 4 LiteRT".to_string()
    }
}
