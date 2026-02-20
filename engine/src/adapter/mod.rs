use std::ffi::{CString, c_char, c_void};

#[repr(C)]
struct ExecuTorchModuleOpaque(c_void);

unsafe extern "C" {
    fn executorch_module_load(pte_path: *const c_char) -> *mut ExecuTorchModuleOpaque;
    fn executorch_module_destroy(module: *mut ExecuTorchModuleOpaque);
    fn executorch_module_forward(
        module: *mut ExecuTorchModuleOpaque,
        input_tokens: *const i64,
        input_len: usize,
        output_logits: *mut f32,
        output_vocab_size: *mut usize,
    ) -> i32;
}

pub struct Module {
    ptr: *mut ExecuTorchModuleOpaque,
}

impl Module {
    pub fn new(pte_path: &str) -> Result<Self, String> {
        let c_path = CString::new(pte_path).map_err(|e| e.to_string())?;
        let ptr = unsafe { executorch_module_load(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err("Failed to load ExecuTorch module".to_string());
        }
        Ok(Self { ptr })
    }

    pub fn forward(&mut self, tokens: &[i64]) -> Result<(Vec<f32>, usize), String> {
        // Output size is (1, 128, vocab_size). 
        // We don't know vocab_size beforehand, but we can guess or use a large enough buffer.
        // Qwen3 vocab size is around 152064.
        let mut vocab_size = 0;
        let mut output_logits = vec![0.0f32; 128 * 152064]; 
        
        let status = unsafe {
            executorch_module_forward(
                self.ptr,
                tokens.as_ptr(),
                tokens.len(),
                output_logits.as_mut_ptr(),
                &mut vocab_size,
            )
        };

        if status != 0 {
            return Err(format!("Forward failed with status {}", status));
        }

        // Truncate to actual size
        output_logits.truncate(128 * vocab_size);
        Ok((output_logits, vocab_size))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { executorch_module_destroy(self.ptr) };
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}
