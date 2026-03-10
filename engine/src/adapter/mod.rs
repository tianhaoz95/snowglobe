use std::ffi::{CString, c_char, c_void};

#[repr(C)]
struct ExecuTorchModuleOpaque(c_void);

#[cfg(has_executorch)]
unsafe extern "C" {
    fn executorch_module_load(pte_path: *const c_char) -> *mut ExecuTorchModuleOpaque;
    fn executorch_module_destroy(module: *mut ExecuTorchModuleOpaque);
    fn executorch_module_forward(
        module: *mut ExecuTorchModuleOpaque,
        input_tokens: *const c_void,
        input_len: usize,
        use_int32: i32,
        output_logits: *mut f32,
        output_vocab_size: *mut usize,
    ) -> i32;
}

#[cfg(not(has_executorch))]
unsafe extern "C" fn executorch_module_load(_pte_path: *const c_char) -> *mut ExecuTorchModuleOpaque {
    std::ptr::null_mut()
}

#[cfg(not(has_executorch))]
unsafe extern "C" fn executorch_module_destroy(_module: *mut ExecuTorchModuleOpaque) {}

#[cfg(not(has_executorch))]
unsafe extern "C" fn executorch_module_forward(
    _module: *mut ExecuTorchModuleOpaque,
    _input_tokens: *const c_void,
    _input_len: usize,
    _use_int32: i32,
    _output_logits: *mut f32,
    _output_vocab_size: *mut usize,
) -> i32 {
    -1
}

#[derive(Debug)]
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

    /// Run inference with the module.
    ///
    /// Tokens must be either `&[i32]` or `&[i64]`, depending on the exported model's expected type.
    pub fn forward<T: 'static>(&mut self, tokens: &[T]) -> Result<(Vec<f32>, usize), String> {
        let use_int32 = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            1
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            0
        } else {
            return Err("Unsupported token ID type. Use i32 or i64.".to_string());
        };

        let mut vocab_size = 0;
        let mut output_logits = vec![0.0f32; 128 * 152064];

        let status = unsafe {
            executorch_module_forward(
                self.ptr,
                tokens.as_ptr() as *const c_void,
                tokens.len(),
                use_int32,
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
