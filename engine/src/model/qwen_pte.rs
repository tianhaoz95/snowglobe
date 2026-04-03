use crate::GLOBAL_MODEL;
use crate::adapter::Module as ExecModule;
use crate::model::{KVCache, Model};
use burn::module::Module;
use burn::tensor::{Int, Tensor, backend::Backend, Shape, TensorData};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;
use std::marker::PhantomData;

pub struct QwenPteConfig {
    pub pte_path: String,
}

#[derive(Debug)]
pub struct QwenPte<B: Backend> {
    module: Arc<Mutex<ExecModule>>,
    logit_buffer: Mutex<Vec<f32>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> Clone for QwenPte<B> {
    fn clone(&self) -> Self {
        Self {
            module: self.module.clone(),
            logit_buffer: Mutex::new(Vec::new()),
            _phantom: PhantomData,
        }
    }
}

// Manual implementation of burn::module::Module for QwenPte to handle non-Burn types
impl<B: Backend> Module<B> for QwenPte<B> {
    type Record = ();

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {}
    fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self { self }
    fn load_record(self, _record: Self::Record) -> Self { self }
    fn into_record(self) -> Self::Record { () }
    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }
    fn fork(self, _device: &B::Device) -> Self { self }
    fn to_device(self, _device: &B::Device) -> Self { self }
}

impl<B: Backend> QwenPte<B> {
    pub fn new(pte_path: &str) -> Result<Self, String> {
        let module = ExecModule::new(pte_path)?;
        Ok(Self {
            module: Arc::new(Mutex::new(module)),
            logit_buffer: Mutex::new(Vec::new()),
            _phantom: PhantomData,
        })
    }

    pub fn get_name(&self) -> String {
        self.module.lock().get_name()
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String, String> {
        let global_lock = GLOBAL_MODEL.read();
        let loaded_model = global_lock.as_ref().ok_or("Model not initialized")?;
        let tokenizer = &loaded_model.tokenizer;

        // 1. Tokenize prompt with chat template
        let im_start_id = tokenizer
            .token_to_id("<|im_start|>")
            .expect("Missing <|im_start|>");
        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .expect("Missing <|im_end|>");
        let newline_id = tokenizer
            .token_to_id("\n")
            .unwrap_or(198);

        let mut tokens = vec![im_start_id];
        tokens.extend(tokenizer.encode("user", false).unwrap().get_ids());
        tokens.push(newline_id);
        tokens.extend(tokenizer.encode(prompt, false).unwrap().get_ids());
        tokens.push(im_end_id);
        tokens.push(newline_id);
        tokens.push(im_start_id);
        tokens.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
        tokens.push(newline_id);

        let mut response_tokens = Vec::new();

        for i in 0..max_new_tokens {
            let current_len = tokens.len();
            if current_len >= 128 {
                break;
            }
            println!("--- Token {} ---", i);
            let start_token = Instant::now();

            // 2. Forward using the trait method
            let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(
                    tokens.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                    Shape::new([1, current_len]),
                ),
                &B::Device::default(),
            );

            let (output, _) = crate::model::Model::forward(self, input_tensor, None, 0);

            let [_, seq_len, vocab_size]: [usize; 3] = output.dims();
            let next_token_id = output
                .slice([0..1, (seq_len - 1)..(seq_len), 0..vocab_size])
                .argmax(2)
                .to_data()
                .into_vec::<i32>()
                .expect("Failed to get data")[0] as u32;

            if next_token_id == im_end_id {
                println!("  EOS detected.");
                break;
            }

            tokens.push(next_token_id);
            response_tokens.push(next_token_id);
            println!(
                "  Next token: {} (time: {:?})",
                next_token_id,
                start_token.elapsed()
            );
            if let Ok(current_text) = tokenizer.decode(&response_tokens, true) {
                println!("  Current sequence: \"{}\"", current_text);
            }
        }

        let completion = tokenizer
            .decode(&response_tokens, true)
            .map_err(|e| format!("Decoding failed: {}", e))?;

        Ok(completion)
    }
}

impl<B: Backend> Model<B> for QwenPte<B> {
    type Config = QwenPteConfig;

    fn init(config: &Self::Config, _device: &B::Device) -> Self {
        Self::new(&config.pte_path).expect("Failed to load PTE model")
    }

    fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        _cache: Option<Vec<Option<KVCache<B>>>>,
        _offset: usize,
    ) -> (Tensor<B, 3>, Vec<KVCache<B>>) {
        let device = input.device();
        let [batch_size, seq_len] = input.dims();
        let token_vec = input.to_data().into_vec::<i32>().expect("Failed to get tokens");

        let use_mps = std::env::var("EXECUTORCH_USE_MPS").is_ok();
        
        let mut module = self.module.lock();
        let (logits_vec, vocab_size) = if use_mps {
            let mut input_tokens = vec![0i32; 128];
            for (j, &token) in token_vec.iter().enumerate() {
                if j < 128 { input_tokens[j] = token; }
            }
            module.forward(&input_tokens)
        } else {
            let mut input_tokens = vec![0i64; 128];
            for (j, &token) in token_vec.iter().enumerate() {
                if j < 128 { input_tokens[j] = token as i64; }
            }
            module.forward(&input_tokens)
        }.expect("PTE forward failed");

        let output = Tensor::<B, 3>::from_data(
            TensorData::new(logits_vec, Shape::new([1, 128, vocab_size])),
            &device,
        );

        // Truncate back to [batch_size, seq_len, vocab_size]
        // Ensure we don't slice beyond 128 (model fixed limit)
        let effective_seq_len = seq_len.min(128);
        let output = output.slice([0..batch_size, 0..effective_seq_len, 0..vocab_size]);

        (output, Vec::new())
    }
}

use crate::model::runner::{EngineSession, ModelRunner, ExecutionMode, LogitView, BackendInfo};
use std::any::Any;

impl<B: Backend> ModelRunner for QwenPte<B> {
    fn load(path: &std::path::Path, _config: &serde_json::Value) -> Result<Box<Self>, String> {
        Ok(Box::new(Self::new(path.to_str().ok_or("Invalid path")?)?))
    }

    fn execute(
        &self,
        session: &mut EngineSession,
        input_tokens: &[u32],
        _mode: ExecutionMode,
    ) -> Result<LogitView, String> {
        let num_new = input_tokens.len();
        if num_new == 0 {
            return Err("No new tokens".to_string());
        }

        let use_mps = std::env::var("EXECUTORCH_USE_MPS").is_ok();
        
        let token_pos_start = session.current_kv_len % 128; // Simple rolling window for PTE demo

        let mut module = self.module.lock();
        let (logits_vec, vocab_size) = if use_mps {
            let mut input_tokens_buf = vec![0i32; 128];
            for (j, &token) in input_tokens.iter().enumerate() {
                if j < 128 { input_tokens_buf[j] = token as i32; }
            }
            module.forward_range(&input_tokens_buf, token_pos_start, num_new)
        } else {
            let mut input_tokens_buf = vec![0i64; 128];
            for (j, &token) in input_tokens.iter().enumerate() {
                if j < 128 { input_tokens_buf[j] = token as i64; }
            }
            module.forward_range(&input_tokens_buf, token_pos_start, num_new)
        }.map_err(|e| format!("PTE forward failed: {}", e))?;

        session.current_kv_len += num_new;

        let (num_output_rows, final_logits) = if _mode == ExecutionMode::Prefill {
            let start = (num_new - 1) * vocab_size;
            (1, logits_vec[start..start + vocab_size].to_vec())
        } else {
            (num_new, logits_vec)
        };

        Ok(LogitView {
            data: final_logits,
            shape: (num_output_rows, vocab_size),
        })
    }

    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String> {
        if len < session.tokens.len() {
            session.tokens.truncate(len);
        }
        session.current_kv_len = len;
        Ok(())
    }

    fn fork_state(&self, _session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String> {
        Ok(Box::new(()))
    }

    fn get_backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: self.backend_name(),
            max_sequence_length: 128,
            max_batch_size: 1,
        }
    }

    fn model_name(&self) -> String {
        self.get_name()
    }
}

impl<B: Backend> QwenPte<B> {
    fn backend_name(&self) -> String {
        let use_mps = std::env::var("EXECUTORCH_USE_MPS").is_ok();
        if use_mps {
            "ExecuTorch (MPS)".to_string()
        } else {
            "ExecuTorch (CPU)".to_string()
        }
    }
}

