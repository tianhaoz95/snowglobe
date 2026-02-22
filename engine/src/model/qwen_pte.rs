use crate::GLOBAL_MODEL;
use crate::adapter::Module;
use std::time::Instant;

pub struct QwenPte {
    module: Module,
}

impl QwenPte {
    pub fn new(pte_path: &str) -> Result<Self, String> {
        let module = Module::new(pte_path)?;
        Ok(Self { module })
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String, String> {
        let loaded_model = GLOBAL_MODEL.get().ok_or("Model not initialized")?;
        let tokenizer = &loaded_model.tokenizer;

        // 1. Tokenize prompt with chat template
        let im_start_id = tokenizer
            .token_to_id("<|im_start|>")
            .expect("Missing <|im_start|>");
        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .expect("Missing <|im_end|>");
        let newline_id = tokenizer
            .token_to_id(
                "
",
            )
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

        // Use environment variable to decide if we use i32 or i64 tokens.
        // MPS backend usually requires i32 for consistency with the causal mask in our script.
        let use_mps = std::env::var("EXECUTORCH_USE_MPS").is_ok();

        for i in 0..max_new_tokens {
            let current_len = tokens.len();
            if current_len >= 128 {
                break;
            }
            println!("--- Token {} ---", i);
            let start_token = Instant::now();

            // 2. Forward
            println!("  Forwarding (use_mps: {})...", use_mps);
            let forward_start = Instant::now();

            let (logits, vocab_size) = if use_mps {
                let mut input_tokens = vec![0i32; 128];
                for (j, &token) in tokens.iter().enumerate() {
                    input_tokens[j] = token as i32;
                }
                self.module.forward(&input_tokens)
            } else {
                let mut input_tokens = vec![0i64; 128];
                for (j, &token) in tokens.iter().enumerate() {
                    input_tokens[j] = token as i64;
                }
                self.module.forward(&input_tokens)
            }
            .map_err(|e| format!("Model forward failed: {:?}", e))?;

            println!("  Forward pass done in {:?}", forward_start.elapsed());

            // 3. Get logits for the last token position
            let last_pos = current_len - 1;
            let start_idx = last_pos * vocab_size;

            let mut max_logit = f32::NEG_INFINITY;
            let mut next_token = 0;

            println!("  Calculating next token...");
            for v in 0..vocab_size {
                let logit = logits[start_idx + v];
                if logit > max_logit {
                    max_logit = logit;
                    next_token = v;
                }
            }
            println!("  Next token calculated...");

            if next_token == im_end_id as usize {
                println!("  EOS detected.");
                break;
            }

            tokens.push(next_token as u32);
            response_tokens.push(next_token as u32);
            println!(
                "  Next token: {} (time: {:?})",
                next_token,
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
