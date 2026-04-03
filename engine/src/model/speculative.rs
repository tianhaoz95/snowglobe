use crate::model::runner::{EngineSession, ModelRunner, ExecutionMode, LogitView, BackendInfo};
use std::any::Any;
use std::path::Path;
use lru::LruCache;
use std::num::NonZeroUsize;
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachebackConfig {
    pub leader_len: usize,
    pub follower_len: usize,
    pub capacity: usize,
    pub follower_capacity: usize,
}

impl Default for CachebackConfig {
    fn default() -> Self {
        Self {
            leader_len: 2,
            follower_len: 3,
            capacity: 2048,
            follower_capacity: 4,
        }
    }
}

pub struct CacheTable {
    // Maps a leader (LL tokens) to a list of followers (FL tokens)
    // We use a Vec of followers to allow multiple matches, 
    // but the most recent one is usually best.
    entries: LruCache<Vec<u32>, Vec<Vec<u32>>>,
    leader_len: usize,
    follower_len: usize,
    follower_capacity: usize,
}

impl CacheTable {
    pub fn new(config: &CachebackConfig) -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(config.capacity).unwrap()),
            leader_len: config.leader_len,
            follower_len: config.follower_len,
            follower_capacity: config.follower_capacity,
        }
    }

    pub fn update(&mut self, tokens: &[u32]) {
        if tokens.len() < self.leader_len + self.follower_len {
            return;
        }

        // Only update with the latest window to avoid redundant work and bias.
        let window_size = 16;
        let start = if tokens.len() > window_size + self.leader_len + self.follower_len {
            tokens.len() - (window_size + self.leader_len + self.follower_len)
        } else {
            0
        };

        for i in start..tokens.len() - self.leader_len - self.follower_len + 1 {
            let leader = tokens[i..i + self.leader_len].to_vec();
            let follower = tokens[i + self.leader_len..i + self.leader_len + self.follower_len].to_vec();

            if let Some(followers) = self.entries.get_mut(&leader) {
                if !followers.contains(&follower) {
                    followers.insert(0, follower);
                    if followers.len() > self.follower_capacity {
                        followers.pop();
                    }
                } else {
                    // Move to front if already present
                    let pos = followers.iter().position(|f| f == &follower).unwrap();
                    let f = followers.remove(pos);
                    followers.insert(0, f);
                }
            } else {
                self.entries.put(leader, vec![follower]);
            }
        }
    }

    pub fn draft(&mut self, current_tokens: &[u32], k: usize) -> Vec<u32> {
        let mut draft_tokens = Vec::new();
        let mut temp_sequence = current_tokens.to_vec();

        while draft_tokens.len() < k {
            if temp_sequence.len() < self.leader_len {
                break;
            }

            let leader = temp_sequence[temp_sequence.len() - self.leader_len..].to_vec();
            if let Some(followers) = self.entries.get(&leader) {
                // Take the most recent follower
                let follower = &followers[0];
                let mut added = 0;
                for &t in follower {
                    if draft_tokens.len() < k {
                        draft_tokens.push(t);
                        temp_sequence.push(t);
                        added += 1;
                    } else {
                        break;
                    }
                }
                if added == 0 {
                    break;
                }
            } else {
                break;
            }
        }
        draft_tokens
    }
}

pub struct FrozenTable {
    pub entries: HashMap<Vec<u32>, Vec<Vec<u32>>>,
    pub leader_len: usize,
    pub follower_len: usize,
}

impl FrozenTable {
    pub fn new(leader_len: usize, follower_len: usize) -> Self {
        Self {
            entries: HashMap::new(),
            leader_len,
            follower_len,
        }
    }

    pub fn draft(&self, current_tokens: &[u32], k: usize) -> Vec<u32> {
        let mut draft_tokens = Vec::new();
        let mut temp_sequence = current_tokens.to_vec();

        while draft_tokens.len() < k {
            if temp_sequence.len() < self.leader_len {
                break;
            }

            let leader = temp_sequence[temp_sequence.len() - self.leader_len..].to_vec();
            if let Some(followers) = self.entries.get(&leader) {
                let follower = &followers[0];
                let mut added = 0;
                for &t in follower {
                    if draft_tokens.len() < k {
                        draft_tokens.push(t);
                        temp_sequence.push(t);
                        added += 1;
                    } else {
                        break;
                    }
                }
                if added == 0 {
                    break;
                }
            } else {
                break;
            }
        }
        draft_tokens
    }
}

pub struct SpeculativeRunner {
    pub target: Box<dyn ModelRunner>,
    pub k: usize,
    pub dynamic_cache: Mutex<CacheTable>,
    pub frozen_cache: Option<FrozenTable>,
}

pub struct DummyDraftRunner {
    logit_buffer: Vec<f32>,
}
impl DummyDraftRunner {
    pub fn new() -> Self {
        Self { logit_buffer: vec![0.0; 151936] }
    }
}

impl ModelRunner for DummyDraftRunner {
    fn load(_path: &Path, _config: &serde_json::Value) -> Result<Box<Self>, String> {
        Ok(Box::new(Self::new()))
    }
    fn execute(
        &self,
        _session: &mut EngineSession,
        _input_tokens: &[u32],
        _mode: ExecutionMode,
    ) -> Result<LogitView, String> {
        Ok(LogitView {
            data: self.logit_buffer.clone(),
            shape: (1, 151936),
        })
    }
    fn truncate_cache(&self, _session: &mut EngineSession, _len: usize) -> Result<(), String> {
        Ok(())
    }
    fn fork_state(&self, _session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String> {
        Ok(Box::new(()))
    }
    fn get_backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Dummy".to_string(),
            max_sequence_length: 32768,
            max_batch_size: 1,
        }
    }
}

impl SpeculativeRunner {
    pub fn new(target: Box<dyn ModelRunner>, k: usize, config: CachebackConfig) -> Self {
        Self { 
            target, 
            k,
            dynamic_cache: Mutex::new(CacheTable::new(&config)),
            frozen_cache: None,
        }
    }

    pub fn load_frozen_table(&mut self, path: &Path) -> Result<(), String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        
        if data.len() < 12 {
            return Err("Invalid frozen table file: too short".to_string());
        }

        let mut offset = 0;
        let leader_len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let follower_len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let num_entries = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;

        let mut entries = HashMap::with_capacity(num_entries);

        for _ in 0..num_entries {
            let mut leader = Vec::with_capacity(leader_len);
            for _ in 0..leader_len {
                if offset + 4 > data.len() { return Err("Unexpected EOF".to_string()); }
                leader.push(u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()));
                offset += 4;
            }

            if offset + 4 > data.len() { return Err("Unexpected EOF".to_string()); }
            let num_followers = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;

            let mut followers = Vec::with_capacity(num_followers);
            for _ in 0..num_followers {
                let mut follower = Vec::with_capacity(follower_len);
                for _ in 0..follower_len {
                    if offset + 4 > data.len() { return Err("Unexpected EOF".to_string()); }
                    follower.push(u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()));
                    offset += 4;
                }
                followers.push(follower);
            }
            entries.insert(leader, followers);
        }

        self.frozen_cache = Some(FrozenTable {
            entries,
            leader_len,
            follower_len,
        });

        Ok(())
    }
}

impl ModelRunner for SpeculativeRunner {
    fn load(_path: &Path, _config: &serde_json::Value) -> Result<Box<Self>, String> {
        Err("load not implemented for SpeculativeRunner directly".to_string())
    }

    fn execute(
        &self,
        session: &mut EngineSession,
        input_tokens: &[u32],
        mode: ExecutionMode,
    ) -> Result<LogitView, String> {
        if mode == ExecutionMode::Prefill {
            // Update cache with prompt tokens
            self.dynamic_cache.lock().update(input_tokens);
            return self.target.execute(session, input_tokens, mode);
        }

        // --- Speculative Drafting ---
        // 1. Generate draft tokens using Cacheback
        let draft_tokens = {
            let mut cache = self.dynamic_cache.lock();
            
            // Context for drafting is session history + current input
            let mut context = session.tokens.clone();
            context.extend_from_slice(input_tokens);
            
            let mut draft = cache.draft(&context, self.k);
            
            // If dynamic cache didn't provide enough tokens, try frozen cache
            if draft.len() < self.k {
                if let Some(frozen) = &self.frozen_cache {
                    // Start drafting from where dynamic cache left off
                    let mut temp_seq = context;
                    temp_seq.extend(&draft);
                    let frozen_draft = frozen.draft(&temp_seq, self.k - draft.len());
                    draft.extend(frozen_draft);
                }
            }
            draft
        };

        if draft_tokens.is_empty() {
            return self.target.execute(session, input_tokens, mode);
        }

        // 2. Combine input with draft
        let mut combined_input = input_tokens.to_vec();
        combined_input.extend(&draft_tokens);

        // 3. Execute target model on the combined sequence
        let initial_kv_len = session.current_kv_len;
        let target_result = self.target.execute(
            session, 
            &combined_input, 
            ExecutionMode::Verify { draft_len: draft_tokens.len() }
        )?;

        let (_seq_len, vocab_size) = target_result.shape;
        
        // 4. Verification Loop
        let mut accepted_count = 1; // Always accept the first token's logit
        for i in 0..draft_tokens.len() {
            // Logits for token at index i are in target_result.data[i * vocab_size .. (i+1) * vocab_size]
            let logits = &target_result.data[i * vocab_size .. (i+1) * vocab_size];

            // Sample (greedy for verification)
            let sampled_token = logits.iter().enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32).unwrap();

            if sampled_token == draft_tokens[i] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        // 5. Cleanup
        if accepted_count <= draft_tokens.len() {
            // We rejected some tokens, need to truncate KV cache
            let correct_kv_len = initial_kv_len + accepted_count;
            self.target.truncate_cache(session, correct_kv_len)?;
        }

        if accepted_count > 1 {
            eprintln!("[SPEC] Accepted {}/{} tokens", accepted_count, draft_tokens.len() + 1);
        }

        // 6. Return logits for accepted tokens
        let final_data = target_result.data[0..accepted_count * vocab_size].to_vec();

        Ok(LogitView {
            data: final_data,
            shape: (accepted_count, vocab_size),
        })
    }

    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String> {
        if len < session.tokens.len() {
            session.tokens.truncate(len);
        }
        self.target.truncate_cache(session, len)
    }

    fn fork_state(&self, session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String> {
        self.target.fork_state(session)
    }

    fn get_backend_info(&self) -> BackendInfo {
        let mut info = self.target.get_backend_info();
        info.name = format!("{} (Cacheback)", info.name);
        info
    }

    fn model_name(&self) -> String {
        self.target.model_name()
    }

    fn update_cache(&self, tokens: &[u32]) {
        self.dynamic_cache.lock().update(tokens);
    }
}
