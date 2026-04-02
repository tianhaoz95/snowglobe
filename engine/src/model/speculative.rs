use crate::model::runner::{EngineSession, ModelRunner, ExecutionMode, LogitView, BackendInfo};
use std::any::Any;
use std::path::Path;
use lru::LruCache;
use std::num::NonZeroUsize;
use parking_lot::Mutex;

pub struct CacheTable {
    // Maps a leader (LL tokens) to a list of followers (FL tokens)
    // We use a Vec of followers to allow multiple matches, 
    // but the most recent one is usually best.
    entries: LruCache<Vec<u32>, Vec<Vec<u32>>>,
    leader_len: usize,
    follower_len: usize,
}

impl CacheTable {
    pub fn new(capacity: usize, leader_len: usize, follower_len: usize) -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            leader_len,
            follower_len,
        }
    }

    pub fn update(&mut self, tokens: &[u32]) {
        if tokens.len() <= self.leader_len + self.follower_len {
            return;
        }

        for i in 0..tokens.len() - self.leader_len - self.follower_len + 1 {
            let leader = tokens[i..i + self.leader_len].to_vec();
            let follower = tokens[i + self.leader_len..i + self.leader_len + self.follower_len].to_vec();

            if let Some(followers) = self.entries.get_mut(&leader) {
                if !followers.contains(&follower) {
                    followers.insert(0, follower);
                    if followers.len() > 4 {
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

pub struct SpeculativeRunner {
    pub target: Box<dyn ModelRunner>,
    pub k: usize,
    pub cache: Mutex<CacheTable>,
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
    pub fn new(target: Box<dyn ModelRunner>, k: usize) -> Self {
        Self { 
            target, 
            k,
            cache: Mutex::new(CacheTable::new(2048, 1, 3)),
        }
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
            self.cache.lock().update(input_tokens);
            return self.target.execute(session, input_tokens, mode);
        }

        // --- Speculative Drafting ---
        // 1. Generate draft tokens using Cacheback
        let draft_tokens = {
            let mut cache = self.cache.lock();
            // We need the full history to draft correctly
            // For now we assume input_tokens is just the last accepted token in Decode mode
            // Actual history should be in session or maintained externally.
            // If we only have input_tokens (last 1), we can only draft if LL=1.
            cache.draft(input_tokens, self.k)
        };

        if draft_tokens.is_empty() {
            return self.target.execute(session, input_tokens, mode);
        }

        // 2. Combine input with draft
        let mut combined_input = input_tokens.to_vec();
        combined_input.extend(&draft_tokens);

        // 3. Execute target model on the combined sequence
        let result = self.target.execute(session, &combined_input, mode);

        // 4. Update cache with whatever the target model produced (ideally we update after verification)
        // But since ModelRunner::execute returns logits, the verification happens in lib.rs.
        // This is a limitation of the current trait if we want to update cache with ACCEPTED tokens.
        // For now, we return the LogitView. lib.rs will handle verification and truncation.
        
        result
    }

    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String> {
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
        self.cache.lock().update(tokens);
    }
}
