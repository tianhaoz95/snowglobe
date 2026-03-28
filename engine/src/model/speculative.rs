use crate::model::runner::{EngineSession, ModelRunner};
use std::any::Any;
use std::path::Path;

pub struct SpeculativeRunner {
    pub target: Box<dyn ModelRunner>,
    pub k: usize,
}

pub struct DummyDraftRunner;
impl ModelRunner for DummyDraftRunner {
    fn load(_path: &Path, _config: &crate::InitConfig) -> Result<Box<Self>, String> {
        Ok(Box::new(Self))
    }
    fn forward_all(&self, _session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String> {
        Ok(vec![vec![0.0; 151936]])
    }
    fn backend_name(&self) -> String {
        "CPU".to_string()
    }
}



impl SpeculativeRunner {
    pub fn new(target: Box<dyn ModelRunner>, k: usize) -> Self {
        Self { target, k }
    }
}

impl ModelRunner for SpeculativeRunner {
    fn load(_path: &Path, _config: &crate::InitConfig) -> Result<Box<Self>, String> {
        Err("load not implemented for SpeculativeRunner directly".to_string())
    }

    fn forward_all(&self, session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String> {
        if self.k == 0 {
            return self.target.forward_all(session);
        }

        let num_confirmed = session.tokens.len();
        if num_confirmed == 0 || num_confirmed == session.offset {
             return self.target.forward_all(session);
        }
        
        let m = num_confirmed - session.offset;
        let original_offset = session.offset;

        // 2. Drafting (Dummy: use pad tokens)
        let pad_token = 151643;
        let mut draft_tokens = Vec::new();
        for _ in 0..self.k {
            draft_tokens.push(pad_token);
        }
        
        // 3. Verification Phase
        // Prepare target for speculative verification (e.g. copy KV cache to seq 1)
        self.target.prepare_speculative_verification(session)?;

        // Verification pass: run target on draft tokens using seq 1 (isolated)
        let start_tokens = session.tokens.clone();
        session.tokens.extend(&draft_tokens);
        session.is_speculative = true; // Signals LlamaCppRunner to use seq 1
        
        let target_logits_all = match self.target.forward_all(session) {
            Ok(logits) => logits,
            Err(e) => {
                session.tokens = start_tokens;
                session.offset = original_offset;
                session.is_speculative = false;
                return Err(format!("Target model verification pass failed: {}", e));
            }
        };
        
        // Cleanup after verification (e.g. clear seq 1)
        self.target.cleanup_speculative_verification(session)?;
        session.is_speculative = false;
        session.offset = original_offset;
        
        // Verify tokens
        let mut accepted_count = 0;
        let mut final_tokens = start_tokens.clone();
        
        for i in 0..self.k {
            let target_pred = target_logits_all[m - 1 + i]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap();
            
            if target_pred == draft_tokens[i] {
                accepted_count += 1;
                final_tokens.push(draft_tokens[i]);
            } else {
                break;
            }
        }

        // 4. Commit Phase (Save accepted tokens to KV cache 0)
        session.tokens = final_tokens;
        session.is_speculative = false; // SAVE TO SEQ 0
        
        let commit_logits = self.target.forward_all(session)?;
        
        session.last_accepted_count = accepted_count;

        Ok(vec![commit_logits.last().unwrap().clone()])
    }

    fn backend_name(&self) -> String {
        format!("{} (Spec)", self.target.backend_name())
    }
}
