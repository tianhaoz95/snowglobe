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
}

struct SpeculativeSessionState {
    target_state: Option<Box<dyn Any + Send + Sync>>,
    target_offset: usize,
    last_logits: Option<Vec<f32>>,
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

        // 1. Initialize or extract SpeculativeSessionState
        let mut spec_state = if let Some(state) = session.state.take() {
            match state.downcast::<SpeculativeSessionState>() {
                Ok(s) => *s,
                Err(original_state) => {
                    SpeculativeSessionState {
                        target_state: Some(original_state),
                        target_offset: session.offset,
                        last_logits: None,
                    }
                }
            }
        } else {
            SpeculativeSessionState {
                target_state: None,
                target_offset: 0,
                last_logits: None,
            }
        };

        let num_confirmed = session.tokens.len();
        if num_confirmed == 0 {
             return self.target.forward_all(session);
        }
        
        // 2. Drafting (Dummy: use pad tokens)
        let pad_token = 151643;
        let mut draft_tokens = Vec::new();
        for _ in 0..self.k {
            draft_tokens.push(pad_token);
        }
        
        // 3. Verification Phase
        
        // Get logits for the last confirmed token
        let initial_logits = if let Some(logits) = spec_state.last_logits.take() {
            logits
        } else {
            session.state = spec_state.target_state.take();
            session.offset = spec_state.target_offset;
            session.is_speculative = false;
            let logits = self.target.forward(session)?;
            spec_state.target_state = session.state.take();
            spec_state.target_offset = session.offset;
            logits
        };

        // Prepare target for speculative verification (e.g. copy KV cache to seq 1)
        session.state = spec_state.target_state.take();
        session.offset = spec_state.target_offset;
        self.target.prepare_speculative_verification(session)?;
        spec_state.target_state = session.state.take();

        // Verification pass: run target on draft tokens using seq 1 (isolated)
        let start_tokens = session.tokens.clone();
        session.tokens.extend(&draft_tokens);
        session.state = spec_state.target_state.take();
        session.offset = spec_state.target_offset;
        session.is_speculative = true; // Signals LlamaCppRunner to use seq 1
        
        let target_logits_all = match self.target.forward_all(session) {
            Ok(logits) => logits,
            Err(e) => {
                session.tokens = start_tokens;
                session.state = Some(Box::new(spec_state));
                session.is_speculative = false;
                return Err(format!("Target model verification pass failed: {}", e));
            }
        };
        
        // Cleanup after verification (e.g. clear seq 1)
        self.target.cleanup_speculative_verification(session)?;
        spec_state.target_state = session.state.take();
        session.is_speculative = false;
        
        // Verify tokens
        let mut accepted_count = 0;
        let mut final_tokens = start_tokens;
        let mut current_logits = initial_logits;
        
        for i in 0..self.k {
            let target_pred = current_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap();
            
            if target_pred == draft_tokens[i] {
                accepted_count += 1;
                final_tokens.push(target_pred);
                current_logits = target_logits_all[i].clone();
            } else {
                final_tokens.push(target_pred);
                current_logits = target_logits_all[i].clone();
                break;
            }
        }

        if accepted_count == self.k {
            let last_target_pred = current_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap();
            final_tokens.push(last_target_pred);
            spec_state.last_logits = None;
        } else {
            spec_state.last_logits = Some(current_logits);
        }

        // 4. Commit Phase (Save accepted tokens to KV cache 0)
        session.tokens = final_tokens.clone();
        session.state = spec_state.target_state.take();
        session.offset = spec_state.target_offset;
        session.is_speculative = false; // SAVE TO SEQ 0
        
        let commit_logits = self.target.forward_all(session)?;
        
        spec_state.target_state = session.state.take();
        spec_state.target_offset = session.offset;
        let last_logits = commit_logits.last().unwrap().clone();
        spec_state.last_logits = Some(last_logits.clone());
        
        session.tokens = final_tokens;
        session.last_accepted_count = accepted_count + 1;
        session.state = Some(Box::new(spec_state));

        Ok(vec![last_logits])
    }
}
