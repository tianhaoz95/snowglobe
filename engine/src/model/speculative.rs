use crate::model::runner::{EngineSession, ModelRunner, ExecutionMode, LogitView, BackendInfo};
use std::any::Any;
use std::path::Path;

pub struct SpeculativeRunner {
    pub target: Box<dyn ModelRunner>,
    pub k: usize,
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
        Self { target, k }
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
        // For now, just delegate to target. 
        // Actual speculative orchestration will happen in lib.rs or we can implement it here later.
        self.target.execute(session, input_tokens, mode)
    }

    fn truncate_cache(&self, session: &mut EngineSession, len: usize) -> Result<(), String> {
        self.target.truncate_cache(session, len)
    }

    fn fork_state(&self, session: &EngineSession) -> Result<Box<dyn Any + Send + Sync>, String> {
        self.target.fork_state(session)
    }

    fn get_backend_info(&self) -> BackendInfo {
        let mut info = self.target.get_backend_info();
        info.name = format!("{} (Spec)", info.name);
        info
    }

    fn model_name(&self) -> String {
        self.target.model_name()
    }
}
