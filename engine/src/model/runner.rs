use std::any::Any;
use std::path::Path;

pub struct EngineSession {
    pub tokens: Vec<u32>,
    pub offset: usize,
    pub state: Option<Box<dyn Any + Send + Sync>>,
    pub last_accepted_count: usize,
    pub is_speculative: bool,
}

impl EngineSession {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            offset: 0,
            state: None,
            last_accepted_count: 0,
            is_speculative: false,
        }
    }
}

impl Default for EngineSession {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ModelRunner: Send {
    /// Load the model from the specified directory/file.
    fn load(path: &Path, config: &crate::InitConfig) -> Result<Box<Self>, String>
    where
        Self: Sized;

    /// Perform a forward pass, returning logits for all new tokens.
    /// The last element in the outer Vec corresponds to the logits for the next token.
    fn forward_all(&self, session: &mut EngineSession) -> Result<Vec<Vec<f32>>, String>;

    /// Perform a forward pass, returning logits for the last token.
    fn forward(&self, session: &mut EngineSession) -> Result<Vec<f32>, String> {
        let all_logits = self.forward_all(session)?;
        Ok(all_logits.into_iter().last().ok_or("No logits returned")?)
    }

    /// Roll back the KV cache to the specified offset.
    fn rollback(&self, session: &mut EngineSession, offset: usize) -> Result<(), String> {
        session.offset = offset;
        Ok(())
    }

    /// Prepare the runner for a speculative verification pass.
    fn prepare_speculative_verification(&self, _session: &mut EngineSession) -> Result<(), String> {
        Ok(())
    }

    /// Cleanup after a speculative verification pass.
    fn cleanup_speculative_verification(&self, _session: &mut EngineSession) -> Result<(), String> {
        Ok(())
    }

    /// Returns the name of the actual hardware backend being used.
    fn backend_name(&self) -> String;
}
