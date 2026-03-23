use std::any::Any;
use std::path::Path;

pub struct EngineSession {
    pub tokens: Vec<u32>,
    pub offset: usize,
    pub state: Option<Box<dyn Any + Send + Sync>>,
}

impl EngineSession {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            offset: 0,
            state: None,
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

    /// Perform a forward pass, returning logits for the last token.
    fn forward(&self, session: &mut EngineSession) -> Result<Vec<f32>, String>;
}
