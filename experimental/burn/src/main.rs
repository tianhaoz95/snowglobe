use std::io::Write;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use burn::backend::{
    wgpu::{Wgpu, WgpuDevice},
    Autodiff,
};
use burn_transformers::generation::{Generation, TextGeneration};
use burn_transformers::pipeline::TextGenerationPipeline;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The model directory, which can be a local path or a repository ID on the Hugging Face Hub.
    #[arg(long)]
    model_dir: String,

    /// The prompt to generate text from.
    #[arg(long)]
    prompt: String,
}

fn main() -> Result<()> {
    // Other backends can be used, but WGPU is the default.
    // LBFGS is not used, so the autodiff backend is not important.
    type Backend = Wgpu;
    type AutodiffBackend = Autodiff<Backend>;

    let args = Args::parse();
    let handle = Api::new()?.model(args.model_dir).repo(RepoType::Model);
    let model_path = handle.get("model.safetensors")?;

    let tokenizer_path = handle.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

    println!("Loading model...");
    let pipeline = TextGenerationPipeline::<AutodiffBackend>::from_safetensors(
        model_path,
        WgpuDevice::default(),
        Arc::new(tokenizer),
    )?;

    println!("Generating text...");
    let mut stream = pipeline.generate_stream(&args.prompt, None);
    print!("{}", args.prompt);
    while let Some(text) = stream.next_token() {
        print!("{text}");
        std::io::stdout().flush()?;
    }
    println!();
    Ok(())
}