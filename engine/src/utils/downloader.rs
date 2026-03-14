use futures_util::StreamExt;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

async fn download_file(url: &str, path: &Path) -> Result<(), String> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Download failed: {}", e))?;
    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status: {}",
            response.status()
        ));
    }
    let mut stream = response.bytes_stream();
    let mut file = File::create(path)
        .await
        .map_err(|e| format!("File creation failed: {}", e))?;
    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| format!("Stream error: {}", e))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
    }
    Ok(())
}

pub async fn download_qwen2_5_0_5b_instruct(cache_dir: String) -> String {
    let model_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors";
    let tokenizer_url =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
    let config_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json";
    download_model(
        cache_dir,
        model_url.to_string(),
        tokenizer_url.to_string(),
        config_url.to_string(),
    )
    .await
}

pub async fn download_qwen3_0_6b(cache_dir: String) -> String {
    let model_url = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors";
    let tokenizer_url = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json";
    let config_url = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json";
    download_model(
        cache_dir,
        model_url.to_string(),
        tokenizer_url.to_string(),
        config_url.to_string(),
    )
    .await
}

pub async fn download_qwen3_5_0_8b(cache_dir: String) -> String {
    let model_url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors";
    let tokenizer_url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json";
    let config_url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json";
    download_model(
        cache_dir,
        model_url.to_string(),
        tokenizer_url.to_string(),
        config_url.to_string(),
    )
    .await
}

pub async fn download_qwen_gguf(cache_dir: String) -> String {
    let model_url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf";
    let tokenizer_url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json";
    let config_url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json";
    
    let model_path = Path::new(&cache_dir).join("model.gguf");
    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");
    let config_path = Path::new(&cache_dir).join("config.json");

    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        return format!("Permission error: {}", e);
    }

    if !model_path.exists() {
        if let Err(e) = download_file(&model_url, &model_path).await {
            return e;
        }
    }

    if !tokenizer_path.exists() {
        if let Err(e) = download_file(&tokenizer_url, &tokenizer_path).await {
            return e;
        }
    }

    if !config_path.exists() {
        if let Err(e) = download_file(&config_url, &config_path).await {
            return e;
        }
    }

    "Success".to_string()
}

pub async fn download_model(
    cache_dir: String,
    model_url: String,
    tokenizer_url: String,
    config_url: String,
) -> String {
    let model_path = Path::new(&cache_dir).join("model.safetensors");
    let tokenizer_path = Path::new(&cache_dir).join("tokenizer.json");
    let config_path = Path::new(&cache_dir).join("config.json");

    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        return format!("Permission error: {}", e);
    }

    if !model_path.exists() {
        if let Err(e) = download_file(&model_url, &model_path).await {
            return e;
        }
    } else {
        // Simple check to ensure file is at least somewhat reasonably sized.
        if let Ok(meta) = std::fs::metadata(&model_path) {
            if meta.len() < 500 * 1024 * 1024 {
                // Smaller threshold for flexibility
                let _ = std::fs::remove_file(&model_path);
                if let Err(e) = download_file(&model_url, &model_path).await {
                    return e;
                }
            }
        }
    }

    if !tokenizer_path.exists() {
        if let Err(e) = download_file(&tokenizer_url, &tokenizer_path).await {
            return e;
        }
    }

    if !config_path.exists() {
        if let Err(e) = download_file(&config_url, &config_path).await {
            return e;
        }
    }

    "Success".to_string()
}
