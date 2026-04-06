# Gemma 4 LiteRT Exporter

This directory contains scripts to export Gemma 4 models to LiteRT (.tflite) format.

## Prerequisites

1. **Python environment**:
   ```bash
   pip install torch transformers litert-torch huggingface_hub
   ```

2. **Hugging Face Access**:
   Make sure you have access to `google/gemma-4-E2B-it` and are logged in:
   ```bash
   huggingface-cli login
   ```

## Usage

To export the default Gemma 4 E2B IT model:
```bash
python export_gemma4_litert.py --model_id google/gemma-4-E2B-it --output ../.test_assets/gemma4_e2b/model.tflite
```

The script will:
1. Download the model from Hugging Face.
2. Convert it to LiteRT using the Generative API signatures (`prefill`, `decode`).
3. Save the `.tflite` model and the associated tokenizer/config files.

## Integration with Snowglobe

The Snowglobe engine (`./engine`) uses the `LiteRTRunner` to load and execute these exported models. The demo app (`./demo`) will automatically look for `model.tflite` when the LiteRT backend is selected.
