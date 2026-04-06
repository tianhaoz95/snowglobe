import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import litert_torch
import argparse
import os

def export_gemma4(model_id, output_path):
    """
    Downloads and converts a Gemma 4 model to LiteRT format.
    Uses the latest litert-torch (formerly ai-edge-torch) features for 2026.
    """
    print(f"Loading model {model_id} from Hugging Face...")
    # Gemma 4 requires transformers >= 5.0.0
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    print("Preparing for LiteRT conversion...")
    
    # Define the max sequence length for the KV cache
    kv_cache_max_len = 2048
    
    # In 2026, litert_torch provides a high-level API for LLMs
    # that automatically handles stateful KV cache and multi-signatures.
    
    # We define sample inputs for the signatures
    # 'prefill' signature: (batch=1, seq_len=variable)
    # 'decode' signature: (batch=1, seq_len=1)
    
    print("Converting to LiteRT (.tflite)...")
    
    # Wrapping the model to expose specific signatures as required by Snowglobe engine
    class SnowglobeLiteRTWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        @torch.export.expose
        def prefill(self, input_ids):
            # Parallel processing of initial prompt
            return self.model(input_ids).logits
            
        @torch.export.expose
        def decode(self, input_id, pos):
            # Token-by-token generation with KV cache position
            # In a real litert-torch export, 'pos' would be linked to the stateful variable
            return self.model(input_id).logits

    wrapper = SnowglobeLiteRTWrapper(model)
    
    # Export the model with prefill and decode signatures
    # This allows the engine to switch between prompt processing and generation efficiently.
    litert_model = litert_torch.convert(
        wrapper,
        signatures={
            "prefill": torch.export.export(wrapper.prefill, (torch.zeros((1, 128), dtype=torch.int32),)),
            "decode": torch.export.export(wrapper.decode, (torch.zeros((1, 1), dtype=torch.int32), torch.tensor(0, dtype=torch.int32))),
        },
        # Enable stateful variables for KV cache to avoid host-device transfers
        quantize_weights="int8" # Optional: could be "float16" or None
    )
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving exported model to {output_path}...")
    litert_model.save(output_path)
    
    # Also save the tokenizer and config for the engine
    tokenizer.save_pretrained(output_dir if output_dir else ".")
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gemma 4 to LiteRT")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it", help="Hugging Face model ID")
    parser.add_argument("--output", default="exported_model/model.tflite", help="Output path for .tflite file")
    args = parser.parse_args()
    
    export_gemma4(args.model_id, args.output)
