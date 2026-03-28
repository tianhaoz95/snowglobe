import sys
from pathlib import Path
# Add executorch root to sys.path
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "executorch"))

import logging
import torch._logging

# 1. Tell PyTorch's internal C++ / Dynamo engine to shut up
torch._logging.set_logs(all=logging.ERROR)

# 2. Blanket disable for any rogue Python loggers
logging.disable(logging.CRITICAL)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import os
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner, CompileSpec as MPSCompileSpec
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

qnn_import_error = None
try:
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_qnn_executorch_compiler_spec,
        generate_htp_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
except ImportError as e:
    QnnPartitioner = None
    qnn_import_error = e

from transformers import AutoTokenizer

def quantize_model(model, example_inputs, backend):
    print(f"--- Starting Quantization for {backend} ---")
    
    # 1. Capture the model
    with torch.no_grad():
        m = export(model, example_inputs).module()

    # 2. Select Quantizer
    if backend == "qualcomm":
        if QnnQuantizer is None:
            print("Qualcomm Quantizer not available, skipping quantization.")
            return model
        quantizer = QnnQuantizer()
        # Configure for INT8 (8-bit activations, 8-bit weights)
        quantizer.add_16bit_output_nodes([]) # Optional
    else:
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)

    # 3. Prepare
    print("Preparing model for PT2E quantization...")
    m = prepare_pt2e(m, quantizer)

    # 4. Calibration
    print("Calibrating with sample inputs...")
    # Use a few dummy tokens for calibration
    # In a real scenario, use actual prompt data
    for _ in range(8):
        m(*example_inputs)

    # 5. Convert
    print("Converting to quantized model...")
    m = convert_pt2e(m)
    
    print("Quantization complete.")
    return m

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # Use float32 for stable variance calculation - REQUIRED for quality
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = (hidden_states_fp32 * hidden_states_fp32).mean(-1, keepdim=True)
        hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states_fp32.to(input_dtype)

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=40960, base=1000000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for fixed 128 sequence length
        t = torch.arange(128, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Register as float32 buffers to avoid alignment issues
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # Use precomputed buffers
        return self.cos_cached[:, :, :seq_len, :].to(x.dtype), self.sin_cached[:, :, :seq_len, :].to(x.dtype)

def rotate_half(x):
    # Use chunk instead of manual slicing for better delegate compatibility
    # Flatten to 2D temporarily to avoid 4D slice bugs in some MPS delegate versions
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    x1, x2 = x_flat.chunk(2, dim=-1)
    res = torch.cat((-x2, x1), dim=-1)
    return res.reshape(shape)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK Norm for Qwen3
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config["rms_norm_eps"])

        self.rotary_emb = Qwen3RotaryEmbedding(self.head_dim, config["max_position_embeddings"], config["rope_theta"])

        # Precompute causal mask for testing (reduced to 128)
        mask = torch.triu(torch.ones(128, 128), diagonal=1)
        # Use float32 explicitly for the mask to avoid alignment issues in MPS
        self.register_buffer("causal_mask", (mask * -10000.0).to(torch.float32), persistent=False)

    def forward(self, hidden_states, cos, sin):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply QK Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K/V for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, q_len, self.head_dim)
            value_states = value_states[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, q_len, self.head_dim)

        # Use 3D bmm for better delegate compatibility by flattening head into batch
        q = query_states.reshape(-1, q_len, self.head_dim)
        k = key_states.reshape(-1, q_len, self.head_dim).transpose(1, 2)
        attn_weights = torch.bmm(q, k) * (1.0 / math.sqrt(self.head_dim))
        
        # Additive causal mask in float32 for stability
        mask = self.causal_mask[:q_len, :q_len]
        attn_weights = attn_weights.to(torch.float32) + mask

        # Softmax in float32 for stability
        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        
        v = value_states.reshape(-1, q_len, self.head_dim)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim).permute(0, 2, 1, 3).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3Block(config) for _ in range(config["num_hidden_layers"])])
        self.norm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.rotary_emb = Qwen3RotaryEmbedding(config["head_dim"], config["max_position_embeddings"], config["rope_theta"])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        seq_len = input_ids.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        return self.norm(hidden_states)

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        # Cast back to float32 for the output to match our C++ adapter's expectation
        return logits.to(torch.float32)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["mps", "xnnpack", "qualcomm", "portable"], default="mps")
    parser.add_argument("--quantize", action="store_true", help="Perform static PTQ (INT8) before export.")
    args = parser.parse_args()

    repo_id = "Qwen/Qwen3-0.6B"
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loading weights for {repo_id}...")
    model_path = hf_hub_download(repo_id, "model.safetensors")
    state_dict = load_file(model_path)

    print("Initializing model...")
    # Use float16 for MPS and Qualcomm to avoid type mismatches and improve performance
    # BUT use float32 if we are going to quantize, as observers work better on float32
    model_dtype = torch.float32 if (args.quantize or args.backend not in ["mps", "qualcomm"]) else torch.float16
    model = Qwen3ForCausalLM(config).to(model_dtype)
    # Fix keys if necessary (Safetensors usually match but sometimes there's a prefix)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Running generation test ({model_dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt = "What is the capital of China?"
    # Use 128 for seq_len as per engine expectation
    example_input = torch.randint(0, config["vocab_size"], (1, 128), dtype=torch.int32 if args.backend == "mps" else torch.int64)

    # 0. Quantization (Optional)
    if args.quantize:
        if args.backend == "mps":
            print("❌ Static PTQ is not supported for MPS backend. Skipping quantization.")
        else:
            model = quantize_model(model, (example_input,), args.backend)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    generated_ids = input_ids
    max_new_tokens = 20
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # QwenPte logic expects fixed 128 input usually, but here we just test sanity
            # If we quantized, model might be a GraphModule now
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    assert "beijing" in response.lower(), f"Expected 'beijing' in response, but got: {response}"
    print("Test passed!")

    print(f"Exporting to PTE with backend: {args.backend}...")

    # 1. Export the model
    # If we quantized, model is now a GraphModule. We need to export it to get an ExportedProgram.
    with torch.no_grad():
        exported_model = export(model, (example_input,))
    if args.backend == "mps":
        print("Partitioning for MPS...")
        # 2. Lower to Edge
        edge_program = to_edge(exported_model, compile_config=EdgeCompileConfig(_check_ir_validity=True))
        edge_program = edge_program.to_backend(MPSPartitioner([]))
    elif args.backend == "qualcomm":
        print("Partitioning for Qualcomm (Snapdragon 8 Gen 2 / SM8550)...")
        if QnnPartitioner is None:
            print(f"\n❌ Error importing Qualcomm backend: {qnn_import_error}")
            print("\nNOTE: Qualcomm AOT model conversion requires:")
            print("1. A Linux host (Ubuntu 22.04+). macOS is NOT supported for Qualcomm AOT.")
            print("2. Building the ExecuTorch Qualcomm backend (including PyQnnManagerAdaptor).")
            print("3. QNN SDK installed and LD_LIBRARY_PATH set.")
            sys.exit(1)
        
        # HTP Compiler Configuration
        backend_options = generate_htp_compiler_spec(use_fp16=True)

        # QNN Compiler Spec
        compile_spec = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8550,
            backend_options=backend_options,
        )

        # Lower to QNN backend using the dedicated utility
        edge_program = to_edge_transform_and_lower_to_qnn(
            model,
            (example_input,),
            compile_spec
        )
    elif args.backend == "portable":
        print("Using Portable kernels (CPU fallback)...")
        # 2. Lower to Edge
        edge_program = to_edge(exported_model, compile_config=EdgeCompileConfig(_check_ir_validity=True))
        # No partitioning needed
    else:
        print("Partitioning for XNNPACK...")
        # 2. Lower to Edge
        edge_program = to_edge(exported_model, compile_config=EdgeCompileConfig(_check_ir_validity=True))
        edge_program = edge_program.to_backend(XnnpackPartitioner())

    print("\nAnalyzing delegation success...")
    delegated_nodes = 0
    fallback_nodes = {}
    for node in edge_program.exported_program().graph_module.graph.nodes:
        if node.op == "call_function":
            # Check if it was successfully delegated
            if "executorch_call_delegate" in str(node.target):
                delegated_nodes += 1
            # If it has an 'aten.' prefix, the backend rejected it
            elif "aten." in str(node.target):
                op_name = str(node.target)
                fallback_nodes[op_name] = fallback_nodes.get(op_name, 0) + 1

    print(f"\n--- DELEGATION SUMMARY ---")
    print(f"✅ Successfully delegated nodes: {delegated_nodes}")
    print(f"❌ Fallback CPU nodes: {sum(fallback_nodes.values())}")

    if fallback_nodes:
        print("\nTop rejected operations causing the slowdown:")
        # Print the top 10 most common rejected nodes
        for op, count in sorted(fallback_nodes.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {op}: {count} times")
    print("--------------------------\n")

    # 3. Export to PTE
    pte_filename = "qwen3_0.6b.pte"
    exec_prog = edge_program.to_executorch()
    with open(pte_filename, "wb") as f:
        exec_prog.write_to_file(f)

    print(f"Successfully exported to {pte_filename}")

if __name__ == "__main__":
    main()
