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
from executorch.backends.apple.mps.partition import MPSPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from transformers import AutoTokenizer

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=40960, base=1000000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

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

    def forward(self, hidden_states, cos, sin):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply QK Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K/V for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Causal mask (Additive approach is more stable for MPS backend)
        mask = torch.triu(torch.ones(q_len, q_len, device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights - (mask * 1e4)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
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
        return self.lm_head(hidden_states)

def main():
    repo_id = "Qwen/Qwen3-0.6B"
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loading weights for {repo_id}...")
    model_path = hf_hub_download(repo_id, "model.safetensors")
    state_dict = load_file(model_path)

    print("Initializing model...")
    model = Qwen3ForCausalLM(config).to(torch.float32)
    # Fix keys if necessary (Safetensors usually match but sometimes there's a prefix)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Running generation test...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt = "What is the capital of China?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    generated_ids = input_ids
    max_new_tokens = 20
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
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

    print("Exporting to PTE...")
    # Example input
    example_input = torch.randint(0, config["vocab_size"], (1, 128))
    
    # 1. Export the model
    with torch.no_grad():
        exported_model = export(model, (example_input,))

    # 2. Lower to Edge
    edge_program = to_edge(exported_model, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    # 2.1 Partition for MPS (Metal)
    # MPS backend currently has type-matching issues with causal masks in some environments.
    # Disabling by default to ensure reliability on CPU/XNNPACK.
    # print("Partitioning for MPS...")
    # edge_program = edge_program.to_backend(MPSPartitioner([]))
    edge_program = edge_program.to_backend(XnnpackPartitioner())

    # 3. Export to PTE
    pte_filename = "qwen3_0.6b.pte"
    exec_prog = edge_program.to_executorch()
    with open(pte_filename, "wb") as f:
        exec_prog.write_to_file(f)

    print(f"Successfully exported to {pte_filename}")

if __name__ == "__main__":
    main()
