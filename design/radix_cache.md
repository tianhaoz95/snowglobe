# Research Report: Radix Cache for Snowglobe

## Executive Summary
This report explores the implementation of a **Radix Cache** (also known as Prefix Cache) for the Snowglobe inference engine. Radix caching is a technique to reuse Key-Value (KV) cache tensors across multiple inference requests or conversation turns that share a common prefix (e.g., system prompts, few-shot examples, or previous chat history). Implementing this feature can significantly reduce the **Time to First Token (TTFT)** for multi-turn conversations and multi-user scenarios.

## 1. Problem Statement
In the current Snowglobe architecture, each `EngineSession` manages its own private KV cache. When a new turn starts or a new session is initialized with a common system prompt, the model must re-process the entire prompt (the "prefill" phase) to generate the KV cache. This leads to:
1.  **Redundant Computation:** Re-computing KV states for the same tokens (e.g., "You are a helpful assistant...") across every session.
2.  **Increased Latency:** High TTFT for long prompts, even if they have been processed before.
3.  **Memory Inefficiency:** Duplicate copies of the same KV cache data for multiple active sessions.

## 2. Theoretical Background

### 2.1 Prefill vs. Generation
- **Prefill Phase:** The model processes the entire input prompt in a single forward pass, generating KV cache tensors for all tokens. This is compute-bound.
- **Generation Phase:** The model generates one token at a time, using the KV cache from previous steps. This is memory-bandwidth-bound.

### 2.2 Radix Cache (RadixAttention)
Popularized by frameworks like **SGLang**, RadixAttention uses a **Radix Tree** (or Prefix Tree) to manage KV caches.
- **Trie of Tokens:** The cache is organized as a tree where each node represents a sequence of tokens and stores the corresponding KV cache tensors.
- **Prefix Matching:** When a new prompt is received, the engine finds the longest prefix already present in the trie.
- **Partial Prefill:** The engine only performs the prefill for the *new* tokens, using the cached KV tensors for the prefix.
- **LRU Eviction:** When memory is full, the least recently used branches of the trie are evicted.

## 3. Current Snowglobe Architecture
- **KV Cache:** Tensors are stored in a `Vec<KVCache<B>>`, where each `KVCache` contains `key` and `value` tensors of shape `[batch, heads, seq_len, head_dim]`.
- **Cache Growth:** Snowglobe currently uses `Tensor::cat` to append new KV states to the existing cache during each turn.
- **Session State:** Stored in `EngineSession` as `Box<dyn Any>`, typically downcast to `Vec<KVCache<B>>`.

## 4. Proposed Implementation Strategy

### 4.1 Radix Cache Structure
Introduce a `RadixCache` structure in `engine/src/model/mod.rs` or a new `engine/src/utils/cache.rs`.

```rust
struct RadixNode<B: Backend> {
    tokens: Vec<u32>,
    kv_cache: Option<Arc<Vec<KVCache<B>>>>,
    children: HashMap<u32, Box<RadixNode<B>>>,
    last_accessed: Instant,
}
```

### 4.2 Integration Workflow
1.  **Prefix Lookup:** Before starting a `generate_response` call, the `SESSIONS` manager or a global `RadixCacheManager` should search for the longest matching prefix of the input tokens.
2.  **State Initialization:**
    - If a prefix match is found (e.g., length `L`), initialize the `EngineSession` with the cached `Vec<KVCache<B>>` and set `offset = L`.
    - If no match is found, initialize a fresh session with `offset = 0`.
3.  **Partial Forward Pass:**
    - Pass only the `L..N` tokens to the model's `forward` function.
    - The model will concatenate these new tokens with the provided cache.
4.  **Cache Update:**
    - After the prefill of new tokens is complete, the resulting KV cache (or the newly computed segment) is inserted back into the `RadixCache` trie for future reuse.

### 4.3 Memory Management
- **Reference Counting:** Use `Arc` to share KV tensors between the `RadixCache` and active `EngineSession` instances.
- **LRU Policy:** Implement an eviction policy based on `last_accessed` timestamps to keep the memory usage within a configurable limit.

## 5. Challenges and Considerations

### 5.1 Burn Tensor Contiguity
Snowglobe's `QwenAttention` expects a single contiguous tensor for the entire KV history to perform efficient `matmul`. While `Tensor::cat` provides this, it creates a new allocation and copies the data.
- **Optimization:** If possible, explore if the `RadixCache` can store the KV cache in "blocks" (e.g., 16 tokens each) and if the attention mechanism can be modified to handle a list of blocks (similar to PagedAttention).
- **Interim Solution:** Even with `Tensor::cat`, we still save the **computation** of the prefill phase, which is the primary goal for reducing TTFT.

### 5.2 RoPE and Offset
Ensure that the `offset` parameter is correctly passed to `apply_rotary_pos_emb` so that the new tokens have the correct positional embeddings relative to the cached prefix.

### 5.3 Cache Granularity
Decide whether to cache every unique prefix or only "significant" ones (e.g., system prompts, end-of-turn points). Caching at every token might create excessive trie nodes and memory overhead.

## 6. Expected Performance Impact
- **System Prompt Reuse:** For a 200-token system prompt, the TTFT could drop from ~100ms to <10ms (depending on hardware).
- **Multi-turn Chat:** In a conversation with 5 turns, the 6th turn only pays the prefill cost for the latest user message, rather than the entire history.
- **Concurrency:** Multiple concurrent users sharing a system prompt will only trigger a single prefill.

## 7. Recommendation
Implement a simplified Radix Cache that focuses on **segment-based caching** (caching full conversation turns) first. This provides the most significant "bang for the buck" without requiring a complete rewrite of the attention kernels to support PagedAttention.

---
**Author:** Gemini CLI Agent
**Date:** March 11, 2026
