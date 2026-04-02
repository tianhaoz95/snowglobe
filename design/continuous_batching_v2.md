# Design: Continuous Batching with V2 Model Runner

## 1. Overview
This document describes the implementation of a continuous batching inference engine using the V2 Model Runner interface. Continuous batching (or iteration-level batching) allows for multiple requests to be processed concurrently by dynamically adding and removing requests from the active batch at each iteration.

## 2. Reference: Mini-SGLang
The `mini-sglang` implementation provides a robust model for continuous batching:
- **Scheduler**: Manages a pool of requests and determines which tokens to process in each step (Prefill vs. Decode).
- **Page Table / Radix Cache**: Manages KV cache memory as blocks or pages, allowing for efficient prefix sharing and cache reuse.
- **Overlap Scheduling**: Overlaps CPU-side metadata processing (scheduling) with GPU-side tensor computation to hide latency.

## 3. Architecture for Snowglobe (V2 Interface)
The Snowglobe engine will implement a background loop that interacts with the `ModelRunner` V2 interface.

### 3.1. Background Loop
A dedicated thread or async task will run the "Forward Loop":
```rust
loop {
    let requests = request_queue.get_active_batch();
    if requests.is_empty() {
        // Sleep or wait for new request event
        park_thread_until_ready();
        continue;
    }

    // Prepare combined batch input
    let (combined_input, batch_metadata) = prepare_batch(requests);

    // One iteration forward pass
    let logits = runner.execute(&mut global_batch_session, &combined_input, mode);

    // Process results & update request states
    update_requests(requests, logits, batch_metadata);
}
```

### 3.2. Mapping to V2 Interface
The `ModelRunner::execute` method will be the core primitive:
- **Input**: A flattened slice of tokens representing the current step for all active requests.
- **State**: The `EngineSession` will now represent a *global batch state*, containing the concatenated KV caches for all active requests.
- **Truncation**: `truncate_cache` will be used when a request finishes to "compact" the global cache or when a request needs to be re-prefilled from a specific point.

### 3.3. Request State Management
Each request in the queue will maintain:
- `tokens`: The full sequence of tokens generated so far.
- `kv_offset`: Its current position/length in the global KV cache.
- `status`: (Pending, Prefilling, Decoding, Finished).

### 3.4. Cache Management (Paged KV Cache)
To support dynamic batching, the `EngineSession`'s `backend_state` must evolve from a simple `Vec<KVCache>` to a paged structure:
- **Logical to Physical Mapping**: Map request-specific sequence positions to physical blocks in memory.
- **Sharing**: If two requests share a system prompt, they should point to the same physical KV blocks.

## 4. Implementation Steps
1. **Request Queue**: Implement a thread-safe queue for incoming Chat completion requests.
2. **Scheduler**: Implement logic to select requests for the next iteration (prioritizing finishing decodes, then starting new prefills).
3. **Batch Executor**: A wrapper that groups individual requests into a single `runner.execute` call.
4. **Response Streamer**: Pipes tokens back to the client (e.g., Flutter UI) as they are generated.
