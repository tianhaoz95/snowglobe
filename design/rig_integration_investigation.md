# Rig Agent Framework Integration Investigation

## 1. Overview
[Rig](https://github.com/0xPanda/rig) is a modular, ergonomic Rust framework for building LLM-powered applications. It provides high-level abstractions for building AI agents, handling tool calling, and implementing RAG (Retrieval-Augmented Generation).

Integrating Rig into Snowglobe would enable:
- **Agentic Workflows**: Multi-step reasoning and autonomous task execution.
- **Tool Use**: Allowing the local Qwen model to interact with the host system (e.g., file system, shell).
- **Extensible RAG**: Using Rig's unified vector store abstractions to provide long-term memory.

## 2. Current Architecture
Snowglobe currently provides a raw inference API:
- `init()`: Loads model weights into Burn (local) or ExecuTorch.
- `generate_response()`: Streams tokens from a prompt.

The interaction is handled via a simple `StreamSink` which is bridged to Flutter.

## 3. Integration Strategy

### 3.1 Implementing `CompletionModel`
Rig's `CompletionModel` trait is the primary integration point. We would wrap Snowglobe's `LoadedModel` to implement this trait. Rig also supports streaming via the `StreamingCompletion` trait, which is essential for a responsive UI.

```rust
pub struct SnowglobeModel;

#[async_trait]
impl CompletionModel for SnowglobeModel {
    type Response = String;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let mut full_response = String::new();
        let mut stream = self.stream_completion(request).await?;

        while let Some(chunk) = stream.next().await {
            full_response.push_str(&chunk?.choice);
        }

        Ok(CompletionResponse {
            choice: full_response,
            ..Default::default()
        })
    }
}

#[async_trait]
impl StreamingCompletion for SnowglobeModel {
    async fn stream_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, CompletionError>> + Send>>,
        CompletionError,
    > {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        // Spawn the generation in a separate task
        tokio::spawn(async move {
            struct RigSink(tokio::sync::mpsc::Sender<Result<CompletionChunk, CompletionError>>);
            impl snowglobe::StreamSink<String> for RigSink {
                fn add(&self, value: String) -> bool {
                    let _ = self.0.try_send(Ok(CompletionChunk {
                        choice: value,
                        ..Default::default()
                    }));
                    true
                }
            }
            
            let session_id = snowglobe::init_session();
            if let Err(e) = snowglobe::generate_response(&session_id, &request.prompt, 512, RigSink(tx)) {
                // Handle error
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}
```

### 3.2 Agent Orchestration
Once `CompletionModel` is implemented, we can create agents in the `demo` crate:

```rust
let agent = Rig::agent(SnowglobeModel)
    .preamble("You are a local system assistant.")
    .tool(FileSystemTool::new())
    .build();

let response = agent.prompt("What is in my current directory?").await?;
```

### 3.3 Flutter Integration
The `demo/rust/src/api/simple.rs` would be extended with agent-related functions. Instead of just sending a prompt, Flutter could send a "Task", and the Rust agent would execute it, potentially performing multiple tool calls before returning the final answer.

### 3.4 Multi-Backend Transparency
One of the strengths of Snowglobe is its dual-backend support (Burn and ExecuTorch). The Rig integration should happen at the `LoadedModel` level, ensuring that Rig Agents remain agnostic to whether the inference is running on Burn's WGPU backend or Meta's ExecuTorch runtime. This allows us to switch backends for performance or compatibility without changing the agent logic.

## 4. Proposed Implementation Roadmap

1.  **Dependency Addition**:
    Add `rig-core` to `engine/Cargo.toml`.
2.  **Engine Trait Implementations**:
    Create `engine/src/rig.rs` to house the `CompletionModel` and `EmbeddingModel` implementations.
3.  **Local Tool Library**:
    Define a set of safe, local tools that the agent can use (e.g., `FileRead`, `CurrentTime`, `SystemStats`).
4.  **Vector Store Integration**:
    Optionally add a local vector store like `LanceDB` to enable RAG features.
5.  **UI Enhancements**:
    Update the Flutter UI to display "Thinking" states and "Tool Use" logs to give the user visibility into the agent's actions.

## 5. Potential Challenges
- **Latency**: Agentic loops involve multiple LLM passes, which might be slow on mobile devices.
- **Context Window**: Tool definitions and reasoning traces consume tokens, requiring efficient KV cache management (already partially implemented in Snowglobe).
- **Concurrency**: Managing multiple agent sessions on limited mobile hardware.

## 6. Conclusion
Rig provides the necessary abstractions to elevate Snowglobe from a simple chatbot to a functional AI agent platform. By implementing the `CompletionModel` trait, we can leverage Rig's entire ecosystem of agents and tools while keeping the computation entirely local.
