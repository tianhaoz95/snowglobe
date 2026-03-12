# Research Report: Unified AI Adapter for Snowglobe and Firebase AI Logic

## Introduction

This report proposes a unified Flutter API to abstract the differences between **Snowglobe** (local LLM inference) and **Firebase AI Logic** (cloud-based Gemini inference). The goal is to allow developers to switch between local and cloud inference seamlessly or even use a hybrid approach (e.g., local by default, cloud for complex tasks).

## API Comparison

### Snowglobe (Local Inference)

Snowglobe uses a Rust-based engine with multiple backends (Burn, ExecuTorch, llama.cpp). Its API is session-based but relatively low-level:

| Feature | Snowglobe API |
| :--- | :--- |
| **Initialization** | `initEngine(cacheDir, config)` |
| **Session Management** | `initSession()` -> `sessionId` |
| **Inference (Stream)** | `generateResponse(sessionId, prompt, maxGenLen)` |
| **Backends** | llama.cpp (GGUF), ExecuTorch (PTE), Burn (Safetensors) |

### Firebase AI Logic (Cloud Inference)

Firebase AI Logic provides a high-level SDK for interacting with Gemini models:

| Feature | Firebase AI Logic API |
| :--- | :--- |
| **Initialization** | `FirebaseAI.googleAI().generativeModel(model: '...')` |
| **Session Management** | `model.startChat(history: [...])` -> `ChatSession` |
| **Inference (Stream)** | `chatSession.sendMessageStream(prompt)` |
| **Models** | Gemini 2.5 Flash, Gemini 1.5 Pro, etc. |

## Proposed Unified API: `snowglobe_adapter`

The proposed architecture introduces an abstract `AIProvider` that can be implemented for both Snowglobe and Firebase.

### 1. Core Interfaces

```dart
/// Base class for all AI providers.
abstract class AIProvider {
  /// Initializes the provider (e.g., downloading models for Snowglobe, 
  /// checking Firebase connection for Firebase AI).
  Future<void> initialize();

  /// Simple one-off generation.
  Stream<String> generateStream(String prompt, {AIConfig? config});

  /// Starts a multi-turn conversation.
  AIChatSession startChat({List<AIChatMessage>? history});
  
  /// Get provider metadata (e.g., name, local vs cloud).
  AIProviderInfo get info;
}

/// Represents a multi-turn chat session.
abstract class AIChatSession {
  /// Sends a message and receives a streaming response.
  Stream<String> sendMessageStream(String prompt, {AIConfig? config});
  
  /// The history of the current session.
  List<AIChatMessage> get history;
}

class AIChatMessage {
  final AIRole role;
  final String text;
  
  AIChatMessage({required this.role, required this.text});
}

enum AIRole { user, model }

class AIConfig {
  final int? maxTokens;
  final double? temperature;
  final double? topP;
  
  AIConfig({this.maxTokens, this.temperature, this.topP});
}

class AIProviderInfo {
  final String name;
  final bool isLocal;
  
  AIProviderInfo({required this.name, required this.isLocal});
}
```

### 2. Implementation Strategies

#### Snowglobe Implementation (`SnowglobeProvider`)
- **Initialization**: Downloads required model files to `getApplicationSupportDirectory()` and calls `initEngine`.
- **Chat Session**: Calls `initSession` on start and maintains a local list of `AIChatMessage`. Every `sendMessageStream` calls the Rust `generateResponse` with the same `sessionId`.

#### Firebase Implementation (`FirebaseAIProvider`)
- **Initialization**: Ensures `Firebase.initializeApp()` has been called.
- **Chat Session**: Wraps the `ChatSession` from `firebase_ai` package. Maps `AIChatMessage` to `Content` objects.

### 3. Factory/Registry

To simplify usage, a factory can be used to switch providers:

```dart
class AIService {
  static AIProvider create({
    required AIProviderType type,
    SnowglobeConfig? snowglobeConfig,
    FirebaseConfig? firebaseConfig,
  }) {
    switch (type) {
      case AIProviderType.snowglobe:
        return SnowglobeProvider(snowglobeConfig);
      case AIProviderType.firebase:
        return FirebaseAIProvider(firebaseConfig);
    }
  }
}

enum AIProviderType { snowglobe, firebase }
```

## Benefits of the Unified API

1.  **Code Reusability**: UI components can be written to accept an `AIProvider` without caring where the inference happens.
2.  **Hybrid Fallback**: An app can attempt to use `SnowglobeProvider` (local) for privacy and offline support, then fallback to `FirebaseAIProvider` (cloud) if the device is underpowered or the local model fails to initialize.
3.  **Cost Optimization**: Use local inference for simple tasks and cloud inference for complex reasoning, saving on API costs.
4.  **Testing**: Easy to mock `AIProvider` for unit and widget tests.

## Next Steps

1.  Create a new Flutter package (e.g., `packages/snowglobe_adapter`).
2.  Define the abstract interfaces in a `core/` directory.
3.  Implement `SnowglobeProvider` using the existing `demo/lib/src/rust` bridge.
4.  Implement `FirebaseAIProvider` using the `firebase_ai` package.
5.  Provide a unified `ChatWidget` or similar component in the adapter package for rapid development.
