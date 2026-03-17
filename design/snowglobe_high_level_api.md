# Design Doc: Snowglobe High-Level Flutter Library (`snowglobe_ai`)

## Objective
Create a high-level Flutter library that wraps the Snowglobe Rust engine and provides an API compatible with `firebase_vertexai`. This allows developers to switch between local and cloud models with minimal code changes.

## Background
Snowglobe provides high-performance local LLM inference. Firebase AI Logic (via `firebase_vertexai`) provides cloud-based Gemini inference. Currently, Snowglobe's API is low-level and requires manual session and state management. A high-level wrapper will improve developer experience and enable easy integration into existing Firebase-powered apps.

## Feature Support Matrix

| Feature | `firebase_vertexai` (Gemini) | `snowglobe_ai` (Local) |
| :--- | :--- | :--- |
| **Text Generation** | Supported | Supported |
| **Streaming** | Supported | Supported |
| **Chat Sessions** | Supported | Supported |
| **Multimodal (Images)** | Supported | Planned (Qwen 3.5 VL) |
| **Count Tokens** | Supported | Supported (via Engine) |
| **Safety Settings** | Supported | Basic (via Prompting) |
| **Generation Config** | Supported | Limited Support |
| **Function Calling** | Supported | Not Supported |
| **Offline Mode** | Not Supported | Supported |

## Proposed API Surface

The `snowglobe_ai` package will mirror the classes and methods found in `firebase_vertexai`.

### 1. `GenerativeModel` (Local Implementation)

```dart
class SnowglobeGenerativeModel implements GenerativeModel {
  final String model;
  final List<SafetySetting>? safetySettings;
  final GenerationConfig? generationConfig;
  final List<Tool>? tools;
  final ToolConfig? toolConfig;
  final SystemInstruction? systemInstruction;

  SnowglobeGenerativeModel({
    required this.model,
    this.safetySettings,
    this.generationConfig,
    this.tools,
    this.toolConfig,
    this.systemInstruction,
  });

  @override
  Future<GenerateContentResponse> generateContent(
    Iterable<Content> content, {
    List<SafetySetting>? safetySettings,
    GenerationConfig? generationConfig,
    List<Tool>? tools,
    ToolConfig? toolConfig,
  }) async {
    // Implementation using Snowglobe engine
  }

  @override
  Stream<GenerateContentResponse> generateContentStream(
    Iterable<Content> content, {
    List<SafetySetting>? safetySettings,
    GenerationConfig? generationConfig,
    List<Tool>? tools,
    ToolConfig? toolConfig,
  }) {
    // Implementation using Snowglobe engine
  }

  @override
  ChatSession startChat({
    List<Content>? history,
    List<SafetySetting>? safetySettings,
    GenerationConfig? generationConfig,
    List<Tool>? tools,
    ToolConfig? toolConfig,
  }) {
    return SnowglobeChatSession(
      model: this,
      history: history?.toList() ?? [],
      generationConfig: generationConfig,
    );
  }

  @override
  Future<CountTokensResponse> countTokens(Iterable<Content> content) async {
    // Implementation using tokenizer in engine
  }
}
```

### 2. `ChatSession` (Local Implementation)

```dart
class SnowglobeChatSession implements ChatSession {
  final SnowglobeGenerativeModel model;
  final List<Content> history;
  final GenerationConfig? generationConfig;
  String? _sessionId;

  SnowglobeChatSession({
    required this.model,
    required this.history,
    this.generationConfig,
  });

  @override
  Future<GenerateContentResponse> sendMessage(Content content) async {
    // 1. Ensure _sessionId is initialized
    // 2. Add content to history
    // 3. Call model.generateContent with full history
    // 4. Update history with response
  }

  @override
  Stream<GenerateContentResponse> sendMessageStream(Content content) {
    // 1. Ensure _sessionId is initialized
    // 2. Add content to history
    // 3. Call model.generateContentStream
    // 4. Collect response chunks and update history
  }
}
```

### 3. Core Data Types

The library will reuse or provide compatible versions of:
- `Content`: A list of `Part` objects with a `role` (user/model).
- `Part`: `TextPart` or `DataPart` (for multimodal, though local might only support text initially).
- `GenerateContentResponse`: Contains `Candidate`s and `PromptFeedback`.
- `Candidate`: Contains the generated `Content`.

## Engine Integration

The `SnowglobeGenerativeModel` will handle the interaction with the Rust engine:

1.  **Initialization**: The first call to any generative method will trigger `initEngine` if not already initialized. It will use a default configuration or one provided via a global `Snowglobe.initialize()` call.
2.  **Session Mapping**: `ChatSession` maps directly to the engine's `sessionId`.
3.  **Prompt Construction**: The `Iterable<Content>` (history + new message) will be flattened into a prompt string. Since the engine currently supports Qwen-style ChatML (`<|im_start|>user
...<|im_end|>`), the library will handle this formatting.
4.  **Streaming**: The `Stream<String>` from `generateResponse` in Rust will be mapped to `Stream<GenerateContentResponse>`.

## Switching Between Firebase and Snowglobe

A unified factory or provider can be used to switch between the two:

```dart
class AIService {
  static GenerativeModel getModel({bool preferLocal = true}) {
    if (preferLocal && Snowglobe.isSupported) {
      return SnowglobeGenerativeModel(model: 'qwen3.5');
    } else {
      return FirebaseVertexAI.instance.generativeModel(model: 'gemini-1.5-flash');
    }
  }
}
```

Since both implement the same `GenerativeModel` interface, the rest of the application code remains unchanged.

## Implementation Details

### Package Structure
- `lib/src/model.dart`: `SnowglobeGenerativeModel` implementation.
- `lib/src/session.dart`: `SnowglobeChatSession` implementation.
- `lib/src/utils.dart`: Prompt formatting, type conversions.
- `lib/snowglobe_ai.dart`: Public exports.

### Challenges & Solutions
- **Model Files**: Local models need to be downloaded. `SnowglobeGenerativeModel` should handle or delegate model management (downloading/caching).
- **Concurrency**: The engine might only support one active session/generation at a time depending on the backend. The library should handle queuing or return errors gracefully.
- **Parameters**: Parameters like `temperature` and `topP` in `GenerationConfig` need to be supported by the engine. Currently, the engine API is limited; it may need expansion.

## Model Management & Initialization

Before using `SnowglobeGenerativeModel`, the engine must be initialized with the correct model files.

```dart
await Snowglobe.initialize(
  model: 'qwen3.5',
  cacheDir: await getApplicationSupportDirectory(),
  // Optional: Auto-download if missing
  autoDownload: true,
);
```

## Prompt Formatting (ChatML)

The `snowglobe_ai` library will automatically convert the `Iterable<Content>` into the ChatML format expected by Qwen models:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 1+1?<|im_end|>
<|im_start|>assistant
1+1 is 2.<|im_end|>
<|im_start|>user
And what is 2+2?<|im_end|>
<|im_start|>assistant
```

This translation layer ensures that the developer doesn't need to worry about the underlying model's specific prompt template.

## Error Handling

The library will map engine errors to standard exceptions:
- `UnsupportedModelException`: If the requested model is not downloaded or supported.
- `EngineInitializationException`: If the Rust engine fails to start (e.g., incompatible hardware).
- `InferenceException`: If an error occurs during token generation.
