# snowglobe_openai

A Flutter package for high-performance, cross-platform LLM inference with an OpenAI-compatible API.

## Features

- High-performance Rust engine (snowglobe) integrated via `flutter_rust_bridge`.
- OpenAI-compatible API using the `openai_dart` package types.
- Supports initialization and streaming chat completions.

## Getting Started

### Prerequisites

- Flutter SDK
- Rust (for building the native library)

### Initialization

```dart
import 'package:snowglobe_openai/snowglobe_openai.dart';

void main() async {
  // Initialize the bridge
  await SnowglobeOpenAI.initRust();

  // Initialize the engine with a model directory
  await SnowglobeOpenAI.initEngine(
    cacheDir: '/path/to/model/cache',
    config: InitConfig(
      vocabShards: 1,
      maxGenLen: 1024,
      useExecutorch: false,
      backend: BackendType.LlamaCpp,
      speculateTokens: 0,
    ),
  );
}
```

### Usage

```dart
import 'package:openai_dart/openai_dart.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';

void example() async {
  final request = CreateChatCompletionRequest(
    model: ChatCompletionModel.model(ChatCompletionModels.gpt4o),
    messages: [
      ChatCompletionRequestMessage.user(
        content: ChatCompletionUserMessageContent.string('Hello, how are you?'),
      ),
    ],
  );

  // Unary completion
  final response = await SnowglobeOpenAI.createChatCompletion(request);
  print(response.choices.first.message.content);

  // Streaming completion
  final stream = SnowglobeOpenAI.createChatCompletionStream(request);
  await for (final chunk in stream) {
    print(chunk.choices.first.delta.content);
  }
}
```

## Testing

Run tests with `flutter test`.

Note: Integration tests with the real engine require a compiled Rust library and model files.
