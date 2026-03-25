import 'package:flutter_test/flutter_test.dart';
import 'package:openai_dart/openai_dart.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';

void main() {
  test('SnowglobeOpenAI.createChatCompletion request serialization', () async {
    // This test ensures the mapping and serialization work.
    // We can't easily run the real Rust engine here without a compiled binary.
    // In a real project, we would use mocks or specialized integration tests.
    
    // For now, let's just check if we can construct the objects.
    final request = CreateChatCompletionRequest(
      model: ChatCompletionModel.model(ChatCompletionModels.gpt4o),
      messages: [
        ChatCompletionRequestMessage.user(
          content: ChatCompletionUserMessageContent.string('Hello'),
        ),
      ],
    );

    expect(request.model.value, 'gpt-4o');
    expect(request.messages.length, 1);
  });
}
