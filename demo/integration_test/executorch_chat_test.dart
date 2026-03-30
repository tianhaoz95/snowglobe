import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobedemo/main.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';
import 'package:openai_dart/openai_dart.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('ExecuTorch backend: direct chat completion',
      timeout: const Timeout(Duration(minutes: 30)),
      (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    // Wait for initial engine to be ready
    for (int i = 0; i < 120; i++) {
      await tester.pump(const Duration(seconds: 1));
      if (find.text("System ready. Let's chat!").evaluate().isNotEmpty) break;
      if (find.textContaining('Failed to initialize engine').evaluate().isNotEmpty) break;
    }

    // Switch to ExecuTorch backend
    final appState = tester.state<MyAppState>(find.byType(MyApp));
    appState.onBackendChanged(InferenceBackend.executorch);
    await tester.pump();

    // Wait for ExecuTorch engine to be ready
    bool engineReady = false;
    for (int i = 0; i < 120; i++) {
      await tester.pump(const Duration(seconds: 1));
      if (find.text("System ready. Let's chat!").evaluate().isNotEmpty) {
        engineReady = true;
        break;
      }
      final errorFinder = find.textContaining('Failed to initialize engine');
      if (errorFinder.evaluate().isNotEmpty) {
        fail('ExecuTorch engine init failed: ${tester.widget<Text>(errorFinder.first).data}');
      }
    }
    expect(engineReady, true, reason: 'ExecuTorch engine did not become ready within 120 seconds');

    final backendString = await SnowglobeOpenAI.checkBackend();
    final modelInfo = await SnowglobeOpenAI.getModelInfo();
    print('------------------------------------------------------------');
    print('EXECUTORCH CHAT TEST - RUNTIME INFO');
    print('Backend: $backendString | Runner: ${modelInfo?.runner ?? "Unknown"}');
    print('------------------------------------------------------------');

    // Use direct (non-streaming) chat completion
    print('Calling createChatCompletion directly (may take several minutes on CPU)...');
    final stopwatch = Stopwatch()..start();

    final response = await SnowglobeOpenAI.createChatCompletion(
      CreateChatCompletionRequest(
        model: ChatCompletionModel.modelId('snowglobe'),
        messages: [
          ChatCompletionMessage.user(
            content: ChatCompletionUserMessageContent.string('what is the capital of China?'),
          ),
        ],
        maxTokens: 32,
      ),
    );

    stopwatch.stop();
    final responseText = response.choices.first.message.content ?? '';
    final tokPerSec = (responseText.split(' ').length / stopwatch.elapsed.inSeconds).toStringAsFixed(2);
    print('Response in ${stopwatch.elapsed.inSeconds}s (~$tokPerSec tok/s): "$responseText"');

    expect(responseText.isNotEmpty, true, reason: 'ExecuTorch produced no response');
    expect(responseText.toLowerCase(), contains('beijing'));
    print('EXECUTORCH CHAT TEST - SUCCESS');
  });
}
