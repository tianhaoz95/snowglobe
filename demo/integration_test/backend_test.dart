import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobedemo/main.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Snowglobe Backend & Speculative Decoding Tests', () {
    
    final backends = [
      InferenceBackend.llamaCpp,
      InferenceBackend.liteRT,
    ];
    
    final speculativeModes = [false, true];

    for (var backend in backends) {
      for (var useSpeculative in speculativeModes) {
        testWidgets('Test backend: ${backend.name}, Speculative: $useSpeculative', (WidgetTester tester) async {
          print('------------------------------------------------------------');
          print('STARTING TEST: Backend: ${backend.name}, Speculative: $useSpeculative');
          print('------------------------------------------------------------');
          
          // Start the app
          await tester.pumpWidget(const MyApp());
          await tester.pumpAndSettle();

          // Select the backend
          print('Selecting backend: ${backend.name}');
          final backendDropdown = find.byKey(const Key('backend_dropdown'));
          expect(backendDropdown, findsOneWidget);
          
          final MyAppState state = tester.state<MyAppState>(find.byType(MyApp));
          state.onBackendChanged(backend);
          await tester.pumpAndSettle();

          // Toggle speculative decoding
          print('Setting speculative decoding: $useSpeculative');
          final switchFinder = find.byType(Switch);
          expect(switchFinder, findsOneWidget);
          
          final Switch speculativeSwitch = tester.widget<Switch>(switchFinder.first);
          if (speculativeSwitch.value != useSpeculative) {
            await tester.tap(switchFinder.first);
            await tester.pumpAndSettle();
          }

          // 1. Wait for engine to be ready
          bool engineReady = false;
          final readinessStopwatch = Stopwatch()..start();
          for (int i = 0; i < 120; i++) {
            await tester.pump(const Duration(seconds: 1));
            if (find.text("System ready. Let's chat!").evaluate().isNotEmpty) {
              engineReady = true;
              break;
            }
            
            final errorFinder = find.textContaining('Failed to initialize engine');
            if (errorFinder.evaluate().isNotEmpty) {
              final errorText = tester.widget<Text>(errorFinder.first).data;
              fail('Engine initialization failed: $errorText');
            }
          }
          readinessStopwatch.stop();
          print('Engine readiness time: ${readinessStopwatch.elapsed.inSeconds}s');
          if (readinessStopwatch.elapsed.inSeconds > 10) {
            print('WARNING: Readiness took longer than 10s. It might be downloading instead of side-loading.');
          }

          expect(engineReady, true, reason: 'Engine did not become ready for backend ${backend.name}');

          // 2. Log Runtime Info
          final backendString = await SnowglobeOpenAI.checkBackend();
          final modelInfo = await SnowglobeOpenAI.getModelInfo();
          print('CHAT TEST - RUNTIME INFO:');
          print('Hardware Backend: $backendString');
          print('Runner Framework: ${modelInfo?.runner ?? "Unknown"}');
          
          // 3. Send First Prompt
          print('Sending first prompt: what is the capital of China?');
          await tester.enterText(find.byType(TextField), 'what is the capital of China?');
          await tester.testTextInput.receiveAction(TextInputAction.done);
          await tester.pumpAndSettle();
          await Future.delayed(const Duration(seconds: 1));
          await _sendPromptAndVerify(tester, 'Beijing', exactMatch: false);

          // Give it a breather
          await Future.delayed(const Duration(seconds: 2));

          // 4. Send Second Prompt
          print('Sending second prompt: what is 1+1? Answer with only number');
          await tester.enterText(find.byType(TextField), 'what is 1+1? Answer with only number');
          await tester.testTextInput.receiveAction(TextInputAction.done);
          await tester.pumpAndSettle();
          await Future.delayed(const Duration(seconds: 1));
          await _sendPromptAndVerify(tester, '2', exactMatch: false);
          
          print('TEST SUCCESS: Backend: ${backend.name}, Speculative: $useSpeculative');
          print('------------------------------------------------------------\n');
        });
      }
    }
  });
}

Future<void> _sendPromptAndVerify(WidgetTester tester, String expectedText, {bool exactMatch = false}) async {
  final sendButtonFinder = find.byIcon(Icons.send);
  expect(sendButtonFinder, findsOneWidget);
  
  print('Tapping send button...');
  await tester.tap(sendButtonFinder);
  await tester.pump();

  // 1. Wait for loading to start
  bool started = false;
  for (int i = 0; i < 20; i++) {
    await tester.pump(const Duration(milliseconds: 100));
    if (tester.state<MyAppState>(find.byType(MyApp)).isLoading) {
      started = true;
      break;
    }
  }
  print('Generation started: $started');

  // 2. Wait for loading to finish
  bool generationCompleted = false;
  String lastText = "";
  final stopwatch = Stopwatch()..start();
  
  try {
    while (stopwatch.elapsed < const Duration(seconds: 60)) {
      await tester.pump(const Duration(milliseconds: 500));
      
      final MyAppState state = tester.state<MyAppState>(find.byType(MyApp));
      if (!state.isLoading) {
        generationCompleted = true;
        lastText = state.response;
        break;
      }
    }
  } catch (e) {
    print('Error during generation: $e');
    rethrow;
  }

  print('RAW STATE RESPONSE: "$lastText"');

  // Strip <think> tags for comparison
  String cleanedText = lastText;
  if (cleanedText.contains('</think>')) {
    cleanedText = cleanedText.split('</think>').last.trim();
  }
  cleanedText = cleanedText.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '').trim();

  print('Final Response: "$lastText"');
  print('Cleaned Response: "$cleanedText"');
  expect(generationCompleted, true, reason: 'Generation did not complete within 60 seconds');
  expect(cleanedText.isNotEmpty, true, reason: 'Received an empty response after cleaning');
  
  expect(cleanedText.toLowerCase(), contains(expectedText.toLowerCase()), reason: 'Cleaned response did not contain "$expectedText"');
}
