import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobedemo/main.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Send default prompt and receive response', (WidgetTester tester) async {
    // Start the app
    await tester.pumpWidget(const MyApp());

    // 1. Wait for engine to be ready
    bool engineReady = false;
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

    expect(engineReady, true, reason: 'Engine did not become ready within 120 seconds');

    // Toggle speculative decoding on
    final switchFinder = find.byType(Switch);
    if (switchFinder.evaluate().isNotEmpty) {
      print('Toggling Speculative Decoding ON...');
      await tester.tap(switchFinder.first);
      await tester.pumpAndSettle();
      // Wait for re-initialization
      for (int i = 0; i < 60; i++) {
        await tester.pump(const Duration(seconds: 1));
        if (find.text("System ready. Let's chat!").evaluate().isNotEmpty) {
          break;
        }
      }
    }

    // 2. Log Detailed Backend Info
    final backendString = await SnowglobeOpenAI.checkBackend();
    final modelInfo = await SnowglobeOpenAI.getModelInfo();
    
    print('------------------------------------------------------------');
    print('CHAT TEST - RUNTIME INFO');
    print('------------------------------------------------------------');
    print('Hardware Backend (reported by runner): $backendString');
    print('Runner Framework: ${modelInfo?.runner ?? "Unknown"}');
    
    // Log additional details if available
    if (modelInfo != null) {
      print('Model Params: ${(modelInfo.paramCount.toDouble() / 1e6).toStringAsFixed(1)}M');
      print('Model Size: ${(modelInfo.modelSizeBytes.toDouble() / (1024 * 1024)).toStringAsFixed(1)} MB');
    }
    
    print('------------------------------------------------------------');

    // 3. Find the send button and tap it
    final sendButtonFinder = find.byIcon(Icons.send);
    expect(sendButtonFinder, findsOneWidget);

    await tester.tap(sendButtonFinder);
    
    // Pump to trigger the onPressed callback
    await tester.pump();

    // 4. Assert there is non-empty output within 60 seconds
    bool hasResponse = false;
    String lastText = "";
    final stopwatch = Stopwatch()..start();
    
    print('Generation phase started...');
    
    try {
      while (stopwatch.elapsed < const Duration(seconds: 60)) {
        await tester.pump(const Duration(milliseconds: 500));
        
        final markdownFinder = find.byType(MarkdownBody);
        
        if (markdownFinder.evaluate().isNotEmpty) {
          final markdownWidget = tester.widget<MarkdownBody>(markdownFinder.first);
          final currentText = markdownWidget.data;
          
          if (currentText.isNotEmpty && 
              currentText != "System ready. Let's chat!" &&
              currentText != "Type a prompt below to see the magic happen.") {
            
            if (currentText.length > lastText.length) {
              final newTokens = currentText.substring(lastText.length);
              print('Received tokens: "$newTokens"');
              lastText = currentText;
            }
            
            hasResponse = true;
            // Break if we found what we're looking for
            if (currentText.toLowerCase().contains('beijing')) break;
          }
        }
      }
    } catch (e, stack) {
      print('Error during generation: $e');
      print(stack);
      rethrow;
    }

    // Capture metrics from the UI
    final prefillFinder = find.byIcon(Icons.bolt);
    final speedFinder = find.textContaining('tok/s');
    
    String prefillLog = "Unknown";
    String speedLog = "Unknown";
    
    if (prefillFinder.evaluate().isNotEmpty) {
      // Find the text widget that is a sibling or child in the metric item
      // In our implementation, _buildMetricItem puts them in a Row
      final Row metricRow = tester.widget<Row>(find.ancestor(of: prefillFinder, matching: find.byType(Row)).first);
      final Text textWidget = metricRow.children.last as Text;
      prefillLog = textWidget.data ?? "Unknown";
    }
    if (speedFinder.evaluate().isNotEmpty) {
      speedLog = tester.widget<Text>(speedFinder.first).data ?? "Unknown";
    }

    print('------------------------------------------------------------');
    print('PERFORMANCE METRICS:');
    print('Prefill: $prefillLog');
    print('Generation Speed: $speedLog');
    print('Total Text Length: ${lastText.length}');
    print('------------------------------------------------------------');

    expect(hasResponse, true, reason: 'Did not receive a non-empty response within 30 seconds');
    expect(lastText.toLowerCase(), contains('beijing'), reason: 'Response did not contain "beijing"');
    print('CHAT TEST - SUCCESS');
  });
}
