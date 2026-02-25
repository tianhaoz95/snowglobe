import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobedemo/main.dart';
import 'package:snowglobedemo/src/rust/api/simple.dart';
import 'package:snowglobedemo/src/rust/frb_generated.dart';

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

    // 2. Log Detailed Backend Info
    final backendString = await checkBackend();
    print('------------------------------------------------------------');
    print('CHAT TEST - RUNTIME INFO');
    print('------------------------------------------------------------');
    print('Main Backend (Burn/App): $backendString');
    print('Framework Used: ${USE_EXECUTORCH ? "ExecuTorch (.pte)" : "Burn (.safetensors)"}');
    
    // Check for NPU/GPU backends in ExecuTorch if applicable
    if (USE_EXECUTORCH) {
      if (Platform.isMacOS || Platform.isIOS) {
        final useMps = Platform.environment['EXECUTORCH_USE_MPS'] ?? 'not set';
        print('ExecuTorch Backend Context (Apple): ${useMps == '1' ? 'GPU (MPS)' : 'CPU (XNNPACK)'} (env EXECUTORCH_USE_MPS=$useMps)');
      } else if (Platform.isAndroid) {
        print('ExecuTorch Backend Context (Android): Potential NPU (NNAPI) or CPU (XNNPACK)');
      }
    } else {
      print('ExecuTorch: Not active (using Burn backend directly)');
    }
    print('------------------------------------------------------------');

    // 3. Find the send button and tap it
    final sendButtonFinder = find.byIcon(Icons.send);
    expect(sendButtonFinder, findsOneWidget);

    await tester.tap(sendButtonFinder);
    
    // Pump to trigger the onPressed callback
    await tester.pump();

    // 4. Assert there is non-empty output within 30 seconds
    bool hasResponse = false;
    String lastText = "";
    final stopwatch = Stopwatch()..start();
    
    print('Generation phase started...');
    
    try {
      while (stopwatch.elapsed < const Duration(seconds: 30)) {
        await tester.pump(const Duration(milliseconds: 500));
        
        final responseTextFinder = find.descendant(
          of: find.byType(SingleChildScrollView),
          matching: find.byType(Text),
        );
        
        if (responseTextFinder.evaluate().isNotEmpty) {
          final textWidget = tester.widget<Text>(responseTextFinder.first);
          final currentText = textWidget.data ?? "";
          
          if (currentText.isNotEmpty && 
              currentText != "System ready. Let's chat!" &&
              currentText != "Type a prompt below to see the magic happen.") {
            
            if (currentText.length > lastText.length) {
              final newTokens = currentText.substring(lastText.length);
              print('Received tokens: "$newTokens"');
              lastText = currentText;
            }
            
            hasResponse = true;
            // We don't break immediately so we can see some streaming action in logs
            if (currentText.length > 50) break;
          }
        }
      }
    } catch (e, stack) {
      print('Error during generation: $e');
      print(stack);
      rethrow;
    }

    print('Generation finished. Final received text length: ${lastText.length}');
    expect(hasResponse, true, reason: 'Did not receive a non-empty response within 30 seconds');
    print('CHAT TEST - SUCCESS');
  });
}
