import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobedemo/main.dart' as app;
import 'package:path_provider/path_provider.dart';
import 'dart:io';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Backend and Model Switching Test', () {
    testWidgets('Verify dynamic scanning and switching', (tester) async {
      app.main();
      await tester.pumpAndSettle();

      // 0. Expand Settings if it's an ExpansionTile
      final settingsHeader = find.text('Engine Settings');
      if (settingsHeader.evaluate().isNotEmpty) {
        await tester.ensureVisible(settingsHeader);
        await tester.tap(settingsHeader);
        await tester.pumpAndSettle();
      }

      final appSupportDir = await getApplicationSupportDirectory();
      
      // 1. Prepare model directories
      final burnDir = Directory('${appSupportDir.path}/models/burn/qwen3_test');
      final etDir = Directory('${appSupportDir.path}/models/executorch/qwen3_et_test');
      final llamaDir = Directory('${appSupportDir.path}/models/llamaCpp/qwen3_gguf_test');

      await burnDir.create(recursive: true);
      await etDir.create(recursive: true);
      await llamaDir.create(recursive: true);

      // Create dummy marker files to satisfy scanner
      const dummyConfig = '{"model_type": "qwen2", "num_hidden_layers": 1, "hidden_size": 16, "num_attention_heads": 1, "intermediate_size": 32, "vocab_size": 100}';
      const dummyTokenizer = '{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null, "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": null, "byte_fallback": null, "vocab": {}, "merges": []}}';

      await File('${burnDir.path}/model.safetensors').writeAsString('dummy');
      await File('${burnDir.path}/config.json').writeAsString(dummyConfig);
      await File('${burnDir.path}/tokenizer.json').writeAsString(dummyTokenizer);

      await File('${etDir.path}/model.pte').writeAsString('dummy');
      await File('${etDir.path}/config.json').writeAsString(dummyConfig);
      await File('${etDir.path}/tokenizer.json').writeAsString(dummyTokenizer);

      await File('${llamaDir.path}/model.gguf').writeAsString('dummy');
      await File('${llamaDir.path}/config.json').writeAsString(dummyConfig);
      await File('${llamaDir.path}/tokenizer.json').writeAsString(dummyTokenizer);

      // 2. Switch to Burn and verify model list
      print('Testing Burn backend...');
      await _selectBackend(tester, 'Burn');
      
      final burnState = tester.state<app.MyAppState>(find.byType(app.MyApp));
      expect(burnState.availableModels.any((m) => m.name == 'qwen3_test'), isTrue);
      expect(burnState.availableModels.any((m) => m.name == 'qwen3_et_test'), isFalse);

      // 3. Switch to ExecuTorch and verify model list
      print('Testing ExecuTorch backend...');
      await _selectBackend(tester, 'Executorch');
      
      final etState = tester.state<app.MyAppState>(find.byType(app.MyApp));
      expect(etState.availableModels.any((m) => m.name == 'qwen3_et_test'), isTrue);
      expect(etState.availableModels.any((m) => m.name == 'qwen3_test'), isFalse);

      // 4. Switch to LlamaCpp and verify model list
      print('Testing LlamaCpp backend...');
      await _selectBackend(tester, 'LlamaCpp');
      
      final llamaState = tester.state<app.MyAppState>(find.byType(app.MyApp));
      expect(llamaState.availableModels.any((m) => m.name == 'qwen3_gguf_test'), isTrue);
      expect(llamaState.availableModels.any((m) => m.name == 'qwen3_et_test'), isFalse);
    });
  });
}

Future<void> _selectBackend(WidgetTester tester, String backendName) async {
  final state = tester.state<app.MyAppState>(find.byType(app.MyApp));
  
  app.InferenceBackend? backend;
  if (backendName == 'Burn') backend = app.InferenceBackend.burn;
  if (backendName == 'Executorch') backend = app.InferenceBackend.executorch;
  if (backendName == 'LlamaCpp') backend = app.InferenceBackend.llamaCpp;
  
  state.onBackendChanged(backend);
  
  // Give it time to scan and re-init
  await Future.delayed(const Duration(seconds: 3));
  await tester.pumpAndSettle();
}
