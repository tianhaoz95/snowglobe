import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';
import 'package:snowglobedemo/src/rust/api/simple.dart';
import 'package:snowglobedemo/src/rust/frb_generated.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('initialize engine and exit', (WidgetTester tester) async {
    try {
      await RustLib.init();
      final cacheDir = await getApplicationSupportDirectory();
      await initEngine(
        cacheDir: cacheDir.path,
        config: const InitConfig(
          vocabShards: 8,
          maxGenLen: 128,
          backend: BackendType.liteRt,
          speculateTokens: 0,
        ),
      );
      print('Engine initialized successfully.');
    } catch (e, s) {
      print('Error during engine initialization: $e');
      print(s);
      rethrow;
    }
  });
}
