import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:snowglobe/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('end-to-end test', () {
    testWidgets('tap on the floating action button, verify counter',
        (tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Verify that the initial state is empty or has no messages (depending on implementation, but based on reading code it starts empty)
      expect(find.text('Hi!'), findsNothing);
      expect(find.text('Hello AI'), findsNothing);

      // Find the TextField
      final Finder textField = find.byType(TextField);
      expect(textField, findsOneWidget);

      // Enter text
      await tester.enterText(textField, 'Hello AI');
      await tester.pumpAndSettle();

      // Find and tap the Send button
      final Finder sendButton = find.byIcon(Icons.send);
      expect(sendButton, findsOneWidget);
      await tester.tap(sendButton);

      // Trigger a frame.
      await tester.pumpAndSettle();

      // Verify that the messages are displayed
      expect(find.text('Hello AI'), findsOneWidget);
      expect(find.text('Hi!'), findsOneWidget);
    });
  });
}
