import 'dart:convert';
import 'package:openai_dart/openai_dart.dart';
import 'package:snowglobe_openai/src/rust/api.dart' as rust_api;
import 'package:snowglobe_openai/src/rust/frb_generated.dart';

export 'package:snowglobe_openai/src/rust/api.dart' show InitConfig, BackendType;

class SnowglobeOpenAI {
  static Future<void> initRust() async {
    await RustLib.init();
  }

  static Future<String> initEngine({
    required String cacheDir,
    required rust_api.InitConfig config,
  }) async {
    return rust_api.initEngine(cacheDir: cacheDir, config: config);
  }

  static Future<CreateChatCompletionResponse> createChatCompletion(
    CreateChatCompletionRequest request,
  ) async {
    final requestJson = jsonEncode(request.toJson());
    final responseJson = await rust_api.chatCompletion(requestJson: requestJson);
    return CreateChatCompletionResponse.fromJson(jsonDecode(responseJson));
  }

  static Stream<CreateChatCompletionStreamResponse> createChatCompletionStream(
    CreateChatCompletionRequest request,
  ) async* {
    final requestJson = jsonEncode(request.toJson());
    final stream = rust_api.chatCompletionStream(requestJson: requestJson);
    await for (final chunkJson in stream) {
      yield CreateChatCompletionStreamResponse.fromJson(jsonDecode(chunkJson));
    }
  }
}
