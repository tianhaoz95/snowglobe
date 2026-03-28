import 'dart:convert';
import 'package:openai_dart/openai_dart.dart';
import 'package:snowglobe_openai/src/rust/api.dart' as rust_api;
import 'package:snowglobe_openai/src/rust/frb_generated.dart';

export 'package:snowglobe_openai/src/rust/api.dart' show InitConfig, BackendType, HardwareTarget, ModelInfo;

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

  static Future<rust_api.ModelInfo?> getModelInfo() async {
    return rust_api.getModelInfo();
  }

  static Future<String> checkBackend() async {
    return rust_api.checkBackend();
  }

  static Future<String> initSession() async {
    return rust_api.initSession();
  }

  static Future<int> getLastAcceptedCount({required String sessionId}) async {
    return rust_api.getLastAcceptedCount(sessionId: sessionId);
  }

  static Stream<String> generateResponse({
    required String sessionId,
    required String prompt,
    required int maxGenLen,
  }) {
    return rust_api.generateResponse(
      sessionId: sessionId,
      prompt: prompt,
      maxGenLen: maxGenLen,
    );
  }
}
