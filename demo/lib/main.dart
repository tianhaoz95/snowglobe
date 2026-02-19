import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:snowglobedemo/src/rust/api/simple.dart';
import 'package:snowglobedemo/src/rust/frb_generated.dart';
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late final TextEditingController _promptController;
  String _response = 'Type a prompt below to see the magic happen.';
  bool _isLoading = false;
  String? _sessionId;
  Future<void>? _initEngineFuture;

  // Performance metrics
  int _tokenCount = 0;
  double _elapsedSeconds = 0;
  double _tokensPerSecond = 0;

  @override
  void initState() {
    super.initState();
    _promptController = TextEditingController(
      text: 'What are three interesting facts about snow globes?',
    );
    _initEngineFuture = _initEngine();
  }

  Future<void> _initEngine() async {
    print('Initializing Rust library...');
    await RustLib.init();
    print('Rust library initialized');
    final backend = await checkBackend();
    print('Backend check: $backend');
    final cacheDir = await getApplicationSupportDirectory();
    if (!await cacheDir.exists()) {
      await cacheDir.create(recursive: true);
    }

    await _downloadModelAndTokenizer(cacheDir.path);

    final initResult = await initEngine(
      cacheDir: cacheDir.path,
      vocabShards: 8,
    );
    print('Engine initialized: $initResult');
    _sessionId = await initSession();
  }

  Future<void> _downloadModelAndTokenizer(String cacheDir) async {
    const modelUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors';
    const tokenizerUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json';

    final modelPath = '$cacheDir/model.safetensors';
    final tokenizerPath = '$cacheDir/tokenizer.json';

    setState(() {
      _response = 'Preparing model assets...';
      _isLoading = true;
    });

    try {
      if (!await File(tokenizerPath).exists()) {
        await _downloadFile(tokenizerUrl, tokenizerPath, 'Tokenizer');
      }
      if (!await File(modelPath).exists()) {
        await _downloadFile(modelUrl, modelPath, 'Model weights');
      }
    } catch (e) {
      print('Download error: $e');
    } finally {
      setState(() {
        _isLoading = false;
        _response = 'System ready. Let\'s chat!';
      });
    }
  }

  Future<void> _downloadFile(String url, String savePath, String label) async {
    final httpClient = HttpClient();
    try {
      final request = await httpClient.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        int downloaded = 0;
        final file = File(savePath);
        final sink = file.openWrite();

        await for (var chunk in response) {
          downloaded += chunk.length;
          sink.add(chunk);
          if (contentLength > 0) {
            final progress = (downloaded / contentLength * 100).toStringAsFixed(
              1,
            );
            setState(() {
              _response = 'Fetching $label... ($progress%)';
            });
          }
        }
        await sink.close();
      }
    } finally {
      httpClient.close();
    }
  }

  Future<void> _generateResponse() async {
    if (_sessionId == null || _isLoading) return;

    setState(() {
      _isLoading = true;
      _response = '';
      _tokenCount = 0;
      _elapsedSeconds = 0;
      _tokensPerSecond = 0;
    });

    try {
      final String currentPrompt = _promptController.text;
      final stopwatch = Stopwatch()..start();

      final tokenStream = generateResponse(
        sessionId: _sessionId!,
        prompt: currentPrompt,
      );

      await for (final token in tokenStream) {
        setState(() {
          _response += token;
          _tokenCount++;
          _elapsedSeconds = stopwatch.elapsed.inMilliseconds / 1000.0;
          if (_elapsedSeconds > 0) {
            _tokensPerSecond = _tokenCount / _elapsedSeconds;
          }
        });
      }
      stopwatch.stop();
    } catch (e) {
      setState(() {
        _response = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.interTextTheme(),
      ),
      home: Scaffold(
        backgroundColor: const Color(0xFFF6F7FB),
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
          title: Text(
            'SNOWGLOBE',
            style: GoogleFonts.oswald(
              letterSpacing: 3,
              fontWeight: FontWeight.bold,
              color: colorScheme.primary,
            ),
          ),
          actions: [
            IconButton(
              onPressed: () => _promptController.clear(),
              icon: const Icon(Icons.refresh),
              tooltip: 'Reset Conversation',
            ),
          ],
        ),
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 20.0,
              vertical: 8.0,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Engine Status Indicator
                FutureBuilder<void>(
                  future: _initEngineFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 12.0),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: LinearProgressIndicator(
                            minHeight: 6,
                            backgroundColor: colorScheme.surfaceVariant,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              colorScheme.secondary,
                            ),
                          ),
                        ),
                      );
                    }
                    return const SizedBox.shrink();
                  },
                ),

                // Response Card
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24.0),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.04),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24.0),
                      child: Column(
                        children: [
                          Expanded(
                            child: SingleChildScrollView(
                              padding: const EdgeInsets.all(24.0),
                              child: Text(
                                _response,
                                style: GoogleFonts.robotoMono(
                                  fontSize: 16,
                                  height: 1.6,
                                  color: Colors.black87,
                                ),
                              ),
                            ),
                          ),
                          if (_tokenCount > 0 || _isLoading)
                            _buildPerformanceMetrics(colorScheme),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Prompt Section
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.03),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: TextField(
                    controller: _promptController,
                    style: const TextStyle(fontSize: 15),
                    maxLines: 4,
                    minLines: 1,
                    decoration: InputDecoration(
                      hintText: 'Message Snowglobe...',
                      hintStyle: TextStyle(color: Colors.grey[400]),
                      contentPadding: const EdgeInsets.all(20),
                      border: InputBorder.none,
                      suffixIcon: Padding(
                        padding: const EdgeInsets.only(right: 8.0),
                        child: IconButton(
                          onPressed: _isLoading ? null : _generateResponse,
                          icon: Icon(
                            _isLoading ? Icons.hourglass_bottom : Icons.send,
                            color: _isLoading
                                ? colorScheme.secondary
                                : colorScheme.primary,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Footer
                Text(
                  'Powered by Qwen 2.5 & Rust Core',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[500],
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPerformanceMetrics(ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
      decoration: BoxDecoration(
        color: colorScheme.surfaceVariant.withOpacity(0.3),
        border: Border(
          top: BorderSide(color: colorScheme.outlineVariant.withOpacity(0.5)),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildMetricItem(
            Icons.speed,
            '${_tokensPerSecond.toStringAsFixed(1)} tok/s',
            colorScheme.primary,
          ),
          _buildMetricItem(
            Icons.numbers,
            '$_tokenCount tokens',
            colorScheme.secondary,
          ),
          _buildMetricItem(
            Icons.timer,
            '${_elapsedSeconds.toStringAsFixed(1)}s',
            colorScheme.tertiary,
          ),
        ],
      ),
    );
  }

  Widget _buildMetricItem(IconData icon, String value, Color color) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }
}
