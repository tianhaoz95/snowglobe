import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:file_picker/file_picker.dart';
import 'package:snowglobedemo/src/rust/api/simple.dart';
import 'package:snowglobedemo/src/rust/frb_generated.dart';
import 'dart:io';

const bool USE_QWEN3 = true;
const bool USE_EXECUTORCH = true;

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

  // Engine status
  String? _engineErrorMessage;
  bool _isEngineReady = false;
  bool _isRustLibInitialized = false;
  int _maxGenLen = 128;

  // Performance metrics
  int _tokenCount = 0;
  double _elapsedSeconds = 0;
  double _tokensPerSecond = 0;
  double? _prefillTimeSeconds;

  @override
  void initState() {
    super.initState();
    _promptController = TextEditingController(
      text: 'what is the capital of China?',
    );
    _initEngineFuture = _initEngine();
  }

  Future<void> _initEngine() async {
    setState(() {
      _isLoading = true;
      _engineErrorMessage = null;
      _isEngineReady = false;
    });

    try {
      if (!_isRustLibInitialized) {
        print('Initializing Rust library...');
        await RustLib.init();
        _isRustLibInitialized = true;
        print('Rust library initialized');
      }

      final backend = await checkBackend();
      print('Backend check: $backend');

      final cacheDir = await getApplicationSupportDirectory();
      if (!await cacheDir.exists()) {
        await cacheDir.create(recursive: true);
      }

      print('Application Support Directory: ${cacheDir.path}');

      final downloadSuccess = await _downloadModelAndTokenizer(cacheDir.path);
      if (!downloadSuccess) {
        setState(() {
          _engineErrorMessage =
              'Model files not found. Please upload a model file or download it.';
        });
        return;
      }

      final initResult = await initEngine(
        cacheDir: cacheDir.path,
        config: InitConfig(
          vocabShards: 8,
          maxGenLen: _maxGenLen,
          useExecutorch: USE_EXECUTORCH,
        ),
      );
      print('Engine initialized: $initResult');
      _sessionId = await initSession();
      setState(() {
        _isEngineReady = true;
        _response = 'System ready. Let\'s chat!';
      });
    } catch (e) {
      print('Initialization error: $e');
      setState(() {
        _engineErrorMessage = 'Failed to initialize engine: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _deleteModelAssets() async {
    final cacheDir = await getApplicationSupportDirectory();
    final files = [
      File('${cacheDir.path}/model.safetensors'),
      File('${cacheDir.path}/model.pte'),
      File('${cacheDir.path}/tokenizer.json'),
      File('${cacheDir.path}/config.json'),
    ];

    setState(() {
      _isLoading = true;
      _response = 'Deleting model assets...';
    });

    try {
      for (var file in files) {
        if (await file.exists()) {
          await file.delete();
        }
      }
      _sessionId = null;
      _isEngineReady = false;
      setState(() {
        _response = 'Model assets deleted. System not ready.';
        _engineErrorMessage = 'Model files deleted. Please redownload or upload.';
      });
    } catch (e) {
      setState(() {
        _response = 'Error deleting assets: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _redownloadModelAssets() async {
    setState(() {
      _initEngineFuture = _initEngine();
    });
  }

  Future<void> _pickAndInstallModel() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowMultiple: false,
    );

    if (result == null || result.files.single.path == null) {
      return;
    }

    setState(() {
      _isLoading = true;
      _response = 'Installing picked model...';
      _engineErrorMessage = null;
    });

    try {
      final cacheDir = await getApplicationSupportDirectory();
      final pickedFile = File(result.files.single.path!);
      final extension = pickedFile.path.split('.').last.toLowerCase();

      String targetName;
      if (extension == 'pte') {
        targetName = 'model.pte';
      } else if (extension == 'safetensors') {
        targetName = 'model.safetensors';
      } else {
        targetName = USE_EXECUTORCH ? 'model.pte' : 'model.safetensors';
      }

      final targetPath = '${cacheDir.path}/$targetName';
      await pickedFile.copy(targetPath);

      setState(() {
        _response = 'Model installed. Re-initializing engine...';
      });

      final initResult = await initEngine(
        cacheDir: cacheDir.path,
        config: InitConfig(
          vocabShards: 8,
          maxGenLen: _maxGenLen,
          useExecutorch: USE_EXECUTORCH,
        ),
      );
      _sessionId = await initSession();

      setState(() {
        _isEngineReady = true;
        _response = 'Engine ready with manual model: $targetName';
      });
    } catch (e) {
      setState(() {
        _response = 'Error installing model: $e';
        _engineErrorMessage = 'Failed to load picked model: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<bool> _downloadModelAndTokenizer(String cacheDir) async {
    const qwen2_5ModelUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors';
    const qwen2_5TokenizerUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json';
    const qwen2_5ConfigUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json';

    const qwen3ModelUrl =
        'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors';
    const qwen3TokenizerUrl =
        'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json';
    const qwen3ConfigUrl =
        'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json';

    final modelUrl = USE_QWEN3 ? qwen3ModelUrl : qwen2_5ModelUrl;
    final tokenizerUrl = USE_QWEN3 ? qwen3TokenizerUrl : qwen2_5TokenizerUrl;
    final configUrl = USE_QWEN3 ? qwen3ConfigUrl : qwen2_5ConfigUrl;

    final modelPath = USE_EXECUTORCH
        ? '$cacheDir/model.pte'
        : '$cacheDir/model.safetensors';
    final tokenizerPath = '$cacheDir/tokenizer.json';
    final configPath = '$cacheDir/config.json';

    try {
      if (!await File(configPath).exists()) {
        await _downloadFile(configUrl, configPath, 'Config');
      }
      if (!await File(tokenizerPath).exists()) {
        await _downloadFile(tokenizerUrl, tokenizerPath, 'Tokenizer');
      }
      if (!await File(modelPath).exists()) {
        if (USE_EXECUTORCH) {
          final localPte = File('../qwen3_0.6b.pte');
          final androidPte = File('/data/local/tmp/snowglobe/model.pte');
          final androidExternalPte = File('/sdcard/Android/data/com.example.snowglobedemo/cache/model.pte');
          
          if (await localPte.exists()) {
            setState(() => _response = 'Copying local model.pte...');
            await localPte.copy(modelPath);
          } else if (Platform.isAndroid && await androidPte.exists()) {
            setState(() => _response = 'Copying model.pte from /data/local/tmp...');
            await androidPte.copy(modelPath);
          } else if (Platform.isAndroid && await androidExternalPte.exists()) {
            setState(() => _response = 'Copying model.pte from external storage...');
            await androidExternalPte.copy(modelPath);
          } else {
            return false;
          }
        } else {
          await _downloadFile(modelUrl, modelPath, 'Model weights');
        }
      }
      return true;
    } catch (e) {
      print('Download error: $e');
      return false;
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
      } else {
        throw Exception('Failed to download $label: HTTP ${response.statusCode}');
      }
    } finally {
      httpClient.close();
    }
  }

  Future<void> _generateResponse() async {
    if (_sessionId == null || _isLoading || !_isEngineReady) return;

    setState(() {
      _isLoading = true;
      _response = '';
      _tokenCount = 0;
      _elapsedSeconds = 0;
      _tokensPerSecond = 0;
      _prefillTimeSeconds = null;
    });

    try {
      final String currentPrompt = _promptController.text;
      final stopwatch = Stopwatch()..start();

      final tokenStream = generateResponse(
        sessionId: _sessionId!,
        prompt: currentPrompt,
        maxGenLen: _maxGenLen,
      );

      await for (final token in tokenStream) {
        if (!mounted) break;

        if (_prefillTimeSeconds == null && token.isNotEmpty) {
          _prefillTimeSeconds = stopwatch.elapsed.inMilliseconds / 1000.0;
        }

        setState(() {
          _response += token;
          _tokenCount++;
          _elapsedSeconds = stopwatch.elapsed.inMilliseconds / 1000.0;

          // Generation speed calculation (tok/s after prefill)
          if (_prefillTimeSeconds != null) {
            final generationSeconds = _elapsedSeconds - _prefillTimeSeconds!;
            if (generationSeconds > 0) {
              _tokensPerSecond = _tokenCount / generationSeconds;
            } else {
              _tokensPerSecond = 0;
            }
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
              onPressed: _isLoading ? null : _deleteModelAssets,
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Delete Model',
            ),
            IconButton(
              onPressed: _isLoading ? null : _pickAndInstallModel,
              icon: const Icon(Icons.file_upload_outlined),
              tooltip: 'Pick Model File',
            ),
            IconButton(
              onPressed: _isLoading ? null : _redownloadModelAssets,
              icon: const Icon(Icons.download),
              tooltip: 'Redownload Model',
            ),
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
                if (_engineErrorMessage != null)
                  _buildErrorBanner(colorScheme),

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

                const SizedBox(height: 16),

                // Generation Settings
                if (_isEngineReady)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Max Tokens: $_maxGenLen',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: colorScheme.primary,
                              ),
                            ),
                            Text(
                              'Controls response length',
                              style: TextStyle(
                                fontSize: 11,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            trackHeight: 4,
                            thumbShape: const RoundSliderThumbShape(
                              enabledThumbRadius: 8,
                            ),
                            overlayShape: const RoundSliderOverlayShape(
                              overlayRadius: 16,
                            ),
                          ),
                          child: Slider(
                            value: _maxGenLen.toDouble(),
                            min: 16,
                            max: 512,
                            divisions: 31,
                            label: _maxGenLen.toString(),
                            onChanged: _isLoading
                                ? null
                                : (value) {
                                    setState(() {
                                      _maxGenLen = value.round();
                                    });
                                  },
                            onChangeEnd: _isLoading
                                ? null
                                : (value) {
                                    _initEngine();
                                  },
                          ),
                        ),
                      ],
                    ),
                  ),

                const SizedBox(height: 16),

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
                    enabled: _isEngineReady && !_isLoading,
                    style: const TextStyle(fontSize: 15),
                    maxLines: 4,
                    minLines: 1,
                    decoration: InputDecoration(
                      hintText: _isEngineReady
                          ? 'Message Snowglobe...'
                          : 'Engine not ready...',
                      hintStyle: TextStyle(color: Colors.grey[400]),
                      contentPadding: const EdgeInsets.all(20),
                      border: InputBorder.none,
                      suffixIcon: Padding(
                        padding: const EdgeInsets.only(right: 8.0),
                        child: IconButton(
                          onPressed: _isLoading || !_isEngineReady
                              ? null
                              : _generateResponse,
                          icon: Icon(
                            _isLoading ? Icons.hourglass_bottom : Icons.send,
                            color: _isLoading || !_isEngineReady
                                ? colorScheme.secondary.withOpacity(0.5)
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
                  'Powered by ${USE_QWEN3 ? "Qwen 3" : "Qwen 2.5"} & Rust Core',
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

  Widget _buildErrorBanner(ColorScheme colorScheme) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colorScheme.error.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Icon(Icons.warning_amber_rounded, color: colorScheme.error),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _engineErrorMessage!,
              style: TextStyle(
                color: colorScheme.onErrorContainer,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const SizedBox(width: 8),
          TextButton.icon(
            onPressed: _isLoading ? null : _initEngine,
            icon: const Icon(Icons.refresh, size: 18),
            label: const Text('Retry'),
            style: TextButton.styleFrom(
              foregroundColor: colorScheme.error,
              padding: const EdgeInsets.symmetric(horizontal: 12),
            ),
          ),
        ],
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
            Icons.bolt,
            _prefillTimeSeconds != null
                ? '${_prefillTimeSeconds!.toStringAsFixed(2)}s'
                : 'Prefilling...',
            colorScheme.onSurfaceVariant,
          ),
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
