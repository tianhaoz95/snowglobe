import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:file_picker/file_picker.dart';
import 'package:snowglobe_openai/snowglobe_openai.dart';
import 'dart:io';

enum ModelType { qwen2_5, qwen3, qwen3_5, qwen2_5_pte }

enum InferenceBackend { burn, executorch, llamaCpp }

class AvailableModel {
  final String name;
  final String path;
  final ModelType? type; // Can be null if auto-detected

  AvailableModel({required this.name, required this.path, this.type});
}

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  MyAppState createState() => MyAppState();
}

class MyAppState extends State<MyApp> {
  late final TextEditingController _promptController;
  String _response = 'Type a prompt below to see the magic happen.';
  bool _isLoading = false;
  String? _sessionId;
  Future<void>? _initEngineFuture;

  // Engine status
  InferenceBackend _selectedBackend = InferenceBackend.llamaCpp;
  List<AvailableModel> availableModels = [];
  AvailableModel? _selectedModel;
  
  HardwareTarget _selectedHardware = HardwareTarget.auto;
  String? _engineErrorMessage;
  bool _isEngineReady = false;
  bool _isRustLibInitialized = false;
  int _maxGenLen = 128;
  ModelInfo? _modelInfo;

  // Speculative decoding
  bool _useSpeculative = false;
  int _speculateTokens = 4;
  double _avgAcceptedTokens = 0;

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
    // Initialize Rust library first, then refresh models and init engine
    _initEngine();
  }

  Future<void> _refreshAvailableModels() async {
    final appSupportDir = await getApplicationSupportDirectory();
    final modelsBaseDir = Directory('${appSupportDir.path}/models/${_selectedBackend.name}');
    
    if (!await modelsBaseDir.exists()) {
      await modelsBaseDir.create(recursive: true);
    }

    final List<AvailableModel> detected = [];
    final entities = await modelsBaseDir.list().toList();
    
    for (var entity in entities) {
      if (entity is Directory) {
        final name = entity.path.split(Platform.pathSeparator).last;
        // Check if it contains required files for the backend
        bool isValid = false;
        if (_selectedBackend == InferenceBackend.burn) {
          isValid = await File('${entity.path}/model.safetensors').exists();
        } else if (_selectedBackend == InferenceBackend.executorch) {
          isValid = await File('${entity.path}/model.pte').exists();
        } else if (_selectedBackend == InferenceBackend.llamaCpp) {
          isValid = await File('${entity.path}/model.gguf').exists();
        }

        if (isValid) {
          ModelType? type;
          if (name.toLowerCase().contains('qwen3_5')) {
            type = ModelType.qwen3_5;
          } else if (name.toLowerCase().contains('qwen3')) {
            type = ModelType.qwen3;
          } else if (name.toLowerCase().contains('qwen2_5')) {
            type = ModelType.qwen2_5;
          }
          detected.add(AvailableModel(name: name, path: entity.path, type: type));
        }
      }
    }

    // Also check legacy path for backward compatibility if nothing found
    if (detected.isEmpty) {
       // Legacy fallback logic could go here, but for now we enforce the new structure
    }

    setState(() {
      availableModels = detected;
      if (availableModels.isNotEmpty) {
        // Try to keep same model name if switching backend, or just take first
        final previousName = _selectedModel?.name;
        _selectedModel = availableModels.firstWhere(
          (m) => m.name == previousName, 
          orElse: () => availableModels.first
        );
      } else {
        _selectedModel = null;
      }
    });
  }

  Future<void> _initEngine() async {
    if (!mounted) return;
    
    await _refreshAvailableModels();

    setState(() {
      _isLoading = true;
      _engineErrorMessage = null;
      _isEngineReady = false;
      _modelInfo = null;
    });

    try {
      if (!_isRustLibInitialized) {
        print('Initializing Rust library...');
        await SnowglobeOpenAI.initRust();
        _isRustLibInitialized = true;
        print('Rust library initialized');
      }

      final backend = await SnowglobeOpenAI.checkBackend();
      print('Backend check: $backend');

      if (_selectedModel == null) {
        // Try to download default if none available
        final appSupportDir = await getApplicationSupportDirectory();
        final defaultName = _selectedBackend == InferenceBackend.llamaCpp ? 'qwen3_5' : 'qwen3';
        final cacheDir = Directory('${appSupportDir.path}/models/${_selectedBackend.name}/$defaultName');
        if (!await cacheDir.exists()) {
          await cacheDir.create(recursive: true);
        }
        
        print('Downloading default model to ${cacheDir.path}');
        final downloadSuccess = await _downloadModelAndTokenizer(cacheDir.path);
        if (!downloadSuccess) {
          throw Exception('Model files not found and download failed.');
        }
        await _refreshAvailableModels();
      }

      if (_selectedModel == null) {
        setState(() {
          _engineErrorMessage = 'No models available for ${_selectedBackend.name}';
          _isLoading = false;
        });
        return;
      }

      print('Initializing engine with model at: ${_selectedModel!.path}');

      final initResult = await SnowglobeOpenAI.initEngine(
        cacheDir: _selectedModel!.path,
        config: InitConfig(
          vocabShards: 8,
          maxGenLen: _maxGenLen,
          useExecutorch: _selectedBackend == InferenceBackend.executorch,
          backend: _selectedBackend == InferenceBackend.llamaCpp
              ? BackendType.llamaCpp
              : (_selectedBackend == InferenceBackend.executorch ? BackendType.execuTorch : BackendType.burn),
          hardware: _selectedHardware,
          speculateTokens: _useSpeculative ? _speculateTokens : 0,
        ),
      );
      print('Engine initialized: $initResult');

      if (initResult != 'Success') {
        throw Exception(initResult);
      }

      _sessionId = await SnowglobeOpenAI.initSession();
      if (_sessionId!.startsWith('Error:')) {
        throw Exception(_sessionId);
      }
      
      final modelInfo = await SnowglobeOpenAI.getModelInfo();

      if (mounted) {
        setState(() {
          _isEngineReady = true;
          _response = 'System ready. Let\'s chat!';
          _modelInfo = modelInfo;
        });
      }
    } catch (e) {
      print('Initialization error: $e');
      if (mounted) {
        setState(() {
          _engineErrorMessage = 'Failed to initialize engine: $e';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void onBackendChanged(InferenceBackend? value) {
    if (value != null && value != _selectedBackend) {
      setState(() {
        _selectedBackend = value;
      });
      _initEngine();
    }
  }

  Future<void> _deleteModelAssets() async {
    if (_selectedModel == null) return;
    
    final cacheDir = Directory(_selectedModel!.path);
    
    final files = [
      File('${cacheDir.path}/model.safetensors'),
      File('${cacheDir.path}/model.pte'),
      File('${cacheDir.path}/model.gguf'),
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
      
      // Also try to delete the directory if empty
      if (await cacheDir.exists()) {
        try {
          await cacheDir.delete();
        } catch (_) {}
      }

      _sessionId = null;
      _isEngineReady = false;
      _modelInfo = null;
      setState(() {
        _response = 'Model assets deleted. System not ready.';
        _engineErrorMessage =
            'Model files deleted. Please redownload or upload.';
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
      final appSupportDir = await getApplicationSupportDirectory();
      // Default to picking name from the file itself or use a generic 'manual'
      final pickedFile = File(result.files.single.path!);
      final fileName = pickedFile.path.split(Platform.pathSeparator).last;
      final modelFolderName = fileName.split('.').first;
      
      final cacheDir = Directory('${appSupportDir.path}/models/${_selectedBackend.name}/$modelFolderName');
      if (!await cacheDir.exists()) {
        await cacheDir.create(recursive: true);
      }

      final extension = pickedFile.path.split('.').last.toLowerCase();

      String targetName;
      if (extension == 'pte') {
        targetName = 'model.pte';
      } else if (extension == 'gguf') {
        targetName = 'model.gguf';
      } else if (extension == 'safetensors') {
        targetName = 'model.safetensors';
      } else {
        targetName = _selectedBackend == InferenceBackend.llamaCpp
            ? 'model.gguf'
            : (_selectedBackend == InferenceBackend.executorch ? 'model.pte' : 'model.safetensors');
      }

      final targetPath = '${cacheDir.path}/$targetName';
      await pickedFile.copy(targetPath);

      setState(() {
        _response = 'Model installed. Re-initializing engine...';
      });

      await _refreshAvailableModels();
      
      await SnowglobeOpenAI.initEngine(
        cacheDir: cacheDir.path,
        config: InitConfig(
          vocabShards: 8,
          maxGenLen: _maxGenLen,
          useExecutorch: _selectedBackend == InferenceBackend.executorch,
          backend: _selectedBackend == InferenceBackend.llamaCpp
              ? BackendType.llamaCpp
              : (_selectedBackend == InferenceBackend.executorch ? BackendType.execuTorch : BackendType.burn),
          hardware: _selectedHardware,
          speculateTokens: _useSpeculative ? _speculateTokens : 0,
        ),
      );
      _sessionId = await SnowglobeOpenAI.initSession();
      final modelInfo = await SnowglobeOpenAI.getModelInfo();

      setState(() {
        _isEngineReady = true;
        _modelInfo = modelInfo;
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
    const qwen2_5BaseUrl = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main';
    const qwen3BaseUrl = 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main';
    const qwen3_5BaseUrl = 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main';
    
    final modelFolderName = cacheDir.split(Platform.pathSeparator).last.toLowerCase();
    String baseUrl = qwen3BaseUrl;
    if (modelFolderName.contains('qwen3_5')) {
      baseUrl = qwen3_5BaseUrl;
    } else if (modelFolderName.contains('qwen2_5')) {
      baseUrl = qwen2_5BaseUrl;
    }

    String modelUrl = '$baseUrl/model.safetensors';
    String tokenizerUrl = '$baseUrl/tokenizer.json';
    String configUrl = '$baseUrl/config.json';

    if (_selectedBackend == InferenceBackend.llamaCpp) {
      if (modelFolderName.contains('qwen3_5')) {
        modelUrl = 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf';
      } else {
        modelUrl = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf';
      }
    } else if (_selectedBackend == InferenceBackend.executorch) {
      // PTE models are usually pushed manually or via side-load, but we provide URLs if available
      modelUrl = ""; 
    }

    final modelPath = _selectedBackend == InferenceBackend.llamaCpp
        ? '$cacheDir/model.gguf'
        : (_selectedBackend == InferenceBackend.executorch
            ? '$cacheDir/model.pte'
            : '$cacheDir/model.safetensors');
    final tokenizerPath = '$cacheDir/tokenizer.json';
    final configPath = '$cacheDir/config.json';

    final localBase = '../.test_assets/$modelFolderName';
    final androidBase = '/data/local/tmp/snowglobe/$modelFolderName';

    Future<bool> trySideLoad(String fileName, String targetPath) async {
      final localFile = File('$localBase/$fileName');
      final androidFile = File('$androidBase/$fileName');

      if (await localFile.exists()) {
        print('Side-loading local $fileName to $targetPath');
        setState(() => _response = 'Copying local $fileName...');
        await localFile.copy(targetPath);
        return true;
      } else if (Platform.isAndroid && await androidFile.exists()) {
        print('Side-loading $fileName from /data/local/tmp to $targetPath');
        setState(() => _response = 'Copying $fileName from /data/local/tmp...');
        await androidFile.copy(targetPath);
        return true;
      }
      return false;
    }

    try {
      if (!await File(configPath).exists()) {
        if (!await trySideLoad('config.json', configPath)) {
          await _downloadFile(configUrl, configPath, 'Config');
        }
      }
      if (!await File(tokenizerPath).exists()) {
        if (!await trySideLoad('tokenizer.json', tokenizerPath)) {
          await _downloadFile(tokenizerUrl, tokenizerPath, 'Tokenizer');
        }
      }
      if (!await File(modelPath).exists()) {
        if (!await trySideLoad(modelPath.split(Platform.pathSeparator).last, modelPath)) {
          if (modelUrl.isNotEmpty) {
            await _downloadFile(modelUrl, modelPath, 'Model weights');
          } else {
            return false;
          }
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
    final tempPath = '$savePath.tmp';
    try {
      final request = await httpClient.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        int downloaded = 0;
        final file = File(tempPath);
        final sink = file.openWrite();

        try {
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
          // Rename temp file to final path upon success
          await File(tempPath).rename(savePath);
        } catch (e) {
          await sink.close();
          if (await File(tempPath).exists()) {
            await File(tempPath).delete();
          }
          rethrow;
        }
      } else {
        throw Exception('Failed to download $label: HTTP ${response.statusCode}');
      }
    } finally {
      httpClient.close();
      // Ensure cleanup of temp file if it still exists
      if (await File(tempPath).exists()) {
        try {
          await File(tempPath).delete();
        } catch (_) {}
      }
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
      _avgAcceptedTokens = 0;
    });

    try {
      final String currentPrompt = _promptController.text;
      final stopwatch = Stopwatch()..start();

      final tokenStream = SnowglobeOpenAI.generateResponse(
        sessionId: _sessionId!,
        prompt: currentPrompt,
        maxGenLen: _maxGenLen,
      );

      int totalAccepted = 0;
      int iterations = 0;

      await for (final token in tokenStream) {
        if (!mounted) break;

        if (_prefillTimeSeconds == null && token.isNotEmpty) {
          _prefillTimeSeconds = stopwatch.elapsed.inMilliseconds / 1000.0;
        }

        final accepted = await SnowglobeOpenAI.getLastAcceptedCount(sessionId: _sessionId!);
        if (accepted > 0) {
          totalAccepted += accepted;
          iterations++;
        }

        setState(() {
          _response += token;
          _tokenCount += (accepted > 0 ? accepted : 1);
          _elapsedSeconds = stopwatch.elapsed.inMilliseconds / 1000.0;
          if (iterations > 0) {
            _avgAcceptedTokens = totalAccepted / iterations;
          }

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
      if (mounted) {
        setState(() {
          _response = 'Error: $e';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
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
                if (_engineErrorMessage != null) _buildErrorBanner(colorScheme),

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
                            backgroundColor: colorScheme.surfaceContainerHighest,
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
                              child: MarkdownBody(
                                data: _response,
                                selectable: true,
                                styleSheet: MarkdownStyleSheet(
                                  p: GoogleFonts.roboto(
                                    fontSize: 16,
                                    height: 1.6,
                                    color: Colors.black87,
                                  ),
                                  code: GoogleFonts.robotoMono(
                                    backgroundColor: Colors.grey[100],
                                    fontSize: 14,
                                  ),
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
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (_isEngineReady && _modelInfo != null) ...[
                          SingleChildScrollView(
                            scrollDirection: Axis.horizontal,
                            child: Row(
                              children: [
                                _buildModelInfoChip(
                                  '${(_modelInfo!.paramCount.toDouble() / 1e6).toStringAsFixed(1)}M params',
                                  Icons.memory,
                                  colorScheme.primary,
                                ),
                                const SizedBox(width: 8),
                                _buildModelInfoChip(
                                  '${(_modelInfo!.modelSizeBytes.toDouble() / (1024 * 1024)).toStringAsFixed(1)}MB',
                                  Icons.storage,
                                  colorScheme.secondary,
                                ),
                                const SizedBox(width: 8),
                                _buildModelInfoChip(
                                  '${_modelInfo!.numLayers} layers',
                                  Icons.layers,
                                  colorScheme.tertiary,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              _buildModelInfoChip(
                                'Runner: ${_modelInfo!.runner}',
                                Icons.directions_run,
                                Colors.orange,
                              ),
                              const SizedBox(width: 8),
                              _buildModelInfoChip(
                                'Backend: ${_modelInfo!.backend}',
                                Icons.settings_input_component,
                                Colors.teal,
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                        ],
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Backend',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: colorScheme.primary,
                              ),
                            ),
                            DropdownButton<InferenceBackend>(
                              key: const Key('backend_dropdown'),
                              value: _selectedBackend,
                              underline: const SizedBox(),
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey[800],
                                fontWeight: FontWeight.w500,
                              ),
                              items: InferenceBackend.values.map((b) {
                                return DropdownMenuItem(
                                  value: b,
                                  child: Text(b.name[0].toUpperCase() + b.name.substring(1)),
                                );
                              }).toList(),
                              onChanged: _isLoading ? null : onBackendChanged,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Model',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: colorScheme.primary,
                              ),
                            ),
                            DropdownButton<AvailableModel>(
                              key: const Key('model_dropdown'),
                              value: _selectedModel,
                              underline: const SizedBox(),
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey[800],
                                fontWeight: FontWeight.w500,
                              ),
                              items: availableModels.map((m) {
                                return DropdownMenuItem(
                                  value: m,
                                  child: Text(m.name),
                                );
                              }).toList(),
                              onChanged: (_isLoading || availableModels.isEmpty)
                                  ? null
                                  : (value) {
                                      if (value != null) {
                                        setState(() {
                                          _selectedModel = value;
                                        });
                                        _initEngine();
                                      }
                                    },
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Hardware',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: colorScheme.primary,
                              ),
                            ),
                            DropdownButton<HardwareTarget>(
                              value: _selectedHardware,
                              underline: const SizedBox(),
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey[800],
                                fontWeight: FontWeight.w500,
                              ),
                              items: const [
                                DropdownMenuItem(
                                  value: HardwareTarget.auto,
                                  child: Text('Auto (Max Accel)'),
                                ),
                                DropdownMenuItem(
                                  value: HardwareTarget.cpu,
                                  child: Text('CPU Only'),
                                ),
                                DropdownMenuItem(
                                  value: HardwareTarget.gpu,
                                  child: Text('GPU (Vulkan/Metal)'),
                                ),
                                DropdownMenuItem(
                                  value: HardwareTarget.npu,
                                  child: Text('NPU (QNN/Hexagon)'),
                                ),
                              ],
                              onChanged: _isLoading
                                  ? null
                                  : (value) {
                                      if (value != null) {
                                        setState(() {
                                          _selectedHardware = value;
                                        });
                                        _initEngine();
                                      }
                                    },
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Speculative Decoding',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: colorScheme.primary,
                              ),
                            ),
                            Switch(
                              value: _useSpeculative,
                              onChanged: _isLoading
                                  ? null
                                  : (value) {
                                      setState(() {
                                        _useSpeculative = value;
                                      });
                                      _initEngine();
                                    },
                            ),
                          ],
                        ),
                        if (_useSpeculative) ...[
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                'Speculated Tokens: $_speculateTokens',
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.bold,
                                  color: colorScheme.primary,
                                ),
                              ),
                            ],
                          ),
                          Slider(
                            value: _speculateTokens.toDouble(),
                            min: 1,
                            max: 10,
                            divisions: 9,
                            label: _speculateTokens.toString(),
                            onChanged: _isLoading
                                ? null
                                : (value) {
                                    setState(() {
                                      _speculateTokens = value.round();
                                    });
                                  },
                            onChangeEnd: _isLoading
                                ? null
                                : (value) {
                                    _initEngine();
                                  },
                          ),
                        ],
                        const SizedBox(height: 8),
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
                  'Powered by Qwen & Rust Core',
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
        color: colorScheme.surfaceContainerHighest.withOpacity(0.3),
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
          if (_useSpeculative)
            _buildMetricItem(
              Icons.check_circle_outline,
              '${_avgAcceptedTokens.toStringAsFixed(2)} acc/step',
              Colors.green,
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

  Widget _buildModelInfoChip(String label, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
