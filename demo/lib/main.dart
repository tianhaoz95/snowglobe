import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
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
  String _response = 'Press the send button to generate a response.';
  bool _isLoading = false;
  String? _sessionId; // Store session ID
  Future<void>? _initEngineFuture; // Changed to Future<void>

  @override
  void initState() {
    super.initState();
    _promptController = TextEditingController(text: 'what is 1+1? only answer with numbers');
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
      print('Creating cache directory: ${cacheDir.path}');
      await cacheDir.create(recursive: true);
    }
    print('Cache directory: ${cacheDir.path}');

    // Download model and tokenizer before initializing engine
    await _downloadModelAndTokenizer(cacheDir.path);

    final initResult = await initEngine(cacheDir: cacheDir.path);
    print('Engine initialized: $initResult');
    _sessionId = await initSession(); // Store the session ID
    print('Session ID: $_sessionId');
  }

  Future<void> _downloadModelAndTokenizer(String cacheDir) async {
    const modelUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors';
    const tokenizerUrl =
        'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json';

    final modelPath = '$cacheDir/model.safetensors';
    final tokenizerPath = '$cacheDir/tokenizer.json';

    setState(() {
      _response = 'Downloading model and tokenizer...';
      _isLoading = true;
    });

    try {
      // Replicate engine logic: tokenizer only if it doesn't exist
      if (!await File(tokenizerPath).exists()) {
        print('Downloading tokenizer...');
        await _downloadFile(tokenizerUrl, tokenizerPath, 'tokenizer');
      }

      // Replicate engine logic: engine currently deletes and re-downloads model
      // so we do the same to ensure it's there for the engine to (potentially)
      // replace or use. Note: engine's debug behavior will still delete it.
      if (!await File(modelPath).exists()) {
        print('Downloading model...');
        await _downloadFile(modelUrl, modelPath, 'model');
      }
    } catch (e) {
      print('Download error: $e');
    } finally {
      setState(() {
        _isLoading = false;
        _response = 'Downloads complete. Initializing engine...';
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
            final progress = (downloaded / contentLength * 100).toStringAsFixed(1);
            setState(() {
              _response = 'Downloading $label: $progress%';
            });
          } else {
            setState(() {
              _response = 'Downloading $label: ${downloaded ~/ 1024} KB';
            });
          }
        }
        await sink.close();
        print('Downloaded: $savePath');
      } else {
        print('Failed to download $url: ${response.statusCode}');
      }
    } finally {
      httpClient.close();
    }
  }

  Future<void> _generateResponse() async {
    if (_sessionId == null || _isLoading) return;

    setState(() {
      _isLoading = true;
      _response = ''; // Clear previous response
    });

    try {
      final String currentPrompt = _promptController.text;
      final stopwatch = Stopwatch()..start(); // Start stopwatch
      
      final tokenStream = generateResponse(
        sessionId: _sessionId!,
        prompt: currentPrompt,
      );

      await for (final token in tokenStream) {
        setState(() {
          _response += token;
        });
      }

      stopwatch.stop(); // Stop stopwatch
      print('Generation finished in ${stopwatch.elapsed.inMilliseconds / 1000.0} seconds');
    } catch (e) {
      setState(() {
        _response = 'Error: $e';
      });
      print('Error generating response: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Rust-powered LLM Demo')),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              FutureBuilder<void>(
                future: _initEngineFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const LinearProgressIndicator();
                  } else if (snapshot.hasError) {
                    return Text('Error initializing engine: ${snapshot.error}');
                  } else {
                    return const SizedBox.shrink(); // Engine initialized
                  }
                },
              ),
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(
                  labelText: 'Enter your prompt',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isLoading ? null : _generateResponse,
                child: _isLoading
                    ? const CircularProgressIndicator(color: Colors.white)
                    : const Text('Generate Response'),
              ),
              const SizedBox(height: 20),
              Expanded(
                child: SingleChildScrollView(
                  child: Text(
                    'Response: $_response',
                    style: const TextStyle(fontSize: 16),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
