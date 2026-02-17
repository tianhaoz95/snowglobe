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
    await RustLib.init();
    final cacheDir = await getApplicationSupportDirectory();
    await initEngine(cacheDir: cacheDir.path);
    _sessionId = await initSession(); // Store the session ID
    print('Session ID: $_sessionId');
  }

  Future<void> _generateResponse() async {
    if (_sessionId == null || _isLoading) return;

    setState(() {
      _isLoading = true;
      _response = 'Generating response...';
    });

    try {
      final String currentPrompt = _promptController.text;
      final stopwatch = Stopwatch()..start(); // Start stopwatch
      final response = await generateResponse(sessionId: _sessionId!, prompt: currentPrompt);
      stopwatch.stop(); // Stop stopwatch
      setState(() {
        _response = 'Generated in ${stopwatch.elapsed.inMilliseconds / 1000.0} seconds\n$response';
      });
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
