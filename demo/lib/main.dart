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
  Future<String>? _initFuture;

  @override
  void initState() {
    super.initState();
    _initFuture = _initEngine();
  }

  Future<String> _initEngine() async {
    await RustLib.init();
    final cacheDir = await getApplicationSupportDirectory();
    await initEngine(cacheDir: cacheDir.path);
    final sessionId = await initSession();
    print('Session ID: $sessionId');
    final response = await generateResponse(sessionId: sessionId, prompt: 'what is 1+1? only answer with numbers');
    print('Response: $response');
    return response;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('flutter_rust_bridge quickstart')),
        body: Center(
          child: FutureBuilder<String>(
            future: _initFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const CircularProgressIndicator();
              } else if (snapshot.hasError) {
                print('Error during initialization: ${snapshot.error}');
                return Text('Error: ${snapshot.error}');
              } else {
                return Text(
                  'Action: Call initEngine() and generateResponse()\nResult: `${snapshot.data}`',
                );
              }
            },
          ),
        ),
      ),
    );
  }
}
