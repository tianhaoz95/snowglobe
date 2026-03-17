# Integration Test Model Caching Design

## Problem Statement
Currently, running the integration test for Snowglobe (`demo/integration_test/chat_test.dart`) requires downloading the LLM model weights (e.g., Qwen 3.5-0.8B GGUF, ~500MB) and its metadata (`tokenizer.json`, `config.json`) onto the device every time a fresh installation occurs or when cache is cleared. This leads to:
1. Significant test latency (up to several minutes depending on network).
2. Unnecessary bandwidth usage.
3. Test instability due to potential network failures during download.

## Current Side-Loading Support
The application already has some basic support for side-loading models from the device's local storage:
- **Android:** Checks for `/data/local/tmp/snowglobe/model.gguf` (for LlamaCpp) or `/data/local/tmp/snowglobe/model.pte` (for ExecuTorch).
- **Generic:** Checks for `../model.gguf` relative to the application's working directory.

However, this support is limited because:
1. It only covers the model weights, not the `tokenizer.json` and `config.json`.
2. It doesn't distinguish between different models (e.g., Qwen 3 vs Qwen 3.5) in the side-loading paths.
3. There is no automated script to manage the host-to-device transfer before running tests.

## Proposed Solution: Host-Device Sync Architecture

We propose a "Download Once, Push Many" strategy to optimize the development and testing loop.

### 1. Host-Side Model Cache
Establish a persistent cache directory in the project root (e.g., `.test_assets/`) to store model files.
```
.test_assets/
└── qwen3_5/
    ├── model.gguf
    ├── tokenizer.json
    └── config.json
```
This directory should be added to `.gitignore`.

### 2. Enhanced App Side-Loading Logic
Update `demo/lib/main.dart` to look for all three required files in model-specific subdirectories on the device. 

**Proposed code change in `_downloadModelAndTokenizer`:**

```dart
    final localBase = '../.test_assets/${_selectedModel.name}';
    final androidBase = '/data/local/tmp/snowglobe/${_selectedModel.name}';

    Future<bool> trySideLoad(String fileName, String targetPath) async {
      final localFile = File('$localBase/$fileName');
      final androidFile = File('$androidBase/$fileName');

      if (await localFile.exists()) {
        setState(() => _response = 'Copying local $fileName...');
        await localFile.copy(targetPath);
        return true;
      } else if (Platform.isAndroid && await androidFile.exists()) {
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
        if (USE_LLAMACPP) {
          if (!await trySideLoad('model.gguf', modelPath)) {
            // Fallback to legacy path for backward compatibility
            if (!await trySideLoad('../model.gguf', modelPath)) {
               await _downloadFile(ggufUrl, modelPath, 'Model weights (GGUF)');
            }
          }
        }
        // ... similar for other backends
      }
```

### 3. Automated Test Runner Script
Create a helper script (e.g., `scripts/run_chat_test.sh`) that automates the following steps:
1. **Cache Check:** Check if the requested model assets exist in `.test_assets/`. If not, download them once using `curl` or `wget`.
2. **Device Detection:** Detect connected Android devices via `adb devices`.
3. **Asset Push:** Use `adb push .test_assets/qwen3_5/* /data/local/tmp/snowglobe/qwen3_5/` to sync files.
4. **Test Execution:** Run the Flutter integration test:
   ```bash
   flutter test integration_test/chat_test.dart -d <device_id>
   ```

### 4. iOS Integration (Simulator)
For iOS simulators, the script can locate the simulator's filesystem path and copy the files directly into the app's `Library/Application Support/` folder before running the test, bypassing the sandbox restrictions usually found on real devices.

## Benefits
- **Zero-Download Tests:** Once the host cache is warm, tests start generating responses almost immediately after app launch.
- **Reliability:** Decouples integration testing from external network conditions and Hugging Face availability.
- **Support for All Backends:** The same logic can be extended to `.pte` (ExecuTorch) and `.safetensors` (Burn) models.

## Implementation Tasks
- [ ] Update `demo/lib/main.dart` to support side-loading metadata (`tokenizer.json`, `config.json`) and model-specific paths.
- [ ] Implement `scripts/sync_test_assets.sh` for downloading and pushing assets.
- [ ] Add `.test_assets/` to `.gitignore`.
- [ ] Update `GEMINI.md` with instructions on using the optimized test runner.
