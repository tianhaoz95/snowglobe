# 0.0.1-dev.4

- Added prebuilt Android binaries (arm64-v8a, x86_64) to support zero-config build for Android.
- Implemented hybrid build logic to fallback to source build if prebuilt binaries are missing.

# 0.0.1-dev.3

- Fixed Rust engine dependency path in published package using public git repository.
- Updated repository URL in pubspec.yaml.

# 0.0.1-dev.2

- Exposed `ModelInfo` and engine metrics.
- Improved session initialization.
- Added token acceptance tracking for speculative decoding.

# 0.0.1-dev.1


- Initial release of snowglobe_openai.
- Wraps snowglobe Rust engine with OpenAI-compatible API.
- Supports Burn, ExecuTorch, and LlamaCpp backends.
