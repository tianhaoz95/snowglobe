# Qwen 3.5 Support Integration Report

This report outlines the technical changes and architectural adaptations required to support Qwen 3.5 models (specifically the 0.8B variant) within the Snowglobe engine and demonstration application.

## 1. Architectural Adaptations

### 1.1 Robust Configuration Parsing
Qwen 3.5 models (especially multimodal variants like the 0.8B-Instruct) often use a nested `text_config` structure in their `config.json` file. The initial implementation of `Snowglobe` expected all configuration fields to be at the top level.

**Change:** Updated `init_model` in `engine/src/lib.rs` to attempt parsing the top-level `config.json` first, and if that fails, look for a nested `text_config` key. This ensures compatibility with both standard and multimodal Qwen model configurations.

### 1.2 Enhanced `QwenConfig` Flexibility
Different versions of Qwen models may omit certain fields or nest them within sub-objects (e.g., `rope_theta` nested within `rope_parameters`).

**Change:** Added default values to all fields in the `QwenConfig` struct in `engine/src/model/qwen.rs` using `#[serde(default = "...")]`. This makes the engine more resilient to variations in model metadata while providing sensible defaults for common LLM parameters.

## 2. Tokenizer Compatibility

Qwen 3.5 features a significantly larger vocabulary (248,320 tokens) compared to earlier versions (e.g., Qwen 3's ~151k). 

**Change:** Updated the demonstration app (`demo/lib/main.dart`) and the engine's downloader (`engine/src/utils/downloader.rs`) to point to the correct Qwen 3.5-specific `tokenizer.json` and `config.json` files. Using the incorrect tokenizer previously resulted in garbled output due to vocabulary ID mismatches.

## 3. Reasoning and Chain-of-Thought (CoT)

Qwen 3.5 0.8B-Instruct natively supports reasoning tokens (e.g., `<think>` and `</think>`).

**Observation:** During integration testing, the model emitted thought tags before providing the final answer.
**Adaptation:** Relaxed integration test expectations (`demo/integration_test/chat_test.dart` and internal unit tests) to allow for these reasoning tokens while still validating the correctness of the final response (e.g., checking if the response *contains* the expected answer rather than being an exact match).

## 4. User Interface Enhancements

To facilitate testing across different model versions, the demonstration application was updated.

**Change:**
- Introduced a `ModelType` enum to manage supported models (Qwen 2.5, Qwen 3, and Qwen 3.5).
- Added a model selection dropdown in the settings section of the app.
- Automated engine re-initialization upon model selection change.

## 5. Stability Fixes

**Async Safety:** Added `mounted` checks in the Flutter demo app's `_generateResponse` and `_initEngine` methods to prevent `setState()` calls after a widget has been disposed, a common source of crashes in async-heavy integration tests.
