#!/bin/bash
set -e

# Navigate to the app directory (parent of the scripts directory)
cd "$(dirname "$0")/.."

echo "Running integration tests on macOS..."
flutter test -d macos integration_test/app_test.dart
