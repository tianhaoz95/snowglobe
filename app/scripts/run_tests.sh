#!/bin/bash
set -e

# Navigate to the app directory (parent of the scripts directory)
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
APP_DIR=$(pwd)
PROJECT_ROOT="$(dirname "$APP_DIR")"

# Setup video recording
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TMP_DIR="$PROJECT_ROOT/tmp"
VIDEO_FILE="$TMP_DIR/test_run_$TIMESTAMP.mp4"

mkdir -p "$TMP_DIR"

echo "Starting video recording..."
# Start ffmpeg in background to record the main screen ("Capture screen 0")
# -y: Overwrite output files
# -r 30: Set frame rate to 30 fps
ffmpeg -f avfoundation -i "Capture screen 0" -r 30 -y "$VIDEO_FILE" > /dev/null 2>&1 &
FFMPEG_PID=$!

# Ensure ffmpeg stops even if the script exits abnormally
cleanup() {
  if kill -0 $FFMPEG_PID 2>/dev/null; then
    echo "Stopping video recording..."
    kill -SIGINT $FFMPEG_PID
    wait $FFMPEG_PID 2>/dev/null || true
    echo "Video saved to $VIDEO_FILE"
  fi
}
trap cleanup EXIT

echo "Running integration tests on macOS..."
# Run the test
set +e
flutter test -d macos integration_test/app_test.dart "$@"
TEST_EXIT_CODE=$?
set -e

exit $TEST_EXIT_CODE
