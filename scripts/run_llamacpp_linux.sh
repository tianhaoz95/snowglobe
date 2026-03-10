#!/bin/bash
set -e

echo "Setting up compiler wrappers for Linux build..."
# Create a local bin directory for the compiler wrappers to bypass Dart's LLVM linker issue
WRAPPER_DIR="$(pwd)/.tmp_bin"
mkdir -p "$WRAPPER_DIR"

echo '#!/bin/bash' > "$WRAPPER_DIR/clang"
echo 'exec /usr/bin/clang "$@"' >> "$WRAPPER_DIR/clang"

echo '#!/bin/bash' > "$WRAPPER_DIR/clang++"
echo 'exec /usr/bin/clang++ "$@"' >> "$WRAPPER_DIR/clang++"

chmod +x "$WRAPPER_DIR/clang"*

# Symlink the system linkers into the wrapper directory
ln -sf /usr/bin/ld "$WRAPPER_DIR/ld"
ln -sf /usr/bin/ld "$WRAPPER_DIR/ld.lld"
ln -sf /usr/bin/ar "$WRAPPER_DIR/llvm-ar"

# Change to the demo directory
cd demo

# Ensure dependencies are fetched
flutter pub get

echo "Running Snowglobe Linux App with Llama.cpp..."
# Run the app with the wrapped PATH
PATH="$WRAPPER_DIR:$PATH" flutter run -d linux --dart-define=USE_LLAMACPP=true "$@"
