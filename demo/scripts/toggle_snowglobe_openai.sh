#!/bin/bash

# Navigate to the demo directory if script is run from project root
cd "$(dirname "$0")/.."

if [ "$1" == "pub" ]; then
  echo "Switching to snowglobe_openai from pub.dev..."
  cat <<OVERRIDE > pubspec_overrides.yaml
dependency_overrides:
  snowglobe_openai: ^0.0.1-dev.2
OVERRIDE
  echo "Created pubspec_overrides.yaml"
elif [ "$1" == "local" ]; then
  echo "Switching to local snowglobe_openai..."
  if [ -f pubspec_overrides.yaml ]; then
    rm pubspec_overrides.yaml
    echo "Removed pubspec_overrides.yaml"
  else
    echo "Already using local version (no override found)."
  fi
else
  echo "Usage: ./scripts/toggle_snowglobe_openai.sh [pub|local]"
  exit 1
fi

flutter pub get
