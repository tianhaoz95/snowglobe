#!/bin/bash
cd demo
flutter run integration_test/chat_test.dart -d HA1EY3WF --flavor highPerf --release --no-pub > test_output_rel.log 2>&1 &
FLUTTER_PID=$!
echo "Started flutter run (RELEASE) with PID $FLUTTER_PID"
# Wait for test to complete or timeout
for i in {1..150}; do
    if grep -q "CHAT TEST - SUCCESS" test_output_rel.log; then
        echo "Test detected SUCCESS!"
        break
    fi
    if grep -q "CHAT TEST - FAILURE" test_output_rel.log; then
        echo "Test detected FAILURE!"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still waiting... ($i iterations)"
    fi
    sleep 2
done
kill $FLUTTER_PID || true
cat test_output_rel.log | grep -A 50 "CHAT TEST"
