#!/bin/bash
cd demo
flutter run integration_test/chat_test.dart -d HA1EY3WF --flavor highPerf --no-pub > test_output.log 2>&1 &
FLUTTER_PID=$!
echo "Started flutter run with PID $FLUTTER_PID"
# Wait for test to complete or timeout
for i in {1..300}; do
    if grep -q "CHAT TEST - SUCCESS" test_output.log; then
        echo "Test detected SUCCESS!"
        break
    fi
    if grep -q "CHAT TEST - FAILURE" test_output.log; then
        echo "Test detected FAILURE!"
        break
    fi
    # Print progress every 10 iterations
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still waiting... ($i iterations)"
        tail -n 5 test_output.log
    fi
    sleep 2
done
kill $FLUTTER_PID || true
cat test_output.log | grep -A 50 "CHAT TEST"
