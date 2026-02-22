# Snowglobe Engine

## Testing

```bash
cargo test test_one_plus_one -- --nocapture

cargo test test_one_plus_one --no-default-features -- --nocapture

cargo test tests::test_one_plus_one --features high_perf -- --nocapture

cargo test tests::test_sharded_one_plus_one --features high_perf -- --nocapture

cargo test tests::test_one_plus_one_pte --release -- --nocapture

cargo test tests::test_one_plus_one_qwen2 --features high_perf -- --nocapture

cargo test tests::test_one_plus_one_qwen3 --features high_perf -- --nocapture
```