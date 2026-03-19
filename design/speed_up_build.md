# Build Acceleration Design

## Problem Statement
Currently, building and testing Snowglobe is slow, primarily due to long Rust compilation times and the lack of artifact caching. Specifically:
1. **Full Recompilation:** Rust dependencies like `burn`, `llama-cpp-2`, and `reqwest` are large and take significant time to compile from scratch.
2. **Vendored OpenSSL:** The `openssl-sys` crate with the `vendored` feature is used in the Flutter demo's Rust library, adding several minutes to the initial build.
3. **Redundant ABI Builds:** By default, Android builds often target multiple architectures (arm64, armv7, x86_64, x86), which quadruples the Rust compilation time during development.
4. **Third-Party Build Overhead:** C/C++ dependencies like `llama.cpp` and `executorch` are often rebuilt or relinked unnecessarily. `executorch` build scripts even delete their build directories, preventing incremental builds.
5. **Lack of Caching:** There is no global or local caching of compilation artifacts (e.g., `sccache`).

## Proposed Solution: Multi-Layered Build Optimization

We propose a set of strategies to reduce build times by up to 70% for both local development and CI/CD.

### 1. Global Rust Compilation Cache (`sccache`)
Integrate `sccache` into the development workflow and CI/CD.
- **Local:** Provide a setup script that installs `sccache` and configures it via `RUSTC_WRAPPER`.
- **CI/CD:** Use `sccache` with a cloud backend (e.g., GCS or S3) to share build artifacts across different runners.

### 2. Dependency Optimization
Replace slow-building dependencies with faster alternatives.
- **OpenSSL:** Remove `openssl-sys` with `vendored` feature from `demo/rust/Cargo.toml`. Replace it with `rustls` or use the system-provided OpenSSL on Linux/macOS. For Android/iOS, `rustls` is preferred for portability and build speed.
- **Selective Features:** Use a "minimal" feature set for `burn` during development and testing, only enabling `high_perf` (WGPU) when required.

### 3. Architecture-Specific Development Builds
Modify the build scripts and documentation to emphasize single-architecture builds during the development loop.
- **Android:** Default to `--target-platform android-arm64` in `flutter build` and `flutter run`.
- **Cargo:** Use `cargo build --target aarch64-linux-android` directly when possible.

### 4. Persistent Third-Party Build Artifacts
Optimize the compilation of C/C++ dependencies.
- **ExecuTorch:** Modify `scripts/build_executorch_android.sh` to *not* delete the `cmake-android-$ABI` directories. Use them for incremental CMake builds.
- **Pre-built Binaries:** For stable versions of `llama.cpp` and `executorch`, provide pre-compiled static libraries in a dedicated `third_party/libs` directory. Update `engine/build.rs` to prioritize these pre-built libraries if they exist.

### 5. Linker Optimization
Use faster linkers like `mold` (Linux) or `zld` (macOS) to reduce the time spent in the final linking phase of the Rust static libraries.

### 6. Fast Developer Setup Script
Create a `scripts/setup_dev_env.sh` to automate the installation of these optimizations:
```bash
# Example setup script actions:
# 1. Install sccache
# 2. Configure ~/.cargo/config.toml to use sccache
# 3. Install mold/zld if available
# 4. Pre-download/build third-party dependencies
```

## Benefits
- **Faster Iteration:** Reductions in "time-to-test" for Rust engine changes.
- **Reduced CI Costs:** Lower runner time and faster feedback loops in PRs.
- **Easier Onboarding:** New contributors can get a functional development environment faster.

## Implementation Tasks
- [ ] Create `scripts/setup_dev_env.sh` to install `sccache` and configure the environment.
- [ ] Modify `demo/rust/Cargo.toml` to remove `openssl-sys` (vendored) and prefer `rustls`.
- [ ] Update `scripts/build_executorch_android.sh` to support incremental builds by preserving the build directory.
- [ ] Update `engine/build.rs` to support linking against pre-built third-party libraries.
- [ ] Update `GEMINI.md` with best practices for fast builds (e.g., using `--target-platform`).
- [ ] (Optional) Integrate `sccache` into the project's CI/CD pipeline (e.g., GitHub Actions).
