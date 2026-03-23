FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libssl-dev \
    curl \
    unzip \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    clang \
    libgoogle-glog-dev \
    libgtest-dev \
    libncurses5 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Flutter dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Set up Android NDK (Version matching the build script or a reasonable default)
# The user might want to mount their own NDK, but providing a default path is helpful.
ENV ANDROID_NDK_HOME=/opt/android-ndk
RUN mkdir -p ${ANDROID_NDK_HOME} && \
    wget -q https://dl.google.com/android/repository/android-ndk-r26b-linux.zip && \
    unzip -q android-ndk-r26b-linux.zip && \
    mv android-ndk-r26b/* ${ANDROID_NDK_HOME} && \
    rm -rf android-ndk-r26b-linux.zip android-ndk-r26b

ENV ANDROID_NDK_ROOT=${ANDROID_NDK_HOME}
ENV ANDROID_NDK=${ANDROID_NDK_HOME}

# Set up Qualcomm QNN SDK environment variables
# Note: User must provide the actual SDK by mounting it to /opt/qnn-sdk
ENV QNN_SDK_ROOT=/opt/qnn-sdk
ENV LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}"
ENV PATH="${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}"

# Install Python dependencies for ExecuTorch and Snowglobe
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    executorch \
    transformers \
    safetensors \
    huggingface_hub \
    pydot \
    numpy \
    onnx

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
