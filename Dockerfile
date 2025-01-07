# Base stage for common dependencies
FROM ubuntu:20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies and newer CMake
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    ninja-build \
    libopenblas-dev \
    software-properties-common \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CPU flags for optimized builds
ENV CFLAGS="-O3 -msse4.2 -mssse3"
ENV CXXFLAGS="-O3 -msse4.2 -mssse3"

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# NCNN stage
FROM base AS ncnn-builder
WORKDIR /build
RUN git clone https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=OFF \
          -DNCNN_SSE2=ON \
          -DNCNN_AVX=OFF \
          -DNCNN_AVX2=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DCMAKE_INSTALL_PREFIX=/install .. && \
    cmake --build . --config Release -j4 && \
    cmake --install .

# MNN stage
FROM base AS mnn-builder
WORKDIR /build
RUN git clone https://github.com/alibaba/MNN.git && \
    cd MNN && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DMNN_BUILD_SHARED_LIBS=ON \
          -DMNN_SEP_BUILD=OFF \
          -DMNN_BUILD_TRAIN=OFF \
          -DMNN_BUILD_DEMO=OFF \
          -DMNN_USE_SSE=ON \
          -DMNN_SUPPORT_BF16=OFF \
          -DCMAKE_INSTALL_PREFIX=/install .. && \
    cmake --build . --config Release -j4 && \
    cmake --install .

# llama.cpp stage
FROM base AS llamacpp-builder
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    mkdir -p build && cd build && \
    cmake .. \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_AVX=OFF \
        -DLLAMA_AVX2=OFF \
        -DLLAMA_AVX512=OFF \
        -DLLAMA_FMA=OFF \
        -DLLAMA_F16C=OFF \
        -DCMAKE_INSTALL_PREFIX=/install \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j4 && \
    mkdir -p /install/bin && \
    cp bin/* /install/bin/

# TVM stage
FROM base AS tvm-builder
WORKDIR /build

# Install additional Python dependencies for TVM
RUN python3 -m pip install \
    numpy \
    scipy \
    tornado \
    typing_extensions \
    psutil \
    pytest

# Clone TVM with recursive submodules
RUN git clone --recursive https://github.com/apache/tvm && \
    cd tvm

# Prepare build directory and configuration
WORKDIR /build/tvm
RUN mkdir -p build && \
    cp cmake/config.cmake build/config.cmake && \
    sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' build/config.cmake && \
    sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA OFF)/' build/config.cmake

# Build TVM
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/install \
        -DUSE_LLVM=ON \
        -DUSE_CUDA=OFF \
        -DUSE_VULKAN=OFF \
        -DUSE_OPENCL=OFF && \
    cmake --build . --config Release -j4

# Install TVM Python bindings with detailed logging
RUN cd /build/tvm/python && \
    python3 -m pip install -v . && \
    python3 -m pip install -v -e .

# Final stage
FROM base

# Set environment variables for libraries
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/lib:${LD_LIBRARY_PATH:-}"
ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:${PYTHONPATH:-}"

# Copy built artifacts from previous stages
COPY --from=ncnn-builder /install/lib /usr/local/lib/
COPY --from=ncnn-builder /install/include /usr/local/include/
COPY --from=mnn-builder /install/lib /usr/local/lib/
COPY --from=mnn-builder /install/include /usr/local/include/
COPY --from=llamacpp-builder /install/bin /usr/local/bin/
COPY --from=tvm-builder /build/tvm/build /usr/local/lib/tvm
COPY --from=tvm-builder /build/tvm/python /usr/local/lib/python3.8/dist-packages/tvm

# Install Python packages from requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install other Python packages (excluding apache-tvm)
RUN pip3 install --no-cache-dir \
    onnx \
    onnxruntime==1.16.0 \
    ncnn \
    MNN \
    llama-cpp-python

# Set runtime CPU affinity and thread count
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Copy benchmark script
COPY benchmark.py .

# Run ldconfig to update library cache
RUN ldconfig

# Set default command
CMD ["python3", "benchmark.py"]