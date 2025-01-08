# Base stage for common dependencies
FROM ubuntu:20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install common system dependencies and newer CMake in one RUN
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget build-essential ninja-build \
    libopenblas-dev software-properties-common gfortran libatlas-base-dev \
    liblapack-dev libblas-dev pkg-config libssl-dev gpg && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
      | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Install LLVM dependencies in a separate RUN
RUN apt-get update && apt-get install -y \
    llvm-10-dev clang libclang-10-dev && \
    rm -rf /var/lib/apt/lists/*

# Set CPU flags and upgrade pip in another RUN
ENV CFLAGS="-O3 -msse4.2 -mssse3"
ENV CXXFLAGS="-O3 -msse4.2 -mssse3"
RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- NCNN Stage ----
FROM base AS ncnn-builder
WORKDIR /build
# Separate NCNN build steps into individual RUN commands for clarity
RUN git clone https://github.com/Tencent/ncnn.git
RUN cd ncnn && mkdir -p build
RUN cd ncnn/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=OFF \
          -DNCNN_SSE2=ON \
          -DNCNN_AVX=OFF \
          -DNCNN_AVX2=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DCMAKE_INSTALL_PREFIX=/install ..
RUN cd ncnn/build && \
    cmake --build . --config Release -j$(nproc) && \
    cmake --install .

# ---- MNN Stage ----
FROM base AS mnn-builder
WORKDIR /build
RUN git clone https://github.com/alibaba/MNN.git
RUN cd MNN && mkdir -p build
RUN cd MNN/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DMNN_BUILD_SHARED_LIBS=ON \
          -DMNN_SEP_BUILD=OFF \
          -DMNN_BUILD_TRAIN=OFF \
          -DMNN_BUILD_DEMO=OFF \
          -DMNN_USE_SSE=ON \
          -DMNN_SUPPORT_BF16=OFF \
          -DCMAKE_INSTALL_PREFIX=/install .. 
RUN cd MNN/build && \
    cmake --build . --config Release -j$(nproc) && \
    cmake --install .

# ---- llama.cpp Stage ----
FROM base AS llamacpp-builder
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp
RUN cd llama.cpp && mkdir -p build
RUN cd llama.cpp/build && \
    cmake .. \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_AVX=OFF \
        -DLLAMA_AVX2=OFF \
        -DLLAMA_AVX512=OFF \
        -DLLAMA_FMA=OFF \
        -DLLAMA_F16C=OFF \
        -DCMAKE_INSTALL_PREFIX=/install \
        -DCMAKE_BUILD_TYPE=Release
RUN cd llama.cpp/build && \
    cmake --build . --config Release -j$(nproc) && \
    mkdir -p /install/bin && \
    cp bin/* /install/bin/

# ---- TVM Stage ----
FROM base AS tvm-builder
WORKDIR /build

# Install additional Python dependencies for TVM in separate RUN
RUN python3 -m pip install numpy scipy tornado typing_extensions psutil pytest wheel onnx

# Clone TVM repository in its own RUN
RUN git clone --recursive https://github.com/apache/tvm tvm

WORKDIR /build/tvm
RUN mkdir -p build && cp cmake/config.cmake build/config.cmake
RUN sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' build/config.cmake && \
    sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA OFF)/' build/config.cmake

RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/install \
        -DUSE_LLVM=ON \
        -DUSE_CUDA=OFF \
        -DUSE_VULKAN=OFF \
        -DUSE_OPENCL=OFF \
        -DCMAKE_CXX_STANDARD=17 && \
    cmake --build . --config Release -j$(nproc)

RUN cd /build/tvm/python && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --target=/install/python-packages -e .

# ---- Final Stage ----
FROM base

# Fixed environment variable assignments to avoid undefined variable warnings
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib
ENV PYTHONPATH=/usr/local/lib/python3.8/dist-packages:/install/python-packages

ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4  
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

COPY requirements.txt .
COPY benchmark_no_tvm.py . 

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir \
    onnx \
    onnxruntime==1.16.0 \
    tvm \
    MNN \
    llama-cpp-python \
    seaborn

RUN ldconfig

CMD ["python3", "benchmark_no_tvm.py"]
