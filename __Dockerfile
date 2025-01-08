FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Intel CPU optimizations
ENV CFLAGS="-O3 -march=native -mavx2 -mfma"
ENV CXXFLAGS="-O3 -march=native -mavx2 -mfma"
ENV MKLDNN_MAX_CPU_ISA=AVX2
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV TF_NUM_INTEROP_THREADS=4
ENV TF_NUM_INTRAOP_THREADS=4
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TORCH_CPU_ARCH=avx2

# Suppress TensorFlow informational logs
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY benchmark_intel.py .

CMD ["python3", "benchmark_intel.py"]
