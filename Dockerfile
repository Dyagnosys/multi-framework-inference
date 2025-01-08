FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set CPU optimization flags
ENV CFLAGS="-O3 -march=native -mavx2 -mfma"
ENV CXXFLAGS="-O3 -march=native -mavx2 -mfma"
ENV TORCH_CPU_ARCH="avx2"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Configure TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1

# Thread control
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV TF_NUM_INTEROP_THREADS=4
ENV TF_NUM_INTRAOP_THREADS=4

COPY benchmark_system.py .

CMD ["python3", "benchmark_system.py"]