FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CFLAGS="-O3 -march=native -mavx2 -mfma"
ENV CXXFLAGS="-O3 -march=native -mavx2 -mfma"
ENV TORCH_CPU_ARCH="avx2"
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TF_NUM_INTEROP_THREADS=4
ENV TF_NUM_INTRAOP_THREADS=4

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY benchmark_intel.py .

CMD ["python3", "benchmark_intel.py"]