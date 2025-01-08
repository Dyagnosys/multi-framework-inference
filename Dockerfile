FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Intel OpenVINO repository
RUN wget https://apt.repos.intel.com/openvino/2022/GPG-PUB-KEY-INTEL-OPENVINO-2022 \
    && apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2022 \
    && echo "deb https://apt.repos.intel.com/openvino/2022 all main" > /etc/apt/sources.list.d/intel-openvino-2022.list \
    && apt-get update \
    && apt-get install -y intel-openvino-dev-ubuntu20-2022.3.0

# Set environment variables for Intel optimizations
ENV MKLDNN_VERBOSE=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy benchmark code
COPY benchmark_intel.py .

# Set environment variables for thread control
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV OPENMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1

CMD ["python3", "benchmark_intel.py"]