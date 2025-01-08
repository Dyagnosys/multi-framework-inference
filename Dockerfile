FROM openvino/onnxruntime_ep_ubuntu20

WORKDIR /app
USER root

# Performance optimization environment variables
ENV MKLDNN_VERBOSE=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV VECLIB_MAXIMUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-base \
    libgomp1 \
    intel-mkl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p assets/models assets/videos

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY assets/ assets/

# Default command
CMD ["python3", "main.py"]