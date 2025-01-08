# Use a base image that supports your specific CPU architecture
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
   build-essential \
   cmake \
   git \
   wget \
   libsndfile1 \
   libopencv-dev \
   python3-opencv \
   && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies without CPU-specific optimizations
RUN pip install --no-cache-dir -r requirements.txt \
   && pip install --no-cache-dir \
   onnxruntime \
   opencv-python-headless

# Set environment variables to disable AVX
ENV MKL_DEBUG_CPU_TYPE=5
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Copy project files
COPY multi_framework_benchmark.py .

# Default command
CMD ["python3", "multi_framework_benchmark.py"]
