# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libatlas-base-dev \
        libprotobuf-dev \
        protobuf-compiler \
        git \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV TFLITE_MODEL_PATH=models/your_model.tflite
ENV ONNX_MODEL_PATH=models/your_model.onnx

# Run the application
CMD ["python", "compare_models.py"]
