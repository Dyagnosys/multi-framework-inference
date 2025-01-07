# ONNX Runtime, NCNN, MNN, TVM, and llama.cpp Benchmark

This repository contains a comprehensive benchmarking suite for multiple deep learning inference frameworks, including ONNX Runtime, NCNN, MNN, TVM, and llama.cpp. The benchmark creates test models (MLP, CNN, Transformer), runs inference across these frameworks, collects performance metrics, and generates visualizations comparing their performance.

## Project Structure

onnx-openvino-benchmark/
├── Dockerfile
├── requirements.txt
├── benchmark.py
└── README.md

- **Dockerfile**: Instructions to build a Docker image with all dependencies and configurations needed for the benchmark.
- **requirements.txt**: Python dependencies required for data handling and plotting.
- **benchmark.py**: The main benchmarking script that creates models, runs inference benchmarks, collects metrics, and generates plots.
- **README.md**: This documentation.

## Overview

The `benchmark.py` script:

- Gathers system information (CPU model, cores, memory, etc.).
- Defines three PyTorch models: 
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
  - Simplified Transformer
- Exports these models to ONNX format.
- Benchmarks inference performance across different frameworks:
  - **ONNX Runtime**
  - **NCNN**
  - **MNN**
  - **TVM**
  - **llama.cpp** (specifically for Transformer models)
- Measures latency, throughput, and other statistics for each framework under various thread configurations.
- Saves raw results as CSV files, system info as JSON, and creates plots for performance comparison.

## Prerequisites

- [Docker](https://www.docker.com/) installed on your system.

## Building the Docker Image

In the project directory, build the Docker image with:

```bash
sudo docker build -t onnx-openvino-benchmark .
sudo docker run --rm -v $(pwd):/app onnx-openvino-benchmark
```

Output and Results

After the benchmark completes, you’ll find:
	•	benchmark_results.csv: Raw results of the benchmarks.
	•	system_info.json: Information about the system used for benchmarking.
	•	{model_type}_benchmark_results.png: Performance comparison plots for each model type (MLP, CNN, Transformer).
	•	throughput_heatmap.png: A heatmap visualizing throughput across different frameworks, models, and thread counts.