
# Comprehensive Framework Benchmark

This repository provides a comprehensive benchmarking suite for various machine learning and inference frameworks, including:

- ONNX Runtime
- NCNN
- MNN
- TVM
- llama.cpp

The benchmarks evaluate performance (latency, throughput, etc.) across different model architectures such as MLP, CNN, and Transformer. The entire setup is containerized using Docker for ease of reproducibility and environment consistency.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Benchmark](#running-the-benchmark)
- [Results and Outputs](#results-and-outputs)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites
- Docker installed on your system
- Basic familiarity with Docker commands and Python

## Repository Structure
```
├── Dockerfile
├── requirements.txt
├── benchmark.py
└── README.md
```
- **Dockerfile**: Defines the multi-stage Docker build for setting up all required frameworks and dependencies
- **requirements.txt**: Lists the Python dependencies needed for running the benchmark
- **benchmark.py**: Contains the Python script to run the benchmarks, generate metrics, and create visualizations
- **README.md**: Documentation for the repository (this file)

## Building the Docker Image
Clone the repository:

```bash
git clone https://github.com/your_username/comprehensive-framework-benchmark.git
cd comprehensive-framework-benchmark
```

Build the Docker image:

```bash
docker build -t comprehensive-benchmark .
```

This command will use the provided Dockerfile to set up the environment, compile necessary libraries, and install all dependencies.

## Running the Benchmark
Once the Docker image is built, run the container to execute the benchmark:

```bash
docker run --rm comprehensive-benchmark
```

The benchmarking script will:
- Create test models for MLP, CNN, and Transformer architectures
- Run inference benchmarks across different frameworks and thread configurations
- Collect and print metrics (average latency, P95 latency, throughput, etc.)
- Save results to CSV and JSON files
- Generate performance comparison plots and heatmaps as PNG images

## Results and Outputs
After running the benchmark, the following files will be generated in the container's working directory:
- **benchmark_results.csv**: Raw benchmarking results
- **system_info.json**: Information about the system where the benchmark was run
- **mlp_benchmark_results.png**: Performance comparison plots for the MLP model
- **cnn_benchmark_results.png**: Performance comparison plots for the CNN model
- **transformer_benchmark_results.png**: Performance comparison plots for the Transformer model
- **throughput_heatmap.png**: A heatmap visualizing throughput across models, frameworks, and thread configurations

To extract these files from the Docker container, consider mounting a host directory as a volume when running the container:

```bash
docker run --rm -v $(pwd)/results:/app comprehensive-benchmark
```

This mounts the local results directory to `/app` in the container, so all output files will be saved locally in the `results` folder.

## Customization
- **Models & Frameworks**: Modify `benchmark.py` to add new models, frameworks, or change configurations
- **Dockerfile**: Adjust the Dockerfile to install different versions of dependencies or support additional frameworks
- **Thread Configurations**: Adjust `thread_configs` in the `run_full_benchmark` method of `benchmark.py` for different thread counts

## Troubleshooting
- **Package Installation Errors**: Ensure that the Docker build process installs all required dependencies. The Dockerfile uses version-specific LLVM packages (llvm-10-dev, etc.) for Ubuntu 20.04. If errors occur, check package names or update to a compatible version.
- **Framework Errors**: If benchmarks fail for a specific framework, verify its installation and configuration steps in the Dockerfile and ensure system compatibility.
- **Resource Limitations**: Ensure your system has sufficient resources (CPU, memory) to run benchmarks, especially for TVM and llama.cpp.

