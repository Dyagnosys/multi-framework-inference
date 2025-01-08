
# Emotion Recognition System with ONNX and OpenVINO

This project implements a high-performance emotion recognition pipeline using ONNX and OpenVINO for real-time video processing. The pipeline is optimized for Intel architectures and utilizes threading, batching, and advanced inference optimizations to achieve high throughput with low latency. The emotion recognition model is based on a pre-trained ONNX model for facial emotion recognition and integrates seamlessly with video processing workflows.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Model Details](#model-details)
- [Performance Optimization](#performance-optimization)
- [Configuration and Environment Variables](#configuration-and-environment-variables)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **High-Performance Inference**: Utilizes ONNX Runtime with Intel-specific optimizations, including MKL and MKLDNN, for fast CPU inference.
- **Real-Time Video Processing**: Processes video frames concurrently using threading and batching, optimizing for real-time emotion detection.
- **Multi-threading and Queuing**: Efficient use of queues and background threads to manage frame preprocessing, batching, and inference.
- **Emotion Recognition**: Recognizes seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Video Annotation**: Overlays emotion labels on video frames and outputs annotated video.
- **Model Flexibility**: Easily switch between different model architectures (e.g., ResNet50, LSTM) and ONNX models.

---

## Repository Structure

```
├── assets
│   ├── models
│   │   └── FER_static_ResNet50_AffectNet.onnx  # Pre-trained ONNX model
│   └── videos
│       └── input.mp4                        # Default input video
├── Dockerfile                               # Docker configuration for setting up environment
├── main.py                                  # Main script to run emotion recognition pipeline
├── model_architectures.py                   # PyTorch model definitions (e.g., ResNet50, LSTMPyTorch)
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies
```

- **main.py**: Contains the `EmotionProcessor` class and video processing logic. It manages frame queues, preprocessing, batching, and inference using ONNX Runtime.
- **model_architectures.py**: Contains PyTorch model definitions for ResNet50 and LSTMPyTorch. These can be used to train or export models to ONNX if needed.
- **assets/**: Directory for models and video assets.
- **Dockerfile**: Provides a Docker environment for reproducible setup and execution.

---

## Requirements

- Python 3.6 or later
- Intel CPU with support for MKLDNN (optional for enhanced performance)
- ONNX Runtime with OpenVINO execution provider

### Python Dependencies

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `onnx`
- `onnxruntime-openvino`
- `matplotlib`
- `librosa`
- `scikit-learn`
- `pillow`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/emotion-recognition.git
    cd emotion-recognition
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that the pre-trained model `FER_static_ResNet50_AffectNet.onnx` is placed in the `assets/models/` directory and the input video `input.mp4` is in `assets/videos/`.

---

## Docker Setup

This project provides a Dockerfile to set up the environment with all dependencies. The container is based on the `openvino/onnxruntime_ep_ubuntu20` image.

### Docker Build and Run

1. **Build the Docker image**:
    ```bash
    docker build -t emotion-recognition .
    ```

2. **Run the Docker container**:
    ```bash
    docker run --rm -v /path/to/your/video:/app emotion-recognition
    ```
    Replace `/path/to/your/video` with the actual path to your video files if needed.

The Docker container ensures that all optimized libraries and dependencies are correctly configured for Intel architectures.

---

## Usage

### Input Video

- Place the input video in the `assets/videos/` directory as `input.mp4` or modify the path in `main.py` accordingly.
- The system processes each frame, overlays emotion predictions, and writes the annotated frames to an output video.

### Running the Application

Execute the main script:
```bash
python main.py
```
The system will:
- Initialize the `EmotionProcessor` with the ONNX model.
- Read frames from the input video.
- Preprocess and batch frames.
- Run inference on each batch.
- Overlay emotion predictions on frames.
- Save the annotated video as `output.mp4`.

### Output Video

- The processed video with emotion labels will be saved as `output.mp4` in the project directory by default.
- You can adjust input/output paths and parameters by modifying the corresponding variables and functions in `main.py`.

---

## Model Details

### ResNet50 Architecture (`model_architectures.py`)

The `ResNet50` class in `model_architectures.py` defines a modified ResNet50 architecture tailored for emotion recognition:

- It uses a custom initial convolution layer with stride 2 and same padding.
- Pre-trained ResNet50 layers are loaded from `torchvision.models`.
- The final fully connected layer (`fc1`) is modified to output predictions for 7 emotion classes.
- Features can be extracted using the `extract_features` method for further processing or analysis.

### LSTM Architecture (`model_architectures.py`)

The `LSTMPyTorch` class defines an LSTM-based model for sequence-based emotion analysis:

- Two sequential LSTM layers process input sequences.
- The final fully connected layer (`fc`) maps LSTM outputs to emotion class predictions.
- This architecture can be useful for processing sequences of features or temporal data.

### Model Conversion and Export

To convert these PyTorch models to ONNX format:
```python
import torch
from model_architectures import ResNet50

model = ResNet50(num_classes=7)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "assets/models/FER_static_ResNet50_AffectNet.onnx", 
                  input_names=["input"], output_names=["output"], opset_version=11)
```
This will generate an ONNX model that can be used by the EmotionProcessor.

---

## Performance Optimization

### Batching and Threading

- **Batch Processing**: Frames are batched in groups of 4 (`BATCH_SIZE = 4`) to maximize throughput while maintaining low latency.
- **Threading**: A separate processing thread handles preprocessing and inference, using queues to manage input and output frames concurrently.

### Optimized Inference Settings

- **ONNX Runtime Configuration**: The `EmotionProcessor` configures the ONNX session for optimal performance:
  - Enables all graph optimizations.
  - Sets intra- and inter-op parallelism threads.
  - Uses Intel MKLDNN execution provider with specific thread settings.

- **Environment Variables**: Tweak system-level threading and library behavior using environment variables:
  - `MKLDNN_VERBOSE=1`
  - `KMP_AFFINITY=granularity=fine,compact,1,0`
  - `KMP_BLOCKTIME=1`
  - `OMP_NUM_THREADS=2`
  - `MKL_NUM_THREADS=2`
  - `OPENBLAS_NUM_THREADS=2`
  - `VECLIB_MAXIMUM_THREADS=2`
  - `NUMEXPR_NUM_THREADS=2`

These variables control how libraries allocate threads and can be set in your shell or Docker environment to optimize performance.

### Additional Optimization Suggestions

- **Quantization**: Use post-training quantization to reduce model size and speed up inference.
- **Multi-threading in Python**: Adjust the number of threads and experiment with Python's Global Interpreter Lock (GIL) limitations.
- **Input Pipeline Optimization**: Preprocess inputs in parallel, cache results, or use more efficient data formats.
- **Framework Parameters**: Tune parameters like batch size, inference precision (FP16/INT8), and thread parallelism for your hardware.

### Model Architecture Considerations

- **CPU-Friendly Architectures**: Consider models like EfficientNet or MobileNet for better performance on CPU.
- **Layer Optimization**: Simplify the network architecture where possible to reduce computation without sacrificing accuracy.

---

## Configuration and Environment Variables

To optimize performance, set these environment variables before running the application:
```bash
export MKLDNN_VERBOSE=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
```
These variables configure threading, affinity, and library behaviors to maximize CPU utilization and inference speed.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Implement your changes with clear commit messages.
4. Submit a pull request, detailing your changes and the problem they solve.

For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
