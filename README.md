
# Emotion Recognition System with ONNX and OpenVINO

This project implements a high-performance emotion recognition pipeline using ONNX and OpenVINO with real-time video processing. The pipeline is optimized for Intel architecture and uses threading to process multiple video frames concurrently, achieving high throughput while maintaining low latency. The emotion recognition model is based on a pre-trained ONNX model for facial emotion recognition.

## Features

- **High-Performance Inference**: Utilizes ONNX Runtime with Intel-specific optimizations, including MKL and MKLDNN, for fast and efficient inference on CPU.
- **Real-Time Processing**: Supports real-time processing of video frames with threading and batching techniques.
- **Batching and Threading**: Frames are processed in batches to improve throughput, with multiple threads managing the preprocessing and inference tasks.
- **Emotion Labels**: The system recognizes the following emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Video Processing**: Supports video input and output, processing frames and overlaying emotion predictions onto the video.
  
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

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Docker Setup

This project provides a Dockerfile to set up the environment and dependencies. The container is based on the `openvino/onnxruntime_ep_ubuntu20` image.

### Docker Build and Run

To build and run the Docker container:

1. **Build the Docker image**:
    ```bash
    docker build -t emotion-recognition .
    ```

2. **Run the Docker container**:
    ```bash
    docker run --rm -v /path/to/your/video:/app emotion-recognition
    ```

Replace `/path/to/your/video` with the actual path to the directory containing your input video.

## Usage

### Input Video

The input video should be placed in the `assets/videos/` directory, and the `input.mp4` file will be used by default. You can modify the input path in the `main()` function if necessary.

### Model

The emotion recognition model used in this pipeline is an ONNX model. By default, it loads from the `assets/models/FER_static_ResNet50_AffectNet.onnx` file. You can replace this model with your own ONNX-based emotion recognition model by updating the `model_path` variable in the `main()` function.

### Running the Application

Once the setup is complete, you can run the application with the following command:

```bash
python main.py
```

The system will process the video, detect emotions in each frame, and generate an output video with emotion labels overlaid.

### Output Video

The processed video will be saved to `output.mp4` by default. The emotion label corresponding to the highest probability in each frame will be overlaid on the video. You can change the output path in the `process_video()` function.

## Code Overview

### EmotionProcessor Class

This class manages the emotion recognition process. It performs the following tasks:

1. **Model Initialization**: Initializes the ONNX runtime session with optimized Intel-specific execution providers (e.g., MKLDNN) for better performance.
2. **Preprocessing**: Prepares each frame by resizing and normalizing it according to the model's expected input format.
3. **Batching and Inference**: Processes frames in batches to maximize throughput and performs inference using the ONNX model.
4. **Multi-threading**: Utilizes multiple threads to handle frame processing and inference concurrently, which speeds up the overall pipeline.

### Emotion Recognition Logic

1. **Frame Preprocessing**: The frame is resized to match the model's input size (224x224), normalized, and prepared for inference.
2. **Inference**: Batched frames are passed to the model for emotion recognition. The model's output is a vector of probabilities corresponding to different emotions.
3. **Labeling**: The emotion with the highest probability is extracted, and the corresponding label is overlaid on the video frame.

### Video Processing

The video is read frame by frame, and emotion recognition is applied to each frame. The system processes every 10th frame to avoid overloading the inference process. The frames with their predicted emotions are then written to the output video file.

## Model Optimization

The ONNX Runtime is used to load the pre-trained model and run inference. The system is configured to use Intel-specific execution providers such as MKLDNN to maximize performance on Intel CPUs.

### Environment Variables for Optimization

- `MKLDNN_VERBOSE=1`: Enables verbose output for MKLDNN.
- `KMP_AFFINITY=granularity=fine,compact,1,0`: Controls thread affinity for parallel tasks.
- `KMP_BLOCKTIME=1`: Sets the amount of time a thread is allowed to remain idle before being killed.
- `OMP_NUM_THREADS=2`: Controls the number of threads OpenMP will use.
- `MKL_NUM_THREADS=2`: Controls the number of threads MKL will use.
- `OPENBLAS_NUM_THREADS=2`: Controls the number of threads OpenBLAS will use.
- `VECLIB_MAXIMUM_THREADS=2`: Controls the number of threads for vector library operations.
- `NUMEXPR_NUM_THREADS=2`: Controls the number of threads for NumExpr operations.

## Threading and Queues

The system uses multiple threads and queues to handle the frames:

1. **Input Queue**: Holds frames for processing.
2. **Output Queue**: Holds results from inference to be written to the output video.

The frames are processed in batches, and the results are placed in the output queue for further processing.

## Performance Optimization

Here are a few suggestions for improving inference performance on a CPU-only system:

### Quantize the Model

Quantizing the model weights and activations to INT8 can significantly reduce model size and speed up CPU inference with minimal accuracy loss. Most deep learning frameworks support post-training quantization. For example, in PyTorch, you can use `torch.quantization.quantize_dynamic` to quantize an existing model.

### Enable Multi-threading

Make sure your inference code is taking advantage of multiple CPU cores. Use your framework's options to control thread count, e.g., `torch.set_num_threads(N)` in PyTorch. Experiment with different thread counts to find the optimal tradeoff between throughput and latency for your workload. Be aware of Python's Global Interpreter Lock (GIL), which can limit multi-threading benefits for Python code.

### Optimize Input Pipeline

Loading, decoding, and preprocessing inputs can be a bottleneck, especially at high concurrency. Preprocess inputs offline if possible and store them in an optimized format like TFRecord, LMDB, etc. Use efficient libraries like OpenCV, PIL, and numpy for decoding and transforming. Move preprocessing to background threads to overlap with inference.

### Tune Performance-related Parameters

- **Batch Size**: Larger batches improve throughput but increase latency. Find the right balance.
- **Inference Precision**: FP16 or even FP32 may be faster on some CPUs compared to INT8.
- **Framework Parameters**: Use parameters like `OMP_NUM_THREADS`, `intra/inter_op_parallelism_threads`, etc., to optimize performance.
- **Grid Search**: Perform a grid search to find the optimal combination of parameters for your workload.

### Consider Alternative Architectures

Some model architectures are more CPU-friendly than others. For example, EfficientNets and MobileNets tend to perform well on CPUs. Avoid very deep models, NAS-based models, or those relying heavily on large GEMMs. Architecture changes may require retraining, so quantization and threading optimizations are easier to try first.

## Contributing

Feel free to fork the repository and submit pull requests. If you encounter bugs or have suggestions for improvements, please create an issue in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
