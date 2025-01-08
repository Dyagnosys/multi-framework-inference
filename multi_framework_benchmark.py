import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
import tensorflow as tf
from tqdm import tqdm

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def benchmark_pytorch(model, input_data, num_iterations=50):
    model.eval()
    torch.set_grad_enabled(False)
    
    # Warm-up
    for _ in range(10):
        _ = model(input_data)
    
    # Timing
    start_time = time.time()
    inference_times = []
    
    for _ in range(num_iterations):
        iter_start = time.time()
        _ = model(input_data)
        inference_times.append(time.time() - iter_start)
    
    total_time = time.time() - start_time
    
    return {
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
        'throughput_samples_per_sec': num_iterations / total_time
    }

def benchmark_onnx(model_path, input_data, num_iterations=50):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_data_np = input_data.numpy()
    
    # Warm-up
    for _ in range(10):
        _ = session.run(None, {input_name: input_data_np})
    
    # Timing
    start_time = time.time()
    inference_times = []
    
    for _ in range(num_iterations):
        iter_start = time.time()
        _ = session.run(None, {input_name: input_data_np})
        inference_times.append(time.time() - iter_start)
    
    total_time = time.time() - start_time
    
    return {
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
        'throughput_samples_per_sec': num_iterations / total_time
    }

def convert_to_onnx(model, input_data, output_path):
    torch.onnx.export(
        model,
        input_data,
        output_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output']
    )
    print(f"ONNX model saved to {output_path}")

def main():
    # Configuration
    batch_size = 16
    input_channels = 3
    image_size = (224, 224)
    num_classes = 7

    # Prepare model and input
    model = SimpleResNet(num_classes=num_classes)
    input_data = torch.randn(batch_size, input_channels, *image_size)

    # PyTorch Benchmark
    print("\n=== PyTorch Benchmark ===")
    pytorch_results = benchmark_pytorch(model, input_data)
    print(pytorch_results)

    # Convert to ONNX
    onnx_path = 'model.onnx'
    convert_to_onnx(model, input_data, onnx_path)

    # ONNX Benchmark
    print("\n=== ONNX Benchmark ===")
    onnx_results = benchmark_onnx(onnx_path, input_data)
    print(onnx_results)

def export_results(pytorch_results, onnx_results):
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'Framework', 'Avg_Inference_Time_MS', 
            'Std_Inference_Time_MS', 'Throughput_Samples_Per_Sec'
        ])
        
        writer.writeheader()
        writer.writerow({
            'Framework': 'PyTorch',
            **{k: v for k, v in pytorch_results.items()}
        })
        writer.writerow({
            'Framework': 'ONNX',
            **{k: v for k, v in onnx_results.items()}
        })
    
    print(f"Results exported to {filename}")

if __name__ == "__main__":
    main()
