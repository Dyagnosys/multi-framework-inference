import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import intel_extension_for_pytorch as ipex
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        
        self.i_downsample = None
        if stride != 1 or in_channels != out_channels:
            self.i_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        return F.relu(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=7, channels=3):
        super().__init__()
        self.conv_layer_s2_same = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Block(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layer_s2_same(x)
        x = F.relu(self.batch_norm1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2, num_classes=7, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm1(x, (h0_1, c0_1))
        
        h0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size // 2).to(x.device)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size // 2).to(x.device)
        
        out, _ = self.lstm2(out, (h0_2, c0_2))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class BenchmarkModels:
    def __init__(self, num_warmup=50, num_iterations=1000):
        self.results = []
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.models = self._create_models()
        self._initialize()

    def _create_models(self):
        return {
            'resnet50': ResNet50(num_classes=7, channels=3),
            'lstm': LSTMModel(input_size=512, hidden_size=512)
        }

    def _initialize(self):
        torch.set_num_threads(psutil.cpu_count(logical=True))
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True

    def benchmark_pytorch(self, model_name, threads):
        model = self.models[model_name].eval()
        
        # Create appropriate input tensor
        if model_name == 'resnet50':
            input_tensor = torch.randn(1, 3, 224, 224)
        else:  # LSTM
            input_tensor = torch.randn(1, 16, 512)
        
        # Optimize with IPEX
        model = ipex.optimize(model)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(input_tensor)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = model(input_tensor)
                latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)
        results = {
            'model': model_name,
            'threads': threads,
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'throughput': 1000 / np.mean(latencies)
        }
        
        logger.info(f"\n{model_name} with {threads} threads:")
        logger.info(f"Average latency: {results['avg_latency']:.2f} ms")
        logger.info(f"P95 latency: {results['p95_latency']:.2f} ms")
        logger.info(f"Throughput: {results['throughput']:.2f} inf/sec")
        
        return results

    def run_benchmark(self):
        thread_configs = [1, 2, 4]  # Based on your CPU
        
        for model_name in self.models.keys():
            for threads in thread_configs:
                torch.set_num_threads(threads)
                try:
                    results = self.benchmark_pytorch(model_name, threads)
                    self.results.append(results)
                except Exception as e:
                    logger.error(f"Error benchmarking {model_name}: {e}")
        
        self._save_results()

    def _save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv('benchmark_results.csv', index=False)
        
        plt.figure(figsize=(12, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.plot(model_data['threads'], model_data['throughput'], 
                    marker='o', label=model)
        
        plt.xlabel('Threads')
        plt.ylabel('Throughput (inf/sec)')
        plt.title('Model Performance vs Thread Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('performance_analysis.png')
        plt.close()

def main():
    benchmark = BenchmarkModels()
    benchmark.run_benchmark()

if __name__ == '__main__':
    main()