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
from pathlib import Path
import onnxruntime as ort
import tensorflow as tf
import intel_extension_for_pytorch as ipex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    input_shape: tuple
    use_int_input: bool = False

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            x = self.shortcut(x)
        return F.relu(out + x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 256, 3)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [Block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class IntelBenchmark:
    def __init__(self):
        self.results = []
        self.system_info = self._get_system_info()
        self.frameworks = ['onnx-cpu', 'pytorch-ipex', 'tensorflow-mkl']
        self.models = {}
        self.num_warmup = 50
        self.num_runs = 1000
        self._initialize()

    def _get_system_info(self):
        return {
            'cpu_model': self._get_cpu_model(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }

    def _get_cpu_model(self):
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except:
            return 'Unknown CPU'

    def _initialize(self):
        # ONNX Runtime
        self.ort_options = ort.SessionOptions()
        self.ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_options.enable_cpu_mem_arena = True
        self.ort_options.add_session_config_entry("session.cache_size_mb", "12288")
        
        # PyTorch/IPEX
        torch.jit.enable_onednn_fusion(True)
        
        # TensorFlow
        tf.config.optimizer.set_jit(True)

    def _get_thread_configs(self):
        return [1, 2, 4]  # Single, half, full cores

    def create_models(self):
        models = {}
        configs = [
            ModelConfig('resnet50', (1, 3, 224, 224))
        ]
        
        for config in configs:
            model = ResNet50()
            input_data = torch.randn(config.input_shape)
            
            # ONNX export
            path = f'{config.name}.onnx'
            torch.onnx.export(model, input_data, path, 
                            opset_version=13,
                            do_constant_folding=True)
            
            # Save model info
            models[config.name] = {
                'path': path,
                'input_shape': config.input_shape,
                'input_data': input_data.numpy()
            }
            
            # Save PyTorch model
            self.models[config.name] = model
            
        return models

    def benchmark_onnx(self, model_info, threads):
        try:
            self.ort_options.intra_op_num_threads = threads
            session = ort.InferenceSession(model_info['path'], 
                                         self.ort_options,
                                         providers=['CPUExecutionProvider'])
            
            def run_inference():
                return session.run(None, {'input': model_info['input_data']})
                
            metrics = self._measure_performance(run_inference)
            metrics['framework'] = 'ONNX-CPU'
            metrics['threads'] = threads
            return metrics
            
        except Exception as e:
            logger.error(f"ONNX error: {e}")
            return {}

    
    def benchmark_pytorch(self, model_name, threads):
        model = self.models[model_name].eval()
        input_shape = (1, 3, 224, 224) if model_name == 'resnet50' else (1, 16, 512)
        input_tensor = torch.randn(input_shape)
        
        # IPEX optimization
        model = ipex.optimize(model)
        input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
        
        # Warmup
        torch.set_num_threads(threads)
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(input_tensor)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = model(input_tensor)
                torch.backends.mkldnn.enabled = True
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

        
    def benchmark_tensorflow(self, model_info, threads):
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
            
            converter = tf.lite.TFLiteConverter.from_saved_model(model_info['path'])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            model = converter.convert()
            
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            
            def run_inference():
                interpreter.set_tensor(input_details[0]['index'], model_info['input_data'])
                interpreter.invoke()
                return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
                
            metrics = self._measure_performance(run_inference)
            metrics['framework'] = 'TensorFlow-MKL'
            metrics['threads'] = threads
            return metrics
            
        except Exception as e:
            logger.error(f"TensorFlow error: {e}")
            return {}

    def _measure_performance(self, run_fn):
        # Warmup
        for _ in range(self.num_warmup):
            run_fn()
            
        # Measure
        latencies = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            run_fn()
            latencies.append((time.perf_counter() - start) * 1000)  # ms
            
        latencies = np.array(latencies)
        return {
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'throughput': 1000 / np.mean(latencies)  # inf/sec
        }

    def run_benchmark(self):
        logger.info("Starting Intel-optimized benchmark...")
        logger.info(f"System info: {json.dumps(self.system_info, indent=2)}")
        
        models = self.create_models()
        thread_configs = self._get_thread_configs()
        
        benchmark_fns = {
            'onnx-cpu': self.benchmark_onnx,
            'pytorch-ipex': self.benchmark_pytorch,
            'tensorflow-mkl': self.benchmark_tensorflow
        }
        
        for model_name, model_info in models.items():
            logger.info(f"\nBenchmarking {model_name}")
            
            for threads in thread_configs:
                for fw_name, benchmark_fn in benchmark_fns.items():
                    try:
                        metrics = benchmark_fn(model_info, threads)
                        if metrics:
                            metrics['model'] = model_name
                            self.results.append(metrics)
                            
                            logger.info(
                                f"{fw_name} ({threads} threads):\n"
                                f"  Avg latency: {metrics['avg_latency']:.2f} ms\n"
                                f"  P95 latency: {metrics['p95_latency']:.2f} ms\n"
                                f"  Throughput: {metrics['throughput']:.2f} inf/sec"
                            )
                    except Exception as e:
                        logger.error(f"Error benchmarking {fw_name}: {e}")
        
        self._save_results()

    def _save_results(self):
        df = pd.DataFrame(self.results)
        
        # Save raw results
        df.to_csv('benchmark_results.csv', index=False)
        
        # Plot thread scaling
        plt.figure(figsize=(12, 6))
        for fw in df['framework'].unique():
            fw_data = df[df['framework'] == fw]
            plt.plot(fw_data['threads'], fw_data['throughput'], 
                    marker='o', label=fw)
        
        plt.xlabel('Threads')
        plt.ylabel('Throughput (inf/sec)')
        plt.title('Framework Performance vs Thread Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('thread_scaling.png')
        
        # Plot latency comparison
        sns.boxplot(data=df, x='framework', y='avg_latency', 
                   hue='threads', dodge=True)
        plt.title('Latency Distribution by Framework and Thread Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('latency_comparison.png')
        
        logger.info("Results saved to benchmark_results.csv")

def main():
    benchmark = IntelBenchmark()
    benchmark.run_benchmark()

if __name__ == '__main__':
    main()