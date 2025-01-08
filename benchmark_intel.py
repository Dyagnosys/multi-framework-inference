import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
import seaborn as sns

# Optional: Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Added missing model definitions ---
class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50, self).__init__()
        # A simplified ResNet-like block for demonstration
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1, num_classes=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
# --- End of added model definitions ---

class BenchmarkFrameworks:
    def __init__(self, num_warmup=50, num_iterations=1000):
        self.results = []
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.frameworks = ['onnx-mkl', 'torch-ipex', 'tensorflow-mkl']
        self.models = self._create_models()
        self._initialize_frameworks()

    def _create_models(self):
        model_configs = {
            'resnet50': {'input_shape': (1, 3, 224, 224), 'model': ResNet50(num_classes=7)},
            'lstm': {'input_shape': (1, 16, 512), 'model': LSTMModel()}
        }

        for name, config in model_configs.items():
            model = config['model'].eval()
            input_data = torch.randn(config['input_shape'])
            
            # Export ONNX
            torch.onnx.export(model, input_data, f'{name}.onnx',
                              input_names=['input_0'],
                              output_names=['output_0'],
                              dynamic_axes={'input_0': {0: 'batch_size'},
                                            'output_0': {0: 'batch_size'}},
                              opset_version=13)
            
            # Export TorchScript
            traced_model = torch.jit.trace(model, input_data)
            torch.jit.save(traced_model, f'{name}.pt')

            config['model'] = model
            config['input_data'] = input_data.numpy()

        return model_configs

    def _initialize_frameworks(self):
        # ONNX Runtime
        self.ort_options = ort.SessionOptions()
        self.ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_options.enable_cpu_mem_arena = True
        self.ort_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        
        # PyTorch/IPEX
        torch.jit.enable_onednn_fusion(True)
        
        # TensorFlow
        tf.config.optimizer.set_jit(True)

    def benchmark_onnx(self, model_name: str, threads: int) -> dict:
        try:
            self.ort_options.intra_op_num_threads = threads
            session = ort.InferenceSession(
                f"{model_name}.onnx",
                self.ort_options,
                providers=['CPUExecutionProvider']
            )

            input_name = session.get_inputs()[0].name
            input_data = {input_name: self.models[model_name]['input_data']}

            def run_inference():
                return session.run(None, input_data)

            metrics = self._measure_performance(run_inference)
            metrics.update({'framework': 'onnx-mkl', 'model': model_name, 'threads': threads})
            return metrics

        except Exception as e:
            logger.error(f"ONNX error: {str(e)}")
            return {}

    def benchmark_pytorch(self, model_name: str, threads: int) -> dict:
        try:
            torch.set_num_threads(threads)
            model = self.models[model_name]['model']
            input_data = torch.from_numpy(self.models[model_name]['input_data'])

            # IPEX optimization
            model = ipex.optimize(model, dtype=torch.float32)
            input_data = input_data.contiguous(memory_format=torch.channels_last)

            def run_inference():
                with torch.no_grad():
                    return model(input_data)

            metrics = self._measure_performance(run_inference)
            metrics.update({'framework': 'torch-ipex', 'model': model_name, 'threads': threads})
            return metrics

        except Exception as e:
            logger.error(f"PyTorch error: {str(e)}")
            return {}

    def benchmark_tensorflow(self, model_name: str, threads: int) -> dict:
        try:
            tf.config.threading.set_inter_op_parallelism_threads(threads)
            tf.config.threading.set_intra_op_parallelism_threads(threads)

            input_data = self.models[model_name]['input_data']

            # Define and load a TensorFlow/Keras model based on model_name
            if model_name == 'resnet50':
                # Convert input from channels-first (PyTorch) to channels-last (TensorFlow)
                input_data_tf = np.transpose(input_data, (0, 2, 3, 1))
                model = tf.keras.applications.ResNet50(
                    weights=None, input_shape=(224, 224, 3), classes=7
                )
                run_input = input_data_tf
            elif model_name == 'lstm':
                input_data_tf = input_data
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(256, input_shape=(16, 512)),
                    tf.keras.layers.Dense(7)
                ])
                run_input = input_data_tf
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Build the model by running a dummy inference once
            model(tf.convert_to_tensor(run_input, dtype=tf.float32), training=False)

            def run_inference():
                return model(tf.convert_to_tensor(run_input, dtype=tf.float32), training=False)

            metrics = self._measure_performance(run_inference)
            metrics.update({'framework': 'tensorflow-mkl', 'model': model_name, 'threads': threads})
            return metrics

        except Exception as e:
            logger.error(f"TensorFlow error: {str(e)}")
            return {}

    def _measure_performance(self, run_fn) -> dict:
        # Warmup runs
        for _ in range(self.num_warmup):
            run_fn()

        # Benchmark runs
        latencies = []
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            run_fn()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        return {
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'throughput': 1000 / np.mean(latencies)
        }

    def run_benchmark(self):
        logger.info("Starting benchmark...")
        
        benchmark_functions = {
            'onnx-mkl': self.benchmark_onnx,
            'torch-ipex': self.benchmark_pytorch,
            'tensorflow-mkl': self.benchmark_tensorflow
        }

        thread_configs = [1, 2, 4]

        for model_name in self.models:
            for framework in self.frameworks:
                for threads in thread_configs:
                    try:
                        benchmark_fn = benchmark_functions[framework]
                        result = benchmark_fn(model_name, threads)
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Error running {framework} benchmark: {str(e)}")

        self._save_results()

    def _save_results(self):
        if not self.results:
            logger.error("No results to save")
            return

        df = pd.DataFrame(self.results)
        df.to_csv('benchmark_results.csv', index=False)

        # Plot throughput comparison
        plt.figure(figsize=(12, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            for framework in model_data['framework'].unique():
                fw_data = model_data[model_data['framework'] == framework]
                plt.plot(fw_data['threads'], fw_data['throughput'],
                         marker='o', label=f'{model}-{framework}')

        plt.xlabel('Threads')
        plt.ylabel('Throughput (inf/sec)')
        plt.title('Framework Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('throughput_comparison.png')
        plt.close()

        # Plot latency heatmap
        pivot_table = pd.pivot_table(
            df,
            values='avg_latency',
            index=['model', 'threads'],
            columns='framework',
            aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Latency (ms) by Model and Framework')
        plt.savefig('latency_heatmap.png')
        plt.close()

def main():
    benchmark = BenchmarkFrameworks()
    benchmark.run_benchmark()

if __name__ == '__main__':
    main()
