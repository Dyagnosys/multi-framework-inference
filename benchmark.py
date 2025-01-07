import onnxruntime as ort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import torch
import json
import ncnn
import MNN
import tvm
from tvm import relay
from pathlib import Path
from typing import Dict, List, Tuple
from llama_cpp import Llama

class ComprehensiveBenchmark:
    def __init__(self):
        self.results = []
        self.system_info = self._get_system_info()
        self.frameworks = ['onnx', 'ncnn', 'mnn', 'tvm', 'llama.cpp']
        
    def _get_system_info(self) -> Dict[str, any]:
        return {
            'cpu_model': self._get_cpu_model(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'onnxruntime_version': ort.__version__,
        }
    
    def _get_cpu_model(self) -> str:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except:
            return 'Unknown CPU'
        return 'Unknown CPU'

    def create_test_models(self):
        """Create test models for different architectures"""
        models = {}
        
        # MLP Model
        class MLPModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # CNN Model
        class CNNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
                self.pool = torch.nn.MaxPool2d(2)
                self.fc = torch.nn.Linear(32 * 6 * 6, 10)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(-1, 32 * 6 * 6)
                x = self.fc(x)
                return x

        # Transformer Model (simplified)
        class TransformerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 64)
                self.transformer = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=64, nhead=4, dim_feedforward=256
                    ),
                    num_layers=2
                )
                self.fc = torch.nn.Linear(64, 10)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = self.fc(x.mean(dim=1))
                return x

        model_configs = {
            'mlp': (MLPModel(), (1, 128)),
            'cnn': (CNNModel(), (1, 3, 24, 24)),
            'transformer': (TransformerModel(), (1, 16))  # Sequence length 16
        }

        for name, (model, input_shape) in model_configs.items():
            model.eval()
            dummy_input = torch.randn(input_shape) if 'transformer' not in name \
                else torch.randint(0, 1000, input_shape)
            path = f'test_{name}.onnx'
            
            torch.onnx.export(model, dummy_input, path,
                            verbose=False,
                            input_names=['input'],
                            output_names=['output'],
                            opset_version=12)
            
            models[name] = {
                'path': path,
                'input_shape': input_shape,
                'input_data': dummy_input.numpy().astype(np.float32)
            }
        
        return models

    def benchmark_onnx(self, model_info: Dict, num_threads: int,
                      num_iterations: int = 100) -> Dict:
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_info['path'],
            sess_options=options,
            providers=['CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        latencies = self._run_inference(
            lambda: session.run(None, {input_name: model_info['input_data']}),
            num_iterations
        )
        
        return self._calculate_metrics('ONNX', num_threads, latencies)

    def benchmark_ncnn(self, model_info: Dict, num_threads: int,
                      num_iterations: int = 100) -> Dict:
        net = ncnn.Net()
        net.load_param(f"{model_info['path']}.param")
        net.load_model(f"{model_info['path']}.bin")
        
        extractor = net.create_extractor()
        extractor.set_num_threads(num_threads)
        
        latencies = self._run_inference(
            lambda: extractor.extract("output", model_info['input_data']),
            num_iterations
        )
        
        return self._calculate_metrics('NCNN', num_threads, latencies)

    def benchmark_mnn(self, model_info: Dict, num_threads: int,
                     num_iterations: int = 100) -> Dict:
        interpreter = MNN.Interpreter(model_info['path'])
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        
        latencies = self._run_inference(
            lambda: interpreter.runSession(session),
            num_iterations
        )
        
        return self._calculate_metrics('MNN', num_threads, latencies)

    def benchmark_tvm(self, model_info: Dict, num_threads: int,
                     num_iterations: int = 100) -> Dict:
        target = "llvm -mcpu=generic"
        
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(model_info['path'], target=target)
        
        module = tvm.contrib.graph_executor.GraphModule(
            lib["default"](tvm.cpu())
        )
        
        latencies = self._run_inference(
            lambda: module.run(),
            num_iterations
        )
        
        return self._calculate_metrics('TVM', num_threads, latencies)

    def benchmark_llamacpp(self, model_info: Dict, num_threads: int,
                         num_iterations: int = 100) -> Dict:
        llm = Llama(
            model_path=model_info['path'],
            n_threads=num_threads
        )
        
        latencies = self._run_inference(
            lambda: llm.eval(model_info['input_data']),
            num_iterations
        )
        
        return self._calculate_metrics('llama.cpp', num_threads, latencies)

    def _run_inference(self, inference_fn, num_iterations: int) -> List[float]:
        # Warmup
        for _ in range(10):
            inference_fn()
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_time = time.time()
            inference_fn()
            latencies.append((time.time() - start_time) * 1000)  # ms
        
        return latencies

    def _calculate_metrics(self, framework: str, num_threads: int,
                         latencies: List[float]) -> Dict:
        latencies = np.array(latencies)
        metrics = {
            'framework': framework,
            'threads': num_threads,
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'throughput': len(latencies) / np.sum(latencies) * 1000
        }
        
        print(f"\n{framework} with {num_threads} threads:")
        print(f"Average latency: {metrics['avg_latency']:.2f} ms")
        print(f"P95 latency: {metrics['p95_latency']:.2f} ms")
        print(f"Throughput: {metrics['throughput']:.2f} inf/sec")
        
        return metrics

    def run_full_benchmark(self):
        models = self.create_test_models()
        thread_configs = [1, 2, self.system_info['physical_cores']]
        
        for model_name, model_info in models.items():
            print(f"\nBenchmarking {model_name.upper()} model...")
            
            for threads in thread_configs:
                # Run benchmarks for each framework
                benchmark_fns = {
                    'onnx': self.benchmark_onnx,
                    'ncnn': self.benchmark_ncnn,
                    'mnn': self.benchmark_mnn,
                    'tvm': self.benchmark_tvm
                }
                
                # Add llama.cpp only for transformer models
                if model_name == 'transformer':
                    benchmark_fns['llama.cpp'] = self.benchmark_llamacpp
                
                for framework, benchmark_fn in benchmark_fns.items():
                    try:
                        result = benchmark_fn(model_info, threads)
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        print(f"Error benchmarking {framework}: {str(e)}")

    def save_results(self, output_dir: str = '.'):
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
        
        # Save system info
        with open(f'{output_dir}/system_info.json', 'w') as f:
            json.dump(self.system_info, f, indent=2)
        
        # Create plots
        self._create_plots(df, output_dir)
    
    def _create_plots(self, df: pd.DataFrame, output_dir: str):
        """Create detailed visualization of benchmark results"""
        # Create plots for each model type
        model_types = ['mlp', 'cnn', 'transformer']
        metrics = ['avg_latency', 'p95_latency', 'throughput']
        
        for model_type in model_types:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 20))
            fig.suptitle(f'{model_type.upper()} Model Performance Comparison')
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                model_data = df[df['model_type'] == model_type]
                
                # Create grouped bar plot
                frameworks = model_data['framework'].unique()
                x = np.arange(len(frameworks))
                width = 0.2
                
                for i, threads in enumerate(model_data['threads'].unique()):
                    thread_data = model_data[model_data['threads'] == threads]
                    values = [thread_data[thread_data['framework'] == fw][metric].iloc[0] 
                             for fw in frameworks]
                    
                    ax.bar(x + i*width, values, width, 
                          label=f'{threads} threads')
                
                ax.set_xlabel('Framework')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} by Framework and Thread Count')
                ax.set_xticks(x + width)
                ax.set_xticklabels(frameworks, rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{model_type}_benchmark_results.png')
            plt.close()
        
        # Create summary heatmap
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(
            values='throughput',
            index=['model_type', 'threads'],
            columns='framework',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Throughput Comparison (inferences/second)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/throughput_heatmap.png')
        plt.close()

def main():
    print("Starting comprehensive framework benchmark...")
    benchmark = ComprehensiveBenchmark()
    
    # Print system information
    print("\nSystem Information:")
    for key, value in benchmark.system_info.items():
        print(f"{key}: {value}")
    
    # Run benchmarks
    benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results()
    print("\nBenchmark complete! Results saved to current directory.")

if __name__ == "__main__":
    main()