import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import seaborn as sns
from dataclasses import dataclass
import onnxruntime as ort
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    input_shape: tuple
    use_int_input: bool = False
    
class BenchmarkException(Exception):
    pass

class ComprehensiveBenchmark:
    def __init__(self, num_warmup: int = 10, num_iterations: int = 100):
        self.results = []
        self.system_info = self._get_system_info()
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.frameworks = ['onnx', 'pytorch_torchscript', 'tensorflow_xla']
        self.torchscript_models = {}
        self._initialize_frameworks()

    def _initialize_frameworks(self):
        # ONNX Runtime optimization
        self.ort_options = ort.SessionOptions()
        self.ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_options.enable_cpu_mem_arena = True
        
        # Thread optimizations
        num_cores = psutil.cpu_count(logical=False)
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)

    def _get_system_info(self) -> Dict[str, any]:
        try:
            return {
                'cpu_model': self._get_cpu_model(),
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
                'onnxruntime_version': ort.__version__,
                'tensorflow_version': tf.__version__,
                'pytorch_version': torch.__version__
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def _get_cpu_model(self) -> str:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except Exception:
            return 'Unknown CPU'
        return 'Unknown CPU'

    def create_test_models(self) -> Dict[str, Dict]:
        models = {}
        model_configs = [
            ModelConfig('mlp', (1, 128)),
            ModelConfig('cnn', (1, 3, 24, 24)),
            ModelConfig('transformer', (1, 16), use_int_input=True)
        ]
        
        for config in model_configs:
            model = self._create_model(config)
            input_data = self._generate_input_data(config)
            path = self._export_to_onnx(model, input_data, config)
            self._create_torchscript(model, input_data, config)
            
            models[config.name] = {
                'path': path,
                'input_shape': config.input_shape,
                'input_data': input_data.numpy()
            }
        
        return models

    def _create_model(self, config: ModelConfig) -> torch.nn.Module:
        if config.name == 'mlp':
            return torch.nn.Sequential(
                torch.nn.Linear(128, 512),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(256),
                torch.nn.Linear(256, 128)
            )
        elif config.name == 'cnn':
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 6 * 6, 10)
            )
        elif config.name == 'transformer':
            return torch.nn.Sequential(
                torch.nn.Embedding(1000, 128),
                torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=128, 
                        nhead=8,
                        dim_feedforward=512,
                        dropout=0.1,
                        activation='gelu'
                    ),
                    num_layers=4
                ),
                torch.nn.LayerNorm(128),
                torch.nn.Linear(128, 10)
            )
        else:
            raise ValueError(f"Unknown model type: {config.name}")

    def _generate_input_data(self, config: ModelConfig) -> torch.Tensor:
        if config.use_int_input:
            return torch.randint(0, 1000, config.input_shape)
        return torch.randn(config.input_shape)

    def _export_to_onnx(self, model: torch.nn.Module, input_data: torch.Tensor, 
                       config: ModelConfig) -> str:
        path = f'test_{config.name}.onnx'
        try:
            torch.onnx.export(
                model, input_data, path,
                input_names=['input'], 
                output_names=['output'],
                opset_version=13,
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
        except Exception as e:
            logger.error(f"ONNX export failed for {config.name}: {str(e)}")
            raise
        return path

    def _create_torchscript(self, model: torch.nn.Module, input_data: torch.Tensor,
                           config: ModelConfig):
        model.eval()
        self.torchscript_models[config.name] = torch.jit.trace(model, input_data)

    def benchmark_onnx(self, model_info: Dict, num_threads: int) -> Dict:
        try:
            self.ort_options.intra_op_num_threads = num_threads
            session = ort.InferenceSession(
                model_info['path'],
                sess_options=self.ort_options,
                providers=['CPUExecutionProvider']
            )
            input_name = session.get_inputs()[0].name
            latencies = self._run_inference(
                lambda: session.run(None, {input_name: model_info['input_data']}),
                model_info['input_shape'][0]
            )
            return self._calculate_metrics('ONNX', num_threads, latencies)
        except Exception as e:
            logger.error(f"ONNX benchmark error: {e}")
            return {}

    def benchmark_pytorch_torchscript(self, model_info: Dict, num_threads: int) -> Dict:
        try:
            torch.set_num_threads(num_threads)
            model_type = Path(model_info['path']).stem.split('_')[1]
            model = self.torchscript_models.get(model_type)
            
            if model is None:
                raise BenchmarkException(f"No TorchScript model found for {model_type}")
                
            input_data = torch.from_numpy(model_info['input_data'])
            latencies = self._run_inference(
                lambda: model(input_data),
                model_info['input_shape'][0]
            )
            return self._calculate_metrics('PyTorch TorchScript', num_threads, latencies)
        except Exception as e:
            logger.error(f"PyTorch benchmark error: {e}")
            return {}

    def benchmark_tensorflow_xla(self, model_info: Dict, num_threads: int) -> Dict:
        logger.warning("TensorFlow XLA benchmarking not implemented")
        return {}

    def _run_inference(self, inference_fn, batch_size: int) -> List[float]:
        # Warmup
        for _ in range(self.num_warmup):
            inference_fn()
            
        latencies = []
        total_samples = 0
        
        for _ in range(self.num_iterations):
            start_time = time.time()
            inference_fn()
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            total_samples += batch_size
            
        return latencies

    def _calculate_metrics(self, framework: str, num_threads: int, 
                         latencies: List[float]) -> Dict:
        if not latencies:
            return {}
            
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
        
        logger.info(f"\n{framework} with {num_threads} threads:")
        logger.info(f"Average latency: {metrics['avg_latency']:.2f} ms")
        logger.info(f"P95 latency: {metrics['p95_latency']:.2f} ms")
        logger.info(f"Throughput: {metrics['throughput']:.2f} inf/sec")
        
        return metrics

    def run_full_benchmark(self):
        logger.info("Starting comprehensive framework benchmark...")
        logger.info("\nSystem Information:")
        for key, value in self.system_info.items():
            logger.info(f"{key}: {value}")

        try:
            models = self.create_test_models()
            thread_configs = [1, 2, self.system_info['physical_cores']]
            
            benchmark_functions = {
                'onnx': self.benchmark_onnx,
                'tensorflow_xla': self.benchmark_tensorflow_xla,
                'pytorch_torchscript': self.benchmark_pytorch_torchscript
            }
            
            with ThreadPoolExecutor(max_workers=len(self.frameworks)) as executor:
                for model_name, model_info in models.items():
                    logger.info(f"\nBenchmarking {model_name.upper()} model...")
                    for threads in thread_configs:
                        futures = []
                        for fw in self.frameworks:
                            benchmark_fn = benchmark_functions.get(fw)
                            if benchmark_fn:
                                futures.append(
                                    executor.submit(benchmark_fn, model_info, threads)
                                )
                        
                        for future in futures:
                            try:
                                result = future.result()
                                if result:
                                    result['model_type'] = model_name
                                    self.results.append(result)
                            except Exception as e:
                                logger.error(f"Benchmark error: {e}")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

        logger.info("\nBenchmark complete!")

    def save_results(self, output_dir: str = '.'):
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
        
        # Save system info
        with open(f'{output_dir}/system_info.json', 'w') as f:
            json.dump(self.system_info, f, indent=2)

        # Generate plots
        self._create_plots(df, output_dir)
        logger.info(f"Results saved to {output_dir}")

    def _create_plots(self, df: pd.DataFrame, output_dir: str):
        model_types = df['model_type'].unique()
        metrics = ['avg_latency', 'p95_latency', 'throughput']
        
        # Per-model performance plots
        for model_type in model_types:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 20))
            fig.suptitle(f'{model_type.upper()} Model Performance Comparison')
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                model_data = df[df['model_type'] == model_type]
                frameworks = model_data['framework'].unique()
                x = np.arange(len(frameworks))
                width = 0.2
                
                for i, threads in enumerate(sorted(model_data['threads'].unique())):
                    thread_data = model_data[model_data['threads'] == threads]
                    values = []
                    for fw in frameworks:
                        fw_data = thread_data[thread_data['framework'] == fw]
                        values.append(fw_data[metric].iloc[0] if not fw_data.empty else np.nan)
                    ax.bar(x + i*width, values, width, label=f'{threads} threads')
                
                ax.set_xlabel('Framework')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} by Framework and Thread Count')
                ax.set_xticks(x + width)
                ax.set_xticklabels(frameworks, rotation=45)
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{output_dir}/{model_type}_benchmark_results.png')
            plt.close()
        
        # Throughput heatmap
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
    benchmark = ComprehensiveBenchmark()
    benchmark.run_full_benchmark()
    benchmark.save_results()

if __name__ == "__main__":
    main()