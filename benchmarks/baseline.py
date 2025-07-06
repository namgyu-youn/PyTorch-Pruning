import torch
import time
import torch.profiler
from data import data_loader
from models import load_model, MODEL_CONFIGS

class Benchmark:
    """Post-training pruning benchmark"""

    def __init__(self, device='cuda'):
        self.device = device

    def measure_inference(self, model, test_loader):
        """Measure inference metrics"""
        model.eval()

        # Model statistics
        params = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        # Latency measurement
        x = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(50):  # warmup
                model(x)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(1000):
                model(x)
        torch.cuda.synchronize()
        latency_ms = (time.time() - start)

        # Accuracy measurement
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(self.device))
                correct += (outputs.argmax(1) == labels.to(self.device)).sum().item()
                total += labels.size(0)
        accuracy = 100 * correct / total

        return {
            'params': params,
            'size_mb': size_mb,
            'latency_ms': latency_ms,
            'accuracy': accuracy
        }

    def compare_models(self, original_model, pruned_model, test_loader):
        """Compare original vs pruned model"""
        print("Measuring original model...")
        original_metrics = self.measure_inference(original_model, test_loader)

        print("Measuring pruned model...")
        pruned_metrics = self.measure_inference(pruned_model, test_loader)

        # Calculate improvements
        param_reduction = (1 - pruned_metrics['params'] / original_metrics['params']) * 100
        size_reduction = (1 - pruned_metrics['size_mb'] / original_metrics['size_mb']) * 100
        speedup = original_metrics['latency_ms'] / pruned_metrics['latency_ms']
        accuracy_drop = original_metrics['accuracy'] - pruned_metrics['accuracy']

        return {
            'original': original_metrics,
            'pruned': pruned_metrics,
            'improvements': {
                'param_reduction_pct': param_reduction,
                'size_reduction_pct': size_reduction,
                'speedup_ratio': speedup,
                'accuracy_drop_pct': accuracy_drop
            }
        }

    def run_baseline(self, model_names=None):
        """Run baseline for multiple models"""
        if model_names is None:
            model_names = list(MODEL_CONFIGS.keys())

        test_loader = data_loader()
        results = {}

        for name in model_names:
            print(f"Benchmarking {name}...")
            model = load_model(name, self.device)
            results[name] = self.measure_inference(model, test_loader)

        return results

    def print_results(self, comparison):
        """Print comparison results"""
        orig = comparison['original']
        pruned = comparison['pruned']
        imp = comparison['improvements']

        print("\n=== Pruning Results ===")
        print(f"Parameters: {orig['params']:,} → {pruned['params']:,} ({imp['param_reduction_pct']:.1f}% reduction)")
        print(f"Size: {orig['size_mb']:.1f}MB → {pruned['size_mb']:.1f}MB ({imp['size_reduction_pct']:.1f}% reduction)")
        print(f"Latency: {orig['latency_ms']:.2f}ms → {pruned['latency_ms']:.2f}ms ({imp['speedup_ratio']:.2f}x speedup)")
        print(f"Accuracy: {orig['accuracy']:.1f}% → {pruned['accuracy']:.1f}% ({imp['accuracy_drop_pct']:.1f}% drop)")

if __name__ == "__main__":
    # Run benchmark
    benchmark = Benchmark()
    baseline_results = benchmark.run_baseline(['vit', 'efficientnet', 'swin'])

    print("=== Baseline Results ===")
    for name, metrics in baseline_results.items():
        print(f"{name}: {metrics['params']:,} params, {metrics['latency_ms']:.2f}ms, {metrics['accuracy']:.1f}%")