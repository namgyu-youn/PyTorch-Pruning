"""
PyTorch native pruning: magnitude, global, and structured methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.utils.prune as prune
from benchmarks.vision.baseline import Benchmark
from benchmarks.vision.models import load_model, MODEL_CONFIGS
from benchmarks.vision.data import data_loader

class PytorchPruning:
    """Native PyTorch pruning for Conv2d/Linear layers"""

    def __init__(self, device='cuda'):
        self.device = device
        self.benchmark = Benchmark(device)

    def _get_modules(self, model):
        """Get prunable Conv2d/Linear modules"""
        return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    def magnitude_prune(self, model, sparsity):
        """L1 magnitude-based unstructured pruning"""
        for m in self._get_modules(model):
            prune.l1_unstructured(m, name='weight', amount=sparsity)
        return model

    def global_prune(self, model, sparsity):
        """Global magnitude pruning across all layers"""
        params = [(m, 'weight') for m in self._get_modules(model)]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
        return model

    def structured_prune(self, model, sparsity):
        """Structured pruning - removes entire filters/neurons"""
        for m in self._get_modules(model):
            prune.ln_structured(m, name='weight', amount=sparsity, n=2, dim=0)
        return model

    def _remove_masks(self, model):
        """Make pruning permanent by removing masks"""
        for m in self._get_modules(model):
            if hasattr(m, 'weight_mask'):
                prune.remove(m, 'weight')
        return model

    def pruning(self, sparsity=0.5, method='magnitude'):
        """Benchmark pruning across all models"""
        methods = {'magnitude': self.magnitude_prune, 'global': self.global_prune, 'structured': self.structured_prune}
        results = {}

        for name in MODEL_CONFIGS.keys():
            try:
                model = load_model(name, self.device)
                model = methods[method](model, sparsity)
                model = self._remove_masks(model)
                results[name] = self.benchmark.measure_inference(model, data_loader())
            except Exception as e:
                print(f"Error: {name} - {e}")

        return results

if __name__ == "__main__":
    pruning = PytorchPruning()

    for sparsity in [0.3, 0.5, 0.7]:
        for method in ['magnitude', 'global', 'structured']:
            print(f"\n=== {method.capitalize()} Pruning (sparsity: {sparsity}) ===")
            for name, metrics in pruning.pruning(sparsity, method).items():
                print(f"{name}: {metrics['params']:,} params, {metrics['latency_ms']:.2f}ms, {metrics['accuracy']:.1f}%")