import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.utils.prune as prune
from benchmarks.baseline import Benchmark
from benchmarks.models import load_model, MODEL_CONFIGS
from benchmarks.data import data_loader

class SimplePruning:
    def __init__(self, device='cuda'):
        self.device = device
        self.benchmark = Benchmark(device)

    def _get_prunable_modules(self, model):
        """Get modules that can be pruned"""
        return [module for module in model.modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))]

    def magnitude_prune(self, model, sparsity):
        """Apply magnitude-based pruning"""
        for module in self._get_prunable_modules(model):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
        return model

    def global_prune(self, model, sparsity):
        """Apply global magnitude pruning"""
        parameters_to_prune = [(module, 'weight')
                              for module in self._get_prunable_modules(model)]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        return model

    def structured_prune(self, model, sparsity):
        """Apply structured pruning"""
        for module in self._get_prunable_modules(model):
            prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
        return model

    def remove_masks(self, model):
        """Remove pruning masks"""
        for module in self._get_prunable_modules(model):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        return model

    def pruning(self, model_names='e', sparsity=0.5, method='magnitude'):
        """Test pruning method on single model or list of models"""
        if isinstance(model_names, str):
            model_names = [model_names]

        results = {}
        test_loader = data_loader()

        prune_methods = {
            'magnitude': self.magnitude_prune,
            'global': self.global_prune,
            'structured': self.structured_prune
        }

        for model_name in model_names:
            print(f"Benchmarking {model_name} with {method} pruning (sparsity: {sparsity})...")

            # Load and prune model
            model = load_model(model_name, self.device)
            model = prune_methods[method](model, sparsity)
            model = self.remove_masks(model)

            # Test pruned model performance
            result = self.benchmark.measure_inference(model, test_loader)
            results[model_name] = result

        return results

if __name__ == "__main__":
    pruning = SimplePruning()

    # Test different sparsity levels
    sparsity_levels = [0.3, 0.5, 0.7, 0.9]

    for sparsity in sparsity_levels:
        for method in ['magnitude', 'global', 'structured']:
            print(f"\n=== {method.capitalize()} Pruning Results (sparsity: {sparsity}) ===")
            results = pruning.pruning(list(MODEL_CONFIGS.keys()), sparsity, method)

            for name, metrics in results.items():
                print(f"{name}: {metrics['params']:,} params, {metrics['latency_ms']:.2f}ms, {metrics['accuracy']:.1f}%")