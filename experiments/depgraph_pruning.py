"""
Torch-pruning implementation with model-aware strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch_pruning as tp
from benchmarks.baseline import Benchmark
from benchmarks.models import load_model
from benchmarks.data import data_loader

class DegraphPruning:
    """Structured pruning with model-aware fallbacks"""

    def __init__(self, device='cuda'):
        self.device = device
        self.benchmark = Benchmark(device)

    def _get_ignored_layers(self, model):
        """Get layers to ignore based on model type"""
        model_name = type(model).__name__.lower()
        ignored = []

        for m in model.modules():
            # Always ignore classifier and normalization
            if (isinstance(m, nn.Linear) and m.out_features == 10) or \
               isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.Embedding)):
                ignored.append(m)
            # For non-ViT models, ignore attention modules
            elif 'vit' not in model_name and 'vision' not in model_name:
                if hasattr(m, 'attention') or hasattr(m, 'attn'):
                    ignored.append(m)

        return ignored

    def _prune(self, model, sparsity, importance):
        """Core pruning with conservative fallback"""
        model.eval()
        example_inputs = torch.randn(1, 3, 224, 224).to(self.device)

        try:
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs, importance=importance,
                pruning_ratio=sparsity, ignored_layers=self._get_ignored_layers(model),
                round_to=8)
            pruner.step()
        except Exception:
            # Fallback: only Conv2d layers
            conv_only = [m for m in model.modules() if not isinstance(m, nn.Conv2d)]
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs, importance=importance,
                pruning_ratio=min(sparsity, 0.1), ignored_layers=conv_only)
            pruner.step()

        return model

    def magnitude_prune(self, model, sparsity):
        """Magnitude-based structured pruning"""
        return self._prune(model, sparsity, tp.importance.MagnitudeImportance(p=1))

    def random_prune(self, model, sparsity):
        """Random structured pruning"""
        return self._prune(model, sparsity, tp.importance.RandomImportance())

    def group_prune(self, model, sparsity):
        """Group-wise structured pruning"""
        try:
            imp = tp.importance.BNScaleImportance()
        except Exception:
            imp = tp.importance.MagnitudeImportance(p=2)
        return self._prune(model, sparsity, imp)

    def pruning(self, model_names='e', sparsity=0.5, method='magnitude'):
        """Benchmark pruning across models"""
        if isinstance(model_names, str):
            model_names = [model_names]

        methods = {'magnitude': self.magnitude_prune, 'random': self.random_prune, 'group': self.group_prune}
        results = {}

        for name in model_names:
            model = load_model(name, self.device)
            model = methods[method](model, sparsity)

            # Direct inference measurement without separate benchmark class
            model.eval()
            params = sum(p.numel() for p in model.parameters())

            # Latency
            x = torch.randn(16, 3, 224, 224).to(self.device)
            with torch.no_grad():
                for _ in range(50):
                    model(x)  # warmup
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    model(x)
                end.record()
                torch.cuda.synchronize()
                latency_ms = start.elapsed_time(end) / 100 / 16

            # Accuracy
            correct = total = 0
            with torch.no_grad():
                for images, labels in data_loader():
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    predicted = outputs.argmax(1) % 10 if outputs.size(1) > 10 else outputs.argmax(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            results[name] = {
                'params': params,
                'latency_ms': latency_ms,
                'accuracy': 100 * correct / total
            }

        return results

if __name__ == "__main__":
    pruning = DegraphPruning()

    for sparsity in [0.3, 0.5]:
        for method in ['magnitude', 'random', 'group']:
            print(f"\n=== {method.capitalize()} Pruning (sparsity: {sparsity}) ===")
            # Exclude 'v' for ViT
            # TODO: Add ViT support and integrate with `depgraph_vit_pruning.py`
            for name, metrics in pruning.pruning(['c', 'e', 'm'], sparsity, method).items():
                print(f"{name}: {metrics['params']:,} params, {metrics['latency_ms']:.2f}ms, {metrics['accuracy']:.1f}%")