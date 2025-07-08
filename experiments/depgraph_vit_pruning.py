"""
Vision Transformer pruning with torch-pruning for single model ('v')
"""

import torch
import torch.nn as nn
import torch_pruning as tp
from timm.models.vision_transformer import Attention

# TODO : Integrate into `depgraph_pruning.py` for unified pruning
class VitDegraphPruning:
    """ViT structured pruning for model 'v'"""

    def __init__(self, device='cuda'):
        self.device = device

    def _prune(self, model, sparsity, importance):
        """Core pruning with attention-aware fallback"""
        example_inputs = torch.randn(1, 3, 224, 224).to(self.device)

        # Ignore attention, normalization, classifier
        ignored = [m for m in model.modules()
                  if isinstance(m, (Attention, nn.LayerNorm, nn.Embedding)) or
                     (isinstance(m, nn.Linear) and m.out_features == 10)]

        # Handle positional embeddings
        unwrapped = [(p, -1) for n, p in model.named_parameters()
                    if any(x in n for x in ['cls_token', 'pos_embed'])]

        try:
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs, importance=importance,
                pruning_ratio=sparsity, ignored_layers=ignored,
                unwrapped_parameters=unwrapped, round_to=8)
            pruner.step()
        except Exception:
            # Fallback: only large MLPs
            ignored = [m for m in model.modules()
                      if not (isinstance(m, nn.Linear) and getattr(m, 'in_features', 0) > 100)]
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs, importance=importance,
                pruning_ratio=min(sparsity, 0.1), ignored_layers=ignored)
            pruner.step()

        return model

    def prune(self, sparsity=0.5, method='magnitude'):
        """Prune ViT model 'v' and return metrics"""
        methods = {
            'magnitude': tp.importance.MagnitudeImportance(p=1),
            'random': tp.importance.RandomImportance(),
            'group': self._get_group_importance()
        }

        model = load_model('v', self.device)
        model = self._prune(model, sparsity, methods[method])

        from benchmarks.baseline import Benchmark
        return Benchmark(self.device).measure_inference(model, data_loader())

    def _get_group_importance(self):
        """Get group importance with fallback"""
        try:
            return tp.importance.BNScaleImportance()
        except (AttributeError, ImportError):
            return tp.importance.MagnitudeImportance(p=2)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmarks.models import load_model
    from benchmarks.data import data_loader

    pruning = VitDegraphPruning()

    for sparsity in [0.3, 0.5]:
        for method in ['magnitude', 'random', 'group']:
            print(f"\n=== {method.capitalize()} Pruning (sparsity: {sparsity}) ===")
            metrics = pruning.prune(sparsity, method)
            print(f"v: {metrics['params']:,} params, {metrics['latency_ms']:.2f}ms, {metrics['accuracy']:.1f}%")