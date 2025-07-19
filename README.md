# PyTorch-Pruning

This project aims to **benchmark and profile variable pruning** techniques for language models ([LLaMA](https://huggingface.co/huggyllama/llama-7b)) and computer vision ([timm](https://github.com/huggingface/pytorch-image-models)) models. Therefore, we uses multiple-metrics (accuracy & latency) and profiling ([torch.profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)).

### Main objective

- Add more pruning methodlogy in "[experiments/](https://github.com/namgyu-youn/PyTorch-Pruning/tree/main/experiments)"
- External users should be able to simply prune and understand it deeply.
- Each pruning should run within baselines ([computer vision (timm)](https://github.com/namgyu-youn/PyTorch-Pruning/blob/main/benchmarks/vision/baseline.py) & [language models (LLaMA)](https://github.com/namgyu-youn/PyTorch-Pruning/blob/main/benchmarks/llama/baseline.py)) module for fair comparison.

## User Guide

This projects recommend `uv` for pypi (python packaging index).

```bash
# Setup virtual environment
uv .venv
source .venv/bin/activate
uv pip install ".[dev]"

# Run any scripts for pruning
uv run experiments/pytorch_pruning.py
```

## Contribution Guide ([Roadmap](https://github.com/namgyu-youn/PyTorch-Pruning/issues/1))

If you have any ideas for this project, please feel free to open Issue or submit PR. Since our main objective is supporting more pruning, we will always welcome new features.

## References

![pruning_taxonomy](images/pruning_taxonomy.png)

[[1](https://hanlab.mit.edu/courses/2024-fall-65940)] TinyML and Efficient Deep Learning Computing (MIT-6.5940)

[[2](https://arxiv.org/abs/2308.06767)] A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations (IEEE'24)

[3] Pruning Deep Neural Networks from a Sparsity Perspective (ICLR'23, [arXiv](https://arxiv.org/abs/2302.05601))

[[4](https://icml.cc/virtual/2024/oral/35453)] APT: Adaptive Pruning and Tuning Pretrained LLM for Efficient Training and Inference (ICML'24)

[[5](https://arxiv.org/abs/2312.11983)] Fluctuation-based Adaptive Structured Pruning for Large Language Models (AAAI'24)

[[6](https://arxiv.org/abs/2407.04616)] Isomorphic Pruning for Vision Models (ECCV'24)

[[7]((https://arxiv.org/abs/2111.13445))] How Well Do Sparse ImageNet Models Transfer? (CVPR'22)

[[8](https://arxiv.org/abs/2306.11695)] A Simple and Effective Pruning Approach for Large Language Models (ICLR'24)

[[9](https://blog.squeezebits.com/how-to-quantize-transformerbased-model-for-tensorrt-deployment-55802)] How to Quantize Transformer-based model for TensorRT Deployment)

[[10](https://arxiv.org/abs/2302.02596)] Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers (ICLR'25)

[[11](https://arxiv.org/abs/2301.12900)] Depgraph: Towards any structural pruning (CVPR'23)

[[12](https://paperswithcode.com/task/network-pruning)] Papers with Code : Pruning Benchmark

[[13](https://github.com/VITA-Group/SMC-Bench)] Sparsity May Cry Benchmark (SMC-Bench, ICLR'23)