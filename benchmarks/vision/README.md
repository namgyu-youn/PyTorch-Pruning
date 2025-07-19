# Computer Vision Benchmarks

This section is the benchmark baseline for comparing Pruning After Training (PAT) techniques in inference level. There are experiments for applying pruning techniques at [here](https://github.com/namgyu-youn/PyTorch-Pruning/tree/main/experiments/vision)

> Since the objective is profiling and deeper understanding, we don't adopt too complicated architecture. For example, windows in Swin Transformer (Swin-T) are hard to being profile.

### Environments

- Metrics : Model (parameter) size, latency, and accuracy
- Resources: NVIDIA RTX A2000
- Models (`timm`) :
    - CNN based : ConvNeXt, EfficientNet, MobileNet
    - Transformer based : ViT



# Benchmarks Result

## Original (not pruned)

```bash
# N(params), Latency, Accuracy
c: 196,245,706 params, 17.05ms, 8.6%
e: 17,566,546 params, 12.72ms, 6.0%
v: 303,311,882 params, 31.07ms, 11.4%
m: 31,322,674 params, 8.64ms, 11.8%
```

> Note: Since `timm` models are pre-trained from ImageNet dataset, accuracy is quiet low.