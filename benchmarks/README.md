# Benchmarks

It aims to compare Pruning After Training (PAT) techniques in inference level.

- Metrics : Model (parameter) size, latency, and accuracy
- Resources: NVIDIA RTX A2000
- Models (timm) :
    - CNN based : ResNet, ResNeXt, ConvNeXt, EfficientNet, MobileNet
    - Transformer based : ViT, Swin-T


## Benchmarks Result
### Original (not pruned)

```bash
# N(params), Latency, Accuracy
c: 196,245,706 params, 17.05ms, 8.6%
e: 17,566,546 params, 12.72ms, 6.0%
s: 195,010,846 params, 20.51ms, 8.3%
v: 303,311,882 params, 31.07ms, 11.4%
m: 31,322,674 params, 8.64ms, 11.8%
r: 58,183,530 params, 14.55ms, 12.5%
x: 86,762,826 params, 9.61ms, 14.0%
```

![result_org](images/benchmark_org.png)


Note: Since `timm` models are pre-trained from ImageNet dataset, accuracy is quiet low.