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


### Ln-norm based Pruning (PyTorch)

Unstructured pruning (threshold=0.3)

```bash
c: 196,245,706 params, 13.81ms, 6.4%
e: 17,566,546 params, 2.23ms, 7.8%
s: 195,010,846 params, 16.50ms, 11.6%
v: 303,311,882 params, 22.43ms, 10.6%
m: 31,322,674 params, 1.16ms, 11.1%
r: 58,183,530 params, 3.45ms, 10.7%
x: 86,762,826 params, 5.49ms, 11.1%
```

Unstrucutral pruning (threshold=0.5)

```bash
c: 196,245,706 params, 14.18ms, 11.1%
e: 17,566,546 params, 2.23ms, 9.8%
s: 195,010,846 params, 16.48ms, 11.7%
v: 303,311,882 params, 22.28ms, 7.8%
m: 31,322,674 params, 1.16ms, 10.0%
r: 58,183,530 params, 3.43ms, 6.4%
x: 86,762,826 params, 5.52ms, 9.5%
```

Global structured pruning (threshold=0.3)

```bash
c: 196,245,706 params, 14.18ms, 9.9%
e: 17,566,546 params, 2.24ms, 13.4%
s: 195,010,846 params, 16.57ms, 11.2%
v: 303,311,882 params, 22.45ms, 15.7%
m: 31,322,674 params, 1.16ms, 11.9%
r: 58,183,530 params, 3.45ms, 11.7%
x: 86,762,826 params, 5.50ms, 7.5%
```

Strucutural pruning (threshold=0.3)

```bash
c: 196,245,706 params, 14.09ms, 9.4%
e: 17,566,546 params, 2.22ms, 10.0%
s: 195,010,846 params, 16.41ms, 10.0%
v: 303,311,882 params, 22.15ms, 10.0%
m: 31,322,674 params, 1.16ms, 10.0%
r: 58,183,530 params, 3.42ms, 9.0%
x: 86,762,826 params, 5.50ms, 8.7%
```

Note: Since `timm` models are pre-trained from ImageNet dataset, accuracy is quiet low.