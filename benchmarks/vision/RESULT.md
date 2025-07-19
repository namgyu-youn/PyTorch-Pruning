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


## Ln-norm based Pruning (PyTorch)

### Unstructured pruning (threshold=0.3)

```bash
c: 196,245,706 params, 13.81ms, 6.4%
e: 17,566,546 params, 2.23ms, 7.8%
v: 303,311,882 params, 22.43ms, 10.6%
m: 31,322,674 params, 1.16ms, 11.1%
```

### Unstrucutral pruning (threshold=0.5)

```bash
c: 196,245,706 params, 14.18ms, 11.1%
e: 17,566,546 params, 2.23ms, 9.8%
v: 303,311,882 params, 22.28ms, 7.8%
m: 31,322,674 params, 1.16ms, 10.0%
```

### Global structured pruning (threshold=0.3)

```bash
c: 196,245,706 params, 14.18ms, 9.9%
e: 17,566,546 params, 2.24ms, 13.4%
v: 303,311,882 params, 22.45ms, 15.7%
m: 31,322,674 params, 1.16ms, 11.9%
```

### Strucutural pruning (threshold=0.3)

```bash
c: 196,245,706 params, 14.09ms, 9.4%
e: 17,566,546 params, 2.22ms, 10.0%
v: 303,311,882 params, 22.15ms, 10.0%
m: 31,322,674 params, 1.16ms, 10.0%
```

## Depgraph : Towards any structural pruning

### Magnitude Strucutral Pruning (not grouped, threshold=0.3)

```bash
c: 139,359,514 params, 10.42ms, 7.5%
e: 16,322,310 params, 2.23ms, 10.0%
m: 31,322,674 params, 1.17ms, 9.7%
v: 242,727,050 params, 18.86ms, 9.8%
```

### Random Structural Pruning (not grouped, threshold=0.3)

```bash
c: 139,359,514 params, 10.76ms, 6.1%
e: 16,322,310 params, 2.24ms, 9.0%
m: 31,322,674 params, 1.17ms, 7.7%
v: 242,727,050 params, 19.00ms, 9.9%
```

### Group Structural Pruning (threshold=0.3)

```bash
c: 196,245,706 params, 14.26ms, 13.4%
e: 17,566,546 params, 2.25ms, 9.5%
m: 31,322,674 params, 1.18ms, 12.2%
v: 303,311,882 params, 22.56ms, 12.0%
```

### Magnitude Strucutral Pruning (ungrouped, threshold=0.5)

```bash
c: 101,967,178 params, 7.96ms, 9.1%
e: 15,887,090 params, 2.23ms, 9.3%
m: 31,322,674 params, 1.17ms, 9.7%
v: 202,599,434 params, 15.77ms, 9.4%
```

### Rndomg Structural Pruning (ungrouped, threshold=0.5)

```bash
c: 101,967,178 params, 8.10ms, 9.0%
e: 15,887,090 params, 2.23ms, 9.1%
m: 31,322,674 params, 1.17ms, 7.7%
v: 202,599,434 params, 15.81ms, 12.9%
```

### Group Structural Pruning (threshold=0.5)

```bash
c: 196,245,706 params, 14.23ms, 13.4%
e: 17,566,546 params, 2.24ms, 9.5%
m: 31,322,674 params, 1.17ms, 12.2%
v: 303,311,882 params, 22.60ms, 12.0%
```

### Magnitude Strucutral Pruning (ungrouped, threshold=0.7)

```bash
c: 64,141,402 params, 5.97ms, 6.9%
e: 15,247,914 params, 2.23ms, 9.4%
m: 31,322,674 params, 1.18ms, 9.6%
```

### Rndomg Structural Pruning (ungrouped, threshold=0.7)

```bash
c: 64,141,402 params, 5.97ms, 10.0%
e: 15,247,914 params, 2.24ms, 14.9%
m: 31,322,674 params, 1.18ms, 12.5%
```

### Group Structural Pruning (threshold=0.7)

```bash
c: 196,245,706 params, 14.25ms, 6.1%
e: 17,566,546 params, 2.24ms, 10.5%
m: 31,322,674 params, 1.17ms, 10.9%
```

Note: Since `timm` models are pre-trained from ImageNet dataset, accuracy is quiet low.