# LLaMA Benchmarks

This section is the benchmark baseline for comparing Pruning After Training (PAT) techniques in inference level. There are experiments for applying pruning techniques at [here](https://github.com/namgyu-youn/PyTorch-Pruning/tree/main/experiments/llama)

## Evaluation Metrics

For deeper understanding, we uses parameter (model) size, latency, and perplexity.

- Perplexity : Inspired by [SparseGPT](https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py), we uses its metrics using a third-party.

> Note: SparseGPT used original LLaMA-1 structures as the weight. But for the conveience and future experiments, we uses pre-trained LLaMA (`huggyllama/llama-7b`). Therefore, our benchmarks are quiet different.

- Latency : Inspired by [SymblAI](https://symbl.ai/developers/blog/a-guide-to-llm-inference-performance-monitoring/), we uses both Time TO First Token (TTFT) and Time Per Output Token (TPOT) for computing "total generation time"

$$
Total \ generation \ time = TTFT + (TPOT \times N(Out_{t}))
$$


- Prompt : Inspired by [BentoML](https://www.bentoml.com/blog/benchmarking-llm-inference-backends), we uses [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
