import torch
import time
import sys
import random
import numpy as np
import requests
import json
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from contextlib import contextmanager

# Use SparseGPT's data utils (third-party library)
sys.path.append('third_party/sparsegpt')
from datautils import get_wikitext2, get_tokenizer

@contextmanager
def inference_mode(model):
    """Context manager for safe inference"""
    model.eval()
    with torch.no_grad():
        yield
    torch.cuda.synchronize()

@dataclass
class BenchmarkResult:
    params: int
    ttft: float
    tpot: float
    latency: float
    perplexity: float

class LLMBenchmark:
    """LLM Benchmark class for measuring latency and perplexity of causal language models."""
    def __init__(self, device='cuda', seed=42):
        self.device = device
        self.seed = seed
        self._set_seed(seed)

    def _set_seed(self, seed):
        """Set all relevant random seeds"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def _load_dataset(self):
        """Download Databricks Dolly 15k dataset"""
        DATASET_PATH = Path("benchmarks/llama/databricks-dolly-15k.jsonl")
        if not DATASET_PATH.exists():
            print("Downloading Databricks Dolly 15k dataset...")
            DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            DATASET_PATH.write_bytes(requests.get("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl").content)

        return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]

    def _get_prompt_set(self, min_length=100, max_length=500):
        dataset = self._load_dataset()
        return [item["instruction"] for item in dataset
                if min_length <= len(item.get("instruction", "").split()) <= max_length][:10]

    def _calculate_perplexity(self, model, tokenizer, nsamples=128, seqlen=2048):
        """Calculate perplexity using SparseGPT's evaluation method"""
        print("Evaluating perplexity...")
        _, testenc = get_wikitext2(nsamples, self.seed, seqlen, "", tokenizer)

        testenc = testenc.input_ids.to(self.device)
        nsamples_test = testenc.numel() // seqlen

        nlls = []  # Negative log likelihoods for perplexity calculation

        with inference_mode(model):
            for i in range(nsamples_test):
                batch = testenc[:, i * seqlen : (i + 1) * seqlen]
                outputs = model(batch).logits

                # Shift logits/labels for next-token prediction
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:]

                loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss.float() * seqlen)

        # Perplexity (PPL) score
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples_test * seqlen))
        return ppl.item()

    def measure_inference(self, model, tokenizer):
        """Measure latency (TTFT & TPOT) and perplexity for the given model"""
        params = sum(p.numel() for p in model.parameters())
        prompts = self._get_prompt_set()

        prompt = prompts[0]
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with inference_mode(model):
            # Measure TTFT (Time to First Token)
            start = time.time()
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
            ttft = time.time() - start

            # Measure TPOT (Time Per Output Token)
            start = time.time()
            model.generate(**inputs, max_new_tokens=50 - 1, do_sample=False)
            tpot = (time.time() - start) / (50 - 1)

        latency = ttft + tpot * (50 - 1)
        perplexity = self._calculate_perplexity(model, tokenizer)

        return BenchmarkResult(params, ttft, tpot, latency, perplexity)

    def run_baseline(self, model_id='huggyllama/llama-7b'):
        print(f"Benchmarking {model_id}...")

        # Load model and tokenizer
        tokenizer = get_tokenizer(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        result = self.measure_inference(model, tokenizer)

        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache()
        return result

if __name__ == "__main__":
    benchmark = LLMBenchmark()
    result = benchmark.run_baseline()

    if result:
        print("\n=== LLaMA Baseline ===")
        print(f"Params: {result.params/1e9:.1f}B")
        print(f"TTFT: {result.ttft:.3f}s")
        print(f"TPOT: {result.tpot:.3f}s")
        print(f"Total Latency: {result.latency:.3f}s")
        print(f"Perplexity: {result.perplexity:.2f}")