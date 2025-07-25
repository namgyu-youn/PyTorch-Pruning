import torch
import torch.nn as nn
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM
from torchao.sparsity import WandaSparsifier

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from benchmarks.llama.baseline import LLMBenchmark
from sparsegpt.datautils import get_wikitext2, get_tokenizer


def wanda_pruning(model, tokenizer, sparsity_ratio=0.5):
    """
    Apply Wanda pruning using TorchAO

    Wanda (Pruning by Weights and activations) removes weights based on the product
    of weight magnitudes and input activation norms on a per-output basis.

    Reference: https://arxiv.org/abs/2306.11695
    """
    sparse_config = [{"tensor_fqn": f"{name}.weight"}
                    for name, module in model.named_modules()
                    if isinstance(module, nn.Linear)]

    wanda = WandaSparsifier(sparsity_level=sparsity_ratio)
    wanda.prepare(model, sparse_config)

    # Calibration
    trainenc, _ = get_wikitext2(128, 42, 2048, "", tokenizer)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(32, trainenc.input_ids.size(0)), 4):
            model(trainenc.input_ids[i:i+4].to(model.device))

    wanda.step()
    wanda.squash_mask()
    return model


def main():
    benchmark = LLMBenchmark()

    # Baseline
    baseline = benchmark.run_baseline('huggyllama/llama-7b')
    baseline.print_results("Baseline")

    # Wanda pruning
    tokenizer = get_tokenizer('huggyllama/llama-7b')
    model = AutoModelForCausalLM.from_pretrained(
        'huggyllama/llama-7b', torch_dtype=torch.float16, device_map='cuda'
    )

    model = wanda_pruning(model, tokenizer, 0.5)
    wanda_result = benchmark.measure_inference(model, tokenizer)
    # Benchmark results
    wanda_result.print_results("Wanda 50%")

    print(f"\nSpeedup: {baseline.latency/wanda_result.latency:.2f}x")


if __name__ == "__main__":
    main()