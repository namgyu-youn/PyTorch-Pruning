import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM


# Add project root and local sparsegpt to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'sparsegpt'))


# Import after path modification
from benchmarks.llama.baseline import LLMBenchmark  # noqa: E402
from datautils import get_wikitext2, get_tokenizer  # noqa: E402
from sparsegpt import SparseGPT  # noqa: E402




def find_layers(module, layers=[nn.Linear], name=''):
    """Recursively find layers of specified types in a module"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res




def sparsegpt_pruning(model, tokenizer, sparsity_ratio=0.5):
    """
    Apply SparseGPT pruning using layer-wise sparse regression

    Reference: https://arxiv.org/abs/2301.00774
    """
    print(f"Starting SparseGPT pruning with {sparsity_ratio:.0%} sparsity...")


    # Get calibration data and setup
    trainenc, _ = get_wikitext2(128, 42, 2048, "", tokenizer)
    model.eval()
    layers = model.model.layers


    # Cache for intermediate activations
    cache = {'i': 0, 'attention_mask': None}


    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError


    # Capture inputs to first transformer layer
    layers[0] = Catcher(layers[0])
    inps = torch.zeros((128, 2048, model.config.hidden_size),
                       dtype=model.dtype, device=model.device)


    with torch.no_grad():
        for batch in trainenc:
            try:
                model(batch.to(model.device))
            except ValueError:
                pass


    layers[0] = layers[0].module  # Restore first layer


    # Apply SparseGPT layer by layer
    for i, layer in enumerate(layers):
        print(f"Pruning layer {i+1}/{len(layers)}")


        subset = find_layers(layer)
        gpts = {name: SparseGPT(subset[name]) for name in subset}


        # Register forward hooks for calibration
        def add_batch(name):
            return lambda _, inp, out: gpts[name].add_batch(inp[0].data, out.data)


        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]


        # Calibration forward pass
        for j in range(128):
            outs = layer(inps[j].unsqueeze(0), attention_mask=cache['attention_mask'])
            if isinstance(outs, tuple):
                outs = outs[0]


        # Remove hooks and apply pruning
        for h in handles:
            h.remove()


        for name in subset:
            gpts[name].fasterprune(sparsity_ratio, prunen=0, prunem=0, blocksize=128, percdamp=.01)
            gpts[name].free()


        # Update inputs for next layer
        for j in range(128):
            outs = layer(inps[j].unsqueeze(0), attention_mask=cache['attention_mask'])
            inps[j] = outs[0] if isinstance(outs, tuple) else outs


    return model


def calculate_actual_sparsity(model):
    """Calculate actual sparsity of pruned model"""
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    return zero_params / total_params


def main():
    benchmark = LLMBenchmark()


    # Run baseline
    print("Running baseline...")
    baseline = benchmark.run_baseline('huggyllama/llama-7b')
    baseline.print_results("LLaMA Baseline")


    # Test SparseGPT with 50% sparsity
    sparsity = 0.5
    print(f"\n{'='*50}")
    print(f"Testing SparseGPT with {sparsity:.0%} target sparsity")
    print('='*50)


    # Load model and tokenizer
    tokenizer = get_tokenizer('huggyllama/llama-7b')
    model = AutoModelForCausalLM.from_pretrained(
        'huggyllama/llama-7b', torch_dtype=torch.float16, device_map='cuda'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Apply SparseGPT pruning
    model = sparsegpt_pruning(model, tokenizer, sparsity)


    # Measure and print results
    result = benchmark.measure_inference(model, tokenizer)
    result.print_results(f"SparseGPT {sparsity:.0%} Sparsity")


    # Additional metrics
    actual_sparsity = calculate_actual_sparsity(model)
    speedup = baseline.latency / result.latency
    print(f"Actual sparsity: {actual_sparsity:.1%}")
    print(f"Speedup: {speedup:.2f}x")


    # Clean up memory
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()