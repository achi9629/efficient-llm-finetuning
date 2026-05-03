import os
import csv
import json
import torch
import argparse
import torch.nn as nn
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..evaluation import compute_rouge
from ..utils import load_config, run_inference, load_model_and_tokenizer

def get_quantizable_layers(model: AutoModelForCausalLM
        ) -> dict[int, dict[str, tuple[str, nn.Module]]]:
    
    """
    Description:
        Extract all quantizable Linear layers from a causal LM, grouped by transformer block index.

        Walks model.named_modules() and collects nn.Linear layers under "model.layers.*",
        organizing them into a nested dict keyed by block index (str) then module name (str).
        Skips lm_head and embed_tokens implicitly (they don't match "model.layers").

    Args:
        model: HuggingFace causal LM following the model.layers.<id>.<submodule> naming
            convention (e.g., Mistral, Llama).

    Returns:
        Nested dict: layer_id (str) -> module_name (str) -> (fully_qualified_name, nn.Module).
        Example::
            {
                '0': {
                    'q_proj': ('model.layers.0.self_attn.q_proj', Linear(...)),
                    'k_proj': ('model.layers.0.self_attn.k_proj', Linear(...)),
                    'gate_proj': ('model.layers.0.mlp.gate_proj', Linear(...)),
                    ...
                },
                '1': { ... },
            }
    """
    
    module_dict  = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "model.layers" in name:
            layer_id = name.split('.')[2]
            layer_name = name.split('.')[-1]
            if layer_id not in module_dict:
                module_dict[layer_id] = {}
            module_dict[layer_id][layer_name] = (name, module)
    
    return module_dict

def quantize_dequantize_layer(module: nn.Linear, 
                              bits: int = 4
        ) -> None:
    
    """
    Description:
        Simulate INT quantization on a single Linear layer via in-place weight round-trip.

        Applies symmetric per-channel (per-output-row) quantization: computes scale from
        max(|W|) per row, rounds weights to the nearest integer in [-2^(bits-1), 2^(bits-1)-1],
        then multiplies back by scale. The resulting weights remain in the original dtype
        (e.g., float16) but carry the rounding error of N-bit quantization.

    Args:
        module: The nn.Linear layer whose weights will be modified in-place.
        bits: Bit-width for quantization. Default 4 (INT4, range [-8, 7]).

    Returns:
        None. Modifies module.weight.data in-place.
    """
    
    W = module.weight.data
    scale = torch.max(W.abs(), dim = 1)[0] / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min = 1e-8) # Prevent division by zero
    W_q = torch.clamp( (W / scale[:, None]).round(), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    W.copy_(W_q * scale[:, None])
    
def save_original_weights(model: AutoModelForCausalLM, 
                          layer_names: list[str]
        ) -> dict[str, torch.Tensor]:
    
    """
    Description:
        Deep-copy weights of specified Linear layers before quantization.

        Walks model.named_modules() and clones .weight.data for each layer whose
        fully qualified name appears in layer_names. Used to snapshot FP16 weights
        so they can be restored after a quantize-dequantize sweep step.

    Args:
        model: HuggingFace causal LM with named modules.
        layer_names: List of fully qualified layer names to save
            (e.g., ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.gate_proj']).

    Returns:
        Dict mapping each layer name to its cloned weight tensor.

    Raises:
        ValueError: If any name in layer_names is not found as an nn.Linear in the model.
    """
    
    layers_set = set(layer_names)
    original_weights_dict = {}
    
    for name, module in model.named_modules():
        if name in layers_set and isinstance(module, nn.Linear):
            original_weights_dict[name] = module.weight.data.clone()
            
    missing_layers = layers_set - set(original_weights_dict.keys())
    if missing_layers:
        raise ValueError(f"Layers not found in model: {missing_layers}")
    
    return original_weights_dict

def restore_weights(model: AutoModelForCausalLM,
                    layer_names: list[str],
                    original_weights_dict: dict[str, torch.Tensor]
        ) -> None:
    
    """
    Description:
        Restore original FP16 weights after a quantize-dequantize sweep step.

        Copies saved weight tensors back into the model's nn.Linear layers,
        undoing any in-place modifications from quantize_dequantize_layer().
        Must be called with the same layer_names used in save_original_weights().

    Args:
        model: HuggingFace causal LM with named modules.
        layer_names: List of fully qualified layer names to restore
            (e.g., ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.gate_proj']).
        original_weights_dict: Dict mapping layer names to their saved FP16 weight
            tensors, as returned by save_original_weights().

    Returns:
        None. Modifies module.weight.data in-place via copy_().

    Raises:
        ValueError: If any name in layer_names is not found as an nn.Linear in the model.
    """
    
    layers_set = set(layer_names)
    restored = set()
    
    for name, module in model.named_modules():
        if name in layers_set and isinstance(module, nn.Linear):
            module.weight.data.copy_(original_weights_dict[name])
            restored.add(name)
            
    missing_layers = layers_set - restored
    if missing_layers:
        raise ValueError(f"Layers not found in model for restoration: {missing_layers}")
            
@torch.no_grad()
def evaluate_rouge_on_subset(model: AutoModelForCausalLM,
                             tokenizer: AutoTokenizer,
                             data_config: dict,
                             eval_config: dict,
                             eval_samples: list[dict]
        ) -> dict:
    
    """
    Description:
        Evaluate ROUGE scores on a subset of samples using the given model.

        Runs inference on the provided sample dataset, extracts predictions and
        references, then computes ROUGE metrics. Used as a lightweight evaluation
        wrapper during sensitivity sweeps where full eval is too expensive.

    Args:
        model: HuggingFace causal LM to evaluate.
        tokenizer: Tokenizer corresponding to the model.
        data_config: Dataset/preprocessing config dict (prompt_template, max_input_length, etc.).
        eval_config: Generation config dict (max_new_tokens, num_beams, temperature, etc.).
        eval_samples: List of dataset sample dicts, each containing the input and
            target fields specified in data_config (e.g., 'article' and 'highlights').

    Returns:
        Dict of ROUGE scores as returned by compute_rouge(), e.g.:
            {'rouge1': 0.35, 'rouge2': 0.14, 'rougeL': 0.20, 'rougeLsum': 0.21}
    """
    
    infer_output = run_inference(model = model, 
                                tokenizer = tokenizer, 
                                data_config = data_config, 
                                eval_config = eval_config,
                                eval_samples = eval_samples)
    predictions = []
    references = []
    for idx in range(len(infer_output)):
        predictions.append(infer_output[idx]['prediction'])
        references.append(infer_output[idx]['reference'])
    
    rouge_scores = compute_rouge(predictions, references)
    return rouge_scores

@torch.no_grad()
def run_per_block_sweep(model: AutoModelForCausalLM,
                        tokenizer: AutoTokenizer,
                        data_config: dict,
                        eval_config: dict,
                        eval_samples: list[dict],
                        baseline_rouge: dict,
                        bits: int = 4
        ) -> list[dict]:
    
    """
    Description:
        Coarse sensitivity sweep: quantize all Linear layers in one block at a time.

        For each of the 32 transformer blocks, quantizes all 7 nn.Linear modules
        (q/k/v/o_proj + gate/up/down_proj) to simulated INT4 via round-trip,
        evaluates ROUGE-L on the eval subset, computes the delta from the FP16
        baseline, then restores original weights before moving to the next block.

    Args:
        model: LoRA-merged FP16 causal LM to sweep.
        tokenizer: Corresponding tokenizer.
        data_config: Dataset/preprocessing config for run_inference().
        eval_config: Generation config for run_inference().
        eval_samples: Subset of dataset samples for evaluation.
        baseline_rouge: FP16 baseline ROUGE scores dict (must contain 'rougeL').
        bits: Bit-width for simulated quantization. Default 4.

    Returns:
        List of dicts sorted by block index, each containing:
            - block_idx (str): Block index ('0', '1', ..., '31')
            - rouge_l (float): ROUGE-L after quantizing this block
            - rouge_l_delta (float): Drop from baseline (negative = degradation)
    """
    
    modules_dict = get_quantizable_layers(model)
    baseline_rouge_l = baseline_rouge['rougeL']
    results = []
    
    for block_idx in sorted(modules_dict.keys(), key=int):
        block_modules = modules_dict[block_idx]
    
        # Collect full names and nn.Linear refs for this block
        layer_names = [full_name for full_name, _ in block_modules.values()]
        layer_modules = [module for _, module in block_modules.values()]
        
        # Save → Quantize → Evaluate → Restore
        originals = save_original_weights(model, layer_names)
        
        for module in layer_modules:
            quantize_dequantize_layer(module, bits=bits)
            
        rouge_scores = evaluate_rouge_on_subset(model, 
                                                tokenizer, 
                                                data_config, 
                                                eval_config, 
                                                eval_samples)
        
        rouge_l = rouge_scores['rougeL']
        
        restore_weights(model, layer_names, originals)
        torch.cuda.empty_cache()
        
        results.append({
            'block_idx': block_idx,
            'rouge_l': rouge_l,
            'rouge_l_delta': rouge_l - baseline_rouge_l,
        })
        
        print(f"Block {block_idx:>2s}: ROUGE-L = {rouge_l:.4f} \
                delta = {rouge_l - baseline_rouge_l:+.4f}")
        
    return results

@torch.no_grad()
def run_per_module_sweep(model: AutoModelForCausalLM,
                        tokenizer: AutoTokenizer,
                        data_config: dict,
                        eval_config: dict,
                        eval_samples: list[dict],
                        baseline_rouge: dict,
                        bits: int = 4
        ) -> list[dict]:
    
    """
    Description:
        Fine sensitivity sweep: quantize one module type across all 32 blocks at a time.

        For each of the 7 module types (q/k/v/o_proj, gate/up/down_proj), quantizes
        that module in all 32 transformer blocks simultaneously via simulated INT4
        round-trip, evaluates ROUGE-L, computes delta from FP16 baseline, then
        restores original weights before moving to the next module type.

    Args:
        model: LoRA-merged FP16 causal LM to sweep.
        tokenizer: Corresponding tokenizer.
        data_config: Dataset/preprocessing config for run_inference().
        eval_config: Generation config for run_inference().
        eval_samples: Subset of dataset samples for evaluation.
        baseline_rouge: FP16 baseline ROUGE scores dict (must contain 'rougeL').
        bits: Bit-width for simulated quantization. Default 4.

    Returns:
        List of dicts sorted by module type, each containing:
            - module_type (str): Module name ('q_proj', 'k_proj', etc.)
            - rouge_l (float): ROUGE-L after quantizing this module type across all blocks
            - rouge_l_delta (float): Drop from baseline (negative = degradation)
    """
    
    module_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    modules_dict = get_quantizable_layers(model)
    module_dict_by_type = {mod_type: [] for mod_type in module_types}
    baseline_rouge_l = baseline_rouge['rougeL']
    results = []
    
    for block_idx in sorted(modules_dict.keys(), key=int):
        block_modules = modules_dict[block_idx]
        for mod_type in module_types:
            if mod_type in block_modules:
                module_dict_by_type[mod_type].append(block_modules[mod_type])
        
    for mod_type, entries in module_dict_by_type.items():
        
        # entries is list of (full_name, nn.Module) tuples across all 32 blocks
        layer_names = [full_name for full_name, _ in entries]
        layer_modules = [module for _, module in entries]
        
        # Save → Quantize → Evaluate → Restore
        originals = save_original_weights(model, layer_names)
        
        for module in layer_modules:
            quantize_dequantize_layer(module, bits=bits)
            
        rouge_scores = evaluate_rouge_on_subset(model, 
                                                tokenizer, 
                                                data_config, 
                                                eval_config, 
                                                eval_samples)
        
        rouge_l = rouge_scores['rougeL']
        
        restore_weights(model, layer_names, originals)
        torch.cuda.empty_cache()
        
        results.append({
            'module_type': mod_type,
            'rouge_l': rouge_l,
            'rouge_l_delta': rouge_l - baseline_rouge_l,
        })
        
        print(f"Module {mod_type:>10s}: ROUGE-L = {rouge_l:.4f} \
                delta = {rouge_l - baseline_rouge_l:+.4f}")
        
    return results

def rank_and_save(per_block_results: list[dict],
                  per_module_results: list[dict],
                  output_dir: str,
                  threshold: float = 0.005
    ) -> list[dict]:
    
    """
    Description:
        Rank sensitivity results by ROUGE-L degradation and save reports.

        Sorts per-block and per-module results by delta (largest drop first),
        assigns ranks, saves CSV tables for both sweeps, and writes a JSON file
        listing only the layers whose ROUGE-L drop exceeds the threshold —
        these become the QAT target list for Day 10.

    Args:
        per_block_results: Output from run_per_block_sweep(). List of dicts with
            keys 'block_idx', 'rouge_l', 'rouge_l_delta'.
        per_module_results: Output from run_per_module_sweep(). List of dicts with
            keys 'module_type', 'rouge_l', 'rouge_l_delta'.
        output_dir: Directory to save reports (e.g., 'outputs/reports').
        threshold: Minimum absolute ROUGE-L drop to flag a layer as QAT target.
            Default 0.005 (0.5 points).

    Returns:
        List of QAT target dicts (blocks exceeding threshold), each with
        'block_idx', 'rouge_l', 'rouge_l_delta', 'rank'.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Rank per-block results (most degraded first)
    block_ranked = sorted(per_block_results, key=lambda x: x['rouge_l_delta'])
    for i, entry in enumerate(block_ranked, 1):
        entry['rank'] = i
    
    # Rank per-module results (most degraded first)
    module_ranked = sorted(per_module_results, key=lambda x: x['rouge_l_delta'])
    for i, entry in enumerate(module_ranked, 1):
        entry['rank'] = i

    # Save per-block CSV
    block_csv_path = os.path.join(output_dir, 'sensitivity_per_block.csv')
    with open(block_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'block_idx', 'rouge_l', 'rouge_l_delta'])
        writer.writeheader()
        writer.writerows(block_ranked)
    print(f"Saved per-block results to {block_csv_path}")
    
    # Save per-module CSV
    module_csv_path = os.path.join(output_dir, 'sensitivity_per_module.csv')
    with open(module_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'module_type', 'rouge_l', 'rouge_l_delta'])
        writer.writeheader()
        writer.writerows(module_ranked)
    print(f"Saved per-module results to {module_csv_path}")
    
    # Filter QAT targets (delta is negative, so check abs)
    qat_targets = [entry for entry in block_ranked 
                   if abs(entry['rouge_l_delta']) > threshold]
    
    targets_path = os.path.join(output_dir, 'sensitivity_qat_targets.json')
    with open(targets_path, 'w') as f:
        json.dump(qat_targets, f, indent=2)
    print(f"Saved {len(qat_targets)} QAT targets (threshold={threshold}) to {targets_path}")
    
    return qat_targets

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    args = parser.parse_args()
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    sensitivity_config = load_config('configs/sensitivity_config.yaml')

    model, tokenizer = load_model_and_tokenizer(model_config = model_config)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    test_split = sensitivity_config['sensitivity']['samples']
    threshold = sensitivity_config['sensitivity']['threshold']
    bits = sensitivity_config['sensitivity']['bits']
    output_dir = sensitivity_config['sensitivity']['output_dir']
    
    eval_samples = load_dataset(os.path.join('assets/datasets', 
                                            dataset_name, 
                                            dataset_config), 
                                            split = test_split)
    
    baseline_rouge = evaluate_rouge_on_subset(model = model,
                                              tokenizer = tokenizer,
                                              data_config = data_config,
                                              eval_config = eval_config,
                                              eval_samples = eval_samples,
                        )               
    
    rouge_per_block =  run_per_block_sweep(model = model,
                                            tokenizer = tokenizer,
                                            data_config = data_config,
                                            eval_config = eval_config,
                                            eval_samples = eval_samples,
                                            baseline_rouge = baseline_rouge,
                                            bits = bits
                        )
    
    rouge_per_module = run_per_module_sweep(model = model,
                                            tokenizer = tokenizer,
                                            data_config = data_config,
                                            eval_config = eval_config,
                                            eval_samples = eval_samples,
                                            baseline_rouge = baseline_rouge,
                                            bits = bits
                            )
    
    _ = rank_and_save(per_block_results = rouge_per_block,
                                per_module_results = rouge_per_module,
                                output_dir = output_dir,
                                threshold = threshold
                    )

if __name__ == "__main__":
    main()