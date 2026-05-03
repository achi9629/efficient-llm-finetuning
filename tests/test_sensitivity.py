import os
import json
import torch
import pytest
import logging
import pandas as pd
import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer

from src.finetuning import get_quantizable_layers, quantize_dequantize_layer, save_original_weights, \
                           restore_weights, rank_and_save

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope = "function")
def tiny_mistral_path(tmp_path_factory):
    
    tmp_dir = tmp_path_factory.mktemp("tiny_mistral")
    config = MistralConfig(vocab_size = 32768,
                            hidden_size=256,
                            intermediate_size=128,
                            num_hidden_layers=2,
                            num_attention_heads=2,
                            num_key_value_heads=2,
                            max_position_embeddings=128,
                )
    model = MistralForCausalLM(config)
    model.save_pretrained(tmp_dir)
    
    model_mistral = MistralForCausalLM.from_pretrained(tmp_dir,
                                                        config = config,
                                                        low_cpu_mem_usage = True,
                                                        device_map = {"": 0},
                                                        torch_dtype = torch.float16,
                                                        )
    
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    
    return model_mistral, tokenizer, tmp_dir

def test_get_quantizable_layers_returns_all_blocks(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    keys = list(module_dict.keys())
    assert len(keys) == 2, f"Expected 2 quantizable layers, but got {len(keys)}"
    assert keys == ['0', '1'], f"Expected keys ['0', '1'], but got {keys}"
    
def test_get_quantizable_layers_correct_module_names(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    expected_module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"]
    
    for block_key, block_module in module_dict.items():
        block_module_keys = list(block_module.keys())
        assert block_module_keys == expected_module_keys, f"Expected module keys {expected_module_keys} in block {block_key}, but got {block_module_keys}"
        
def test_get_quantizable_layers_excludes_lm_head(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    non_expected_keys = ["lm_head", "embed_tokens"]
    
    for block_key, block_module in module_dict.items():
        block_module_keys = list(block_module.keys())
        for non_expected_key in non_expected_keys:
            assert non_expected_key not in block_module_keys, f"Did not expect {non_expected_key} in block {block_key}, but it was found"
            
def test_get_quantizable_layers_returns_linear_modules(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    for block_key, block_module in module_dict.items():
        for module_key, module in block_module.items():
            assert isinstance(module[1], nn.Linear), f"Expected module {module_key} in block {block_key} to be an instance of nn.Linear, but got {type(module)}"
            
def test_quantize_dequantize_changes_weights(tiny_mistral_path):
    
    bits = 4
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    for block_key, block_module in module_dict.items():
        for module_key, module in block_module.items():
            original_weight = module[1].weight.data.clone()
            quantize_dequantize_layer(module[1], bits)
            assert not torch.equal(original_weight, module[1].weight.data), f"Expected quantized weights to differ from original weights in module {module_key} of block {block_key}, but they are the same"
            
def test_quantize_dequantize_no_nan_inf(tiny_mistral_path):
    
    bits = 4
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    for block_key, block_module in module_dict.items():
        for module_key, module in block_module.items():
            quantize_dequantize_layer(module[1], bits)
            assert not torch.isnan(module[1].weight.data).any(), f"Expected no NaN values in quantized weights of module {module_key} in block {block_key}, but found NaN"
            assert not torch.isinf(module[1].weight.data).any(), f"Expected no Inf values in quantized weights of module {module_key} in block {block_key}, but found Inf"
            
def test_quantize_dequantize_within_range(tiny_mistral_path):
    
    bits = 4
    min_val = -8
    max_val = 7
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    for block_key, block_module in module_dict.items():
        for module_key, module in block_module.items():
            quantize_dequantize_layer(module[1], bits)
            assert torch.all(module[1].weight.data >= min_val), f"Expected all quantized weights in module {module_key} of block {block_key} to be >= {min_val}, but found values below {min_val}"
            assert torch.all(module[1].weight.data <= max_val), f"Expected all quantized weights in module {module_key} of block {block_key} to be <= {max_val}, but found values above {max_val}"
            
def test_quantize_dequantize_zero_weight_safe():
    
    linear_layer = nn.Linear(10, 10)
    linear_layer.weight.data.fill_(0)
    
    try:
        quantize_dequantize_layer(linear_layer, bits = 4)
    except Exception as e:
        pytest.fail(f"Quantizing and dequantizing a linear layer with zero weights raised an exception: {e}")
        
def test_save_and_restore_exact_match(tiny_mistral_path):
    
    bits = 4
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    key = '0'
    block_modules = module_dict[key]
    
    layer_names = [full_name for full_name, _ in block_modules.values()]
    layer_modules = [module for _, module in block_modules.values()]
    
    originals = save_original_weights(model, layer_names)
    
    for module in layer_modules:
        quantize_dequantize_layer(module, bits = bits)
        
    restore_weights(model, layer_names, originals)
    
    for name, module in model.named_modules():
        if name in originals.keys():
            original_weight = originals[name]
            restored_weight = module.weight.data
            
            assert torch.equal(original_weight, restored_weight), f"Expected restored weights to exactly match original weights for layer {name}, but they differ"
            
def test_save_raises_on_missing_layer(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    
    layer_names = ['model.embed_tokens']
    with pytest.raises(ValueError, match = f"Layers not found in model: {set(['model.embed_tokens'])}"):
        _ = save_original_weights(model, layer_names)
        
def test_restore_raises_on_missing_layer(tiny_mistral_path):
    
    model, _, _ = tiny_mistral_path
    
    layer_names_exist = ['model.layers.0.self_attn.q_proj']
    originals = save_original_weights(model, layer_names_exist)
    
    layer_names_non_exist = ['model.embed_tokens', 'model.norm', 'model.layers.0.self_attn.q_proj']
    with pytest.raises(ValueError, match = f"Layers not found in model for restoration: {set(layer_names_non_exist) - set(layer_names_exist)}"):
        restore_weights(model, layer_names_non_exist, originals)
        
def test_quantize_does_not_affect_other_blocks(tiny_mistral_path):
    
    bits = 4
    key = '0'
    model, _, _ = tiny_mistral_path
    module_dict = get_quantizable_layers(model)
    
    block_modules_1 = module_dict['1']
    layer_names_1 = [full_name for full_name, _ in block_modules_1.values()]
    originals_block_1_weights = save_original_weights(model, layer_names_1)
    
    block_modules_0 = module_dict['0']
    layer_modules_0 = [module for _, module in block_modules_0.values()]
    
    for module in layer_modules_0:
        quantize_dequantize_layer(module, bits = bits)
        
    for name, module in model.named_modules():
        if name in originals_block_1_weights.keys():
            original_weight = originals_block_1_weights[name]
            current_weight = module.weight.data
            
            assert torch.equal(original_weight, current_weight), f"Expected weights of block 1 to remain unchanged after quantizing block 0, but weights of layer {name} changed"
    
def test_rank_and_save_creates_files(tmp_path):
    
    per_block_results = [
                        {'block_idx': 0, 'rouge_l': 0.5, 'rouge_l_delta': 0.0005},
                        {'block_idx': 1, 'rouge_l': 0.45, 'rouge_l_delta': 0.0003},
    ]
    
    per_module_results = [
        {'module_type': 'q_proj', 'rouge_l': 0.48, 'rouge_l_delta': 0.0004},
        {'module_type': 'k_proj', 'rouge_l': 0.47, 'rouge_l_delta': 0.0002},
        {'module_type': 'v_proj', 'rouge_l': 0.46, 'rouge_l_delta': 0.0001},
        {'module_type': 'o_proj', 'rouge_l': 0.44, 'rouge_l_delta': 0.0003},
        {'module_type': 'gate_proj', 'rouge_l': 0.43, 'rouge_l_delta': 0.0002},
        {'module_type': 'up_proj', 'rouge_l': 0.42, 'rouge_l_delta': 0.0001},
        {'module_type': 'down_proj', 'rouge_l': 0.41, 'rouge_l_delta': 0.00005},
    ]
    
    _ = rank_and_save(per_block_results = per_block_results, 
                      per_module_results = per_module_results, 
                      output_dir = tmp_path)
    
    assert os.path.isfile(os.path.join(tmp_path / "sensitivity_per_block.csv")), "Expected sensitivity_per_block.csv to be created in output directory, but file was not found"
    assert os.path.isfile(os.path.join(tmp_path / "sensitivity_per_module.csv")), "Expected sensitivity_per_module.csv to be created in output directory, but file was not found"
    assert os.path.isfile(os.path.join(tmp_path / "sensitivity_qat_targets.json")), "Expected sensitivity_qat_targets.json to be created in output directory, but file was not found"
    
def test_rank_and_save_correct_ordering(tmp_path):
    
    per_block_results = [
                        {'block_idx': 5, 'rouge_l': 0.5, 'rouge_l_delta': -0.02},
                        {'block_idx': 10, 'rouge_l': 0.45, 'rouge_l_delta': -0.01},
                        {'block_idx': 3, 'rouge_l': 0.45, 'rouge_l_delta': -0.03},
    ]
    
    
    per_module_results = [
        {'module_type': 'q_proj', 'rouge_l': 0.48, 'rouge_l_delta': 0.0004},
        {'module_type': 'k_proj', 'rouge_l': 0.47, 'rouge_l_delta': 0.0002},
        {'module_type': 'v_proj', 'rouge_l': 0.46, 'rouge_l_delta': 0.0001},
        {'module_type': 'o_proj', 'rouge_l': 0.44, 'rouge_l_delta': 0.0003},
        {'module_type': 'gate_proj', 'rouge_l': 0.43, 'rouge_l_delta': 0.0002},
        {'module_type': 'up_proj', 'rouge_l': 0.42, 'rouge_l_delta': 0.0001},
        {'module_type': 'down_proj', 'rouge_l': 0.41, 'rouge_l_delta': 0.00005},
    ]
    
    _ = rank_and_save(per_block_results = per_block_results, 
                      per_module_results = per_module_results, 
                      output_dir = tmp_path)
    
    per_block_results_df = pd.read_csv(os.path.join(tmp_path / "sensitivity_per_block.csv"))
    
    assert per_block_results_df.block_idx.tolist() == [3, 5, 10], f"Expected per-block results to be ordered by block_idx [3, 5, 10], but got {per_block_results_df.block_idx.tolist()}"
    
def test_rank_and_save_threshold_filtering(tmp_path):
    
    per_block_results = [
                        {'block_idx': 5, 'rouge_l': 0.5, 'rouge_l_delta': -0.02},
                        {'block_idx': 10, 'rouge_l': 0.45, 'rouge_l_delta': -0.01},
                        {'block_idx': 3, 'rouge_l': 0.45, 'rouge_l_delta': -0.03},
    ]
    
    
    per_module_results = [
        {'module_type': 'q_proj', 'rouge_l': 0.48, 'rouge_l_delta': 0.0004},
        {'module_type': 'k_proj', 'rouge_l': 0.47, 'rouge_l_delta': 0.0002},
        {'module_type': 'v_proj', 'rouge_l': 0.46, 'rouge_l_delta': 0.0001},
        {'module_type': 'o_proj', 'rouge_l': 0.44, 'rouge_l_delta': 0.0003},
        {'module_type': 'gate_proj', 'rouge_l': 0.43, 'rouge_l_delta': 0.0002},
        {'module_type': 'up_proj', 'rouge_l': 0.42, 'rouge_l_delta': 0.0001},
        {'module_type': 'down_proj', 'rouge_l': 0.41, 'rouge_l_delta': 0.00005},
    ]
    
    _ = rank_and_save(per_block_results = per_block_results, 
                      per_module_results = per_module_results, 
                      output_dir = tmp_path,
                      threshold = 0.015)
    
    json_path = os.path.join(tmp_path / "sensitivity_qat_targets.json")
    with open(json_path, "r") as f:
        qat_targets = json.load(f)

    expected_qat_targets = [{'block_idx': 3, 'rouge_l': 0.45, 'rouge_l_delta': -0.03, 'rank': 1}, 
                            {'block_idx': 5, 'rouge_l': 0.5, 'rouge_l_delta': -0.02, 'rank': 2}]
    
    assert qat_targets == expected_qat_targets, f"Expected QAT targets to be {expected_qat_targets} based on threshold filtering, but got {qat_targets}"
    
def test_rank_and_save_empty_targets(tmp_path):
    
    per_block_results = [
                        {'block_idx': 5, 'rouge_l': 0.5, 'rouge_l_delta': -0.02},
                        {'block_idx': 10, 'rouge_l': 0.45, 'rouge_l_delta': -0.01},
                        {'block_idx': 3, 'rouge_l': 0.45, 'rouge_l_delta': -0.03},
    ]
    
    
    per_module_results = [
        {'module_type': 'q_proj', 'rouge_l': 0.48, 'rouge_l_delta': 0.0004},
        {'module_type': 'k_proj', 'rouge_l': 0.47, 'rouge_l_delta': 0.0002},
        {'module_type': 'v_proj', 'rouge_l': 0.46, 'rouge_l_delta': 0.0001},
        {'module_type': 'o_proj', 'rouge_l': 0.44, 'rouge_l_delta': 0.0003},
        {'module_type': 'gate_proj', 'rouge_l': 0.43, 'rouge_l_delta': 0.0002},
        {'module_type': 'up_proj', 'rouge_l': 0.42, 'rouge_l_delta': 0.0001},
        {'module_type': 'down_proj', 'rouge_l': 0.41, 'rouge_l_delta': 0.00005},
    ]
    
    _ = rank_and_save(per_block_results = per_block_results, 
                      per_module_results = per_module_results, 
                      output_dir = tmp_path,
                      threshold = 0.1)
    
    json_path = os.path.join(tmp_path / "sensitivity_qat_targets.json")
    with open(json_path, "r") as f:
        qat_targets = json.load(f)
        
    assert qat_targets == [], f"Expected QAT targets to be an empty list when all blocks are above the threshold, but got {qat_targets}"