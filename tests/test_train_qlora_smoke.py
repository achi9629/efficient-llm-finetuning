import torch
import pytest
import logging
from bitsandbytes.nn import Linear4bit
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, MistralConfig, MistralForCausalLM, AutoTokenizer

from finetuning import load_config, inject_lora, save_lora_weights, \
                       load_lora_weights, LoRALinear

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dtype_map = {"float16": torch.float16, 
                 "bfloat16": torch.bfloat16, 
                 "float32": torch.float32}

@pytest.fixture
def qlora_model_and_tokenizer(tmp_path_factory):
    
    bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                    bnb_4bit_quant_type = "nf4",
                                    bnb_4bit_use_double_quant = True,
                                    bnb_4bit_compute_dtype = torch.float16
                    )
    
    tmp_dir = tmp_path_factory.mktemp("tiny_mistral")
    config = MistralConfig(vocab_size=1000,
                            hidden_size=64,
                            intermediate_size=128,
                            num_hidden_layers=2,
                            num_attention_heads=2,
                            num_key_value_heads=2,
                            max_position_embeddings=128,
                )
    model = MistralForCausalLM(config)
    model.save_pretrained(tmp_dir)
    
    model_4bit = MistralForCausalLM.from_pretrained(tmp_dir, 
                                                    quantization_config = bnb_config,
                                                    device_map = {"": 0}
                    )
    
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    
    return model_4bit, tokenizer, tmp_dir

def test_bnb_config_creation():
    
    qlora_config = load_config('configs/train_qlora.yaml')
    
    load_in_4bit = qlora_config['quantization']['load_in_4bit']
    bnb_4bit_quant_type = qlora_config['quantization']['bnb_4bit_quant_type']
    bnb_4bit_use_double_quant = qlora_config['quantization']['bnb_4bit_use_double_quant']
    bnb_4bit_compute_dtype = dtype_map[qlora_config['quantization']['bnb_4bit_compute_dtype']]
    
    
    bnb_config = BitsAndBytesConfig(load_in_4bit = load_in_4bit,
                                    bnb_4bit_quant_type = bnb_4bit_quant_type,
                                    bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
                                    bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
                    )
    
    assert bnb_config.load_in_4bit == load_in_4bit, "load_in_4bit does not match"
    assert bnb_config.bnb_4bit_quant_type == bnb_4bit_quant_type, "bnb_4bit_quant_type does not match"
    assert bnb_config.bnb_4bit_use_double_quant == bnb_4bit_use_double_quant, "bnb_4bit_use_double_quant does not match"
    assert bnb_config.bnb_4bit_compute_dtype == bnb_4bit_compute_dtype, "bnb_4bit_compute_dtype does not match"
    
def test_4bit_model_loading(qlora_model_and_tokenizer):
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    
    assert model_4bit is not None, "Model loading failed"
    assert any( isinstance(module, Linear4bit) for name, module in model_4bit.named_modules()), "No 4-bit linear layers found in the model"
    
    '''
    Using device_map={"": 0} should place the entire model on the first CUDA device. 
    We check that at least one parameter is on CUDA to confirm this. If the model is 
    not on CUDA, we raise an assertion error with a message indicating the actual 
    device type found. Checking atleast one parameter is sufficient because if the model
    is correctly loaded on CUDA, all parameters should be on CUDA. Also if the model size if 
    large, checking all parameters might be time consuming, so we check just one parameter 
    to confirm the device placement.
    '''
    assert next(model_4bit.parameters()).device.type == "cuda", f"Model is not on CUDA device, found on {next(model_4bit.parameters()).device.type}"
    
def test_prepare_kbit_training (qlora_model_and_tokenizer):
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    
    assert all(not param.requires_grad for name, param in model_4bit.named_parameters()), f"All parameters should be frozen after prepare_model_for_kbit_training, but found some parameters that are not frozen."
    assert model_4bit.is_gradient_checkpointing == True, "Gradient checkpointing should be enabled after prepare_model_for_kbit_training"
    
def test_inject_lora_on_4bit_model(qlora_model_and_tokenizer):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    
    model_4bit = inject_lora(model = model_4bit, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = 0.0)
    
    assert any(isinstance(m, LoRALinear) for _, m in model_4bit.named_modules()), "No LoRALinear layers found in the model after inject_lora"
    for name, param in model_4bit.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            assert param.requires_grad, f"LoRA param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Non-LoRA param {name} should be frozen"
            
    count_lora_param = sum(param.numel() for name, param in model_4bit.named_parameters() if 'lora_A' in name or 'lora_B' in name)
    count_param = sum(param.numel() for name, param in model_4bit.named_parameters())
    
    assert count_lora_param > 0, "No LoRA parameters found in the model after inject_lora"
    assert count_lora_param/count_param < 0.1, f"LoRA parameters should be a small fraction of total parameters, but found {count_lora_param/count_param:.2f}"
    
def test_forward_pass_with_qlora(qlora_model_and_tokenizer):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    
    model_4bit = inject_lora(model = model_4bit, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    vocab_size = model_4bit.config.vocab_size
    output = model_4bit(input_ids = input_ids)
    assert output is not None, "Forward pass failed, logits is None"
    assert output.logits.shape == (1, 5, vocab_size), f"Logits shape mismatch, expected (1, 5, {vocab_size}), but got {model_4bit(input_ids = input_ids).logits.shape}"
    
def test_backward_pass_with_qlora(qlora_model_and_tokenizer):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    
    model_4bit = inject_lora(model = model_4bit, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model_4bit.parameters()), 
                                  lr=1e-3)
    
    optimizer.zero_grad()
    logits = model_4bit(input_ids = input_ids).logits
    labels = input_ids
    
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    
    for name, param in model_4bit.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            assert param.grad is not None, f"LoRA param {name} should have gradients after backward pass"
        else:
            assert param.grad is None, f"Non-LoRA param {name} should not have gradients after backward pass"
            
def test_save_load_qlora_weights(qlora_model_and_tokenizer, tmp_path):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    base_model_keys = [name for name, param in model_4bit.named_parameters()]
    
    model_4bit = inject_lora(model = model_4bit, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model_4bit.parameters()), 
                                  lr=1e-3)
    
    optimizer.zero_grad()
    logits = model_4bit(input_ids = input_ids).logits
    labels = input_ids
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    
    save_full_model_path = tmp_path / "full_model.pt"
    save_lora_weights(model_4bit, save_full_model_path)
    
    assert save_full_model_path.exists(), "Saved LoRA weights file does not exist"
    
    state_dict = torch.load(save_full_model_path, weights_only = True)
    
    assert 'r' in state_dict.keys() and 'alpha' in state_dict.keys() and 'dropout' in state_dict.keys(), "Saved state dict should contain 'r', 'alpha', and 'dropout' keys"
    for key in state_dict.keys():
        if key not in ['r', 'alpha', 'dropout']:
            assert 'lora_A' in key or 'lora_B' in key, f"Unexpected key {key} found in saved state dict, expected only 'r', 'alpha', 'lora_state_dict', and LoRA parameters"
        assert key not in base_model_keys, f"Base model parameter {key} should not be saved in LoRA state dict"

def test_load_qlora_weights_roundtrip(qlora_model_and_tokenizer, tmp_path):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    
    model_4bit, _, tmp_dir = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)

    model_4bit = inject_lora(model = model_4bit, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model_4bit.parameters()), 
                                  lr=1e-3)
    optimizer.zero_grad()
    logits = model_4bit(input_ids = input_ids).logits
    labels = input_ids
    
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    
    save_full_model_path = tmp_path / "full_model.pt"
    save_lora_weights(model_4bit, save_full_model_path)
    
    bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                    bnb_4bit_quant_type = "nf4",
                                    bnb_4bit_use_double_quant = True,
                                    bnb_4bit_compute_dtype = torch.float16
                    )
    
    model_4bit_new = MistralForCausalLM.from_pretrained(tmp_dir, 
                                                        quantization_config = bnb_config,
                                                        device_map = {"": 0}
                        )
    
    model_4bit_new = prepare_model_for_kbit_training(model_4bit_new)

    model_4bit_new = inject_lora(model = model_4bit_new, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    model_4bit_new = load_lora_weights(model = model_4bit_new,
                                       path = save_full_model_path,
                                       target_modules = target_modules)
    
    logits_original = model_4bit(input_ids = input_ids).logits
    logits_new = model_4bit_new(input_ids = input_ids).logits
    
    assert torch.allclose(logits_original, logits_new, atol = 1e-5), f"Logits mismatch after loading LoRA weights, max absolute difference: {(logits_original - logits_new).abs().max().item()}"
    
def test_qlora_loss_decreases(qlora_model_and_tokenizer):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model_4bit, _, _ = qlora_model_and_tokenizer
    model_4bit = prepare_model_for_kbit_training(model_4bit)

    model_4bit = inject_lora(model = model_4bit, 
                            target_modules = target_modules, 
                            r = r, 
                            alpha = alpha, 
                            dropout = 0.0)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model_4bit.parameters()), 
                                  lr=1e-3)
    
    for step in range(10):
        optimizer.zero_grad()
        logits = model_4bit(input_ids = input_ids).logits
        labels = input_ids
        
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        
        optimizer.step()
        
        if step == 0:
            loss_step_0 = loss.item()
        if step == 9:
            loss_step_9 = loss.item()
            
    assert loss_step_9 < loss_step_0, f"Loss did not decrease after training, initial loss: {loss_step_0:.4f}, final loss: {loss_step_9:.4f}"
