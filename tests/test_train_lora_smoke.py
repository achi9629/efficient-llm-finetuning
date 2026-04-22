import torch
import pytest
import logging
from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer

from finetuning import inject_lora, merge_lora, save_lora_weights, \
                       load_lora_weights, LoRALinear

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def lora_model(tmp_path_factory):
    
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
    
    model_mistral = MistralForCausalLM.from_pretrained(tmp_dir,
                                                        config = config,
                                                        low_cpu_mem_usage = True,
                                                        device_map = {"": 0},
                                                        torch_dtype = torch.float16,
                                                        )
    
    for name, param in model_mistral.named_parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    
    return model_mistral, tokenizer, tmp_dir

def test_inject_lora(lora_model):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    
    model, _, _ = lora_model
    
    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    
    assert any(isinstance(module, LoRALinear) for name, module in model.named_modules()), "No LoRA layers were injected into the model."
    for name, param in model.named_parameters():
        if 'lora_' in name:
            assert param.requires_grad, f"LoRA parameter {name} is not set to require gradients."
        else:
            assert not param.requires_grad, f"Non-LoRA parameter {name} should not require gradients."
    
    lora_param = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    total_param = sum(p.numel() for p in model.parameters())

    assert lora_param > 0, "No LoRA parameters were added to the model."
    assert lora_param/total_param < 0.1, "LoRA parameters should be a small fraction of the total model parameters."
    
def test_forward_pass_with_lora(lora_model):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model, _, _ = lora_model
    
    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float16)
    
    output = model(input_ids=input_ids)
    
    assert output is not None, "Model forward pass with LoRA layers returned None."
    assert output.logits.shape == (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size), f"Expected output shape {(input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)}, but got {output.logits.shape}."

def test_backward_pass_with_lora(lora_model):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model, _, _ = lora_model
    
    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float16)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model.parameters()), 
                                  lr=1e-3)
    model.train()
    
    optimizer.zero_grad()
    logits = model(input_ids = input_ids).logits
    labels = input_ids
        
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    
    optimizer.step()
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            assert param.grad is not None, f"LoRA parameter {name} did not receive gradients during backpropagation."
        else:
            assert param.grad is None, f"Non-LoRA parameter {name} should not receive gradients during backpropagation."
            
def test_merge_lora(lora_model):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model, _, _ = lora_model
    
    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float16)
    
    logits_before = model(input_ids = input_ids).logits
    
    model = merge_lora(model)
    logits_after = model(input_ids = input_ids).logits
    
    assert not any(isinstance(module, LoRALinear) for name, module in model.named_modules()), "LoRA layers were not merged properly and still exist in the model."
    assert torch.allclose(logits_before, logits_after, atol=1e-5), "Logits before and after merging LoRA layers do not match closely enough, indicating a potential issue with the merging process."
    
def test_save_load_lora_weights(lora_model, tmp_path):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model, _, _ = lora_model
    base_model_keys = set(name for name, _ in model.named_parameters())
    
    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float16)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model.parameters()), 
                                  lr=1e-3)
    model.train()
    
    optimizer.zero_grad()
    logits = model(input_ids = input_ids).logits
    labels = input_ids
        
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    
    optimizer.step()
    
    
    save_path = tmp_path / "lora_weights.pt"
    save_lora_weights(model, save_path)
    
    assert save_path.exists(), "LoRA weights file was not saved successfully."
    
    state_dict = torch.load(save_path, weights_only = True)
    keys = list(state_dict.keys())
    assert 'r' in keys and 'alpha' in keys and 'dropout' in keys, "Saved LoRA weights file is missing required keys."
    for key in keys:
        if key not in ['r', 'alpha', 'dropout']:
            assert 'lora_A' in key or 'lora_B' in key, f"Unexpected key {key} found in saved LoRA weights file, which may indicate an issue with the saving process."
        else:
            assert key not in base_model_keys, f"Key {key} from the base model should not be present in the saved LoRA weights file, indicating that only LoRA-specific parameters were saved."
        
def test_load_lora_weights_roundtrip(lora_model, tmp_path):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8]]).cuda()
    
    model, _, tmp_dir = lora_model

    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float32)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model.parameters()), 
                                  lr=1e-3)
    model.train()
    
    optimizer.zero_grad()
    logits = model(input_ids = input_ids).logits
    labels = input_ids
        
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    
    logits_before = model(input_ids = input_ids).logits
    
    save_path = tmp_path / "lora_weights.pt"
    save_lora_weights(model, save_path)
    
    config = MistralConfig(vocab_size=1000,
                            hidden_size=64,
                            intermediate_size=128,
                            num_hidden_layers=2,
                            num_attention_heads=2,
                            num_key_value_heads=2,
                            max_position_embeddings=128,
                )
    
    model_new = MistralForCausalLM.from_pretrained(tmp_dir,
                                                    config = config,
                                                    low_cpu_mem_usage = True,
                                                    device_map = {"": 0},
                                                    torch_dtype = torch.float16,
                                                    )
    
    for name, param in model_new.named_parameters():
        param.requires_grad = False
        
    model_new = inject_lora(model = model_new,
                            target_modules = target_modules,
                            r = r,
                            alpha = alpha,
                            dropout = dropout)
    
    model_new = load_lora_weights(model = model_new, 
                              path = save_path,
                              target_modules = target_modules)
    model_new = model_new.to(dtype = torch.float32)
    
    logits_after = model_new(input_ids = input_ids).logits
    
    assert torch.allclose(logits_before, logits_after, atol=1e-5), "Logits before saving and after loading LoRA weights do not match closely enough, indicating a potential issue with the loading process."
    
def test_lora_loss_decreases(lora_model):
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    r = 4
    alpha = 8
    dropout = 0.0
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    
    model, _, tmp_dir = lora_model

    model = inject_lora(model = model,
                        target_modules = target_modules,
                        r = r,
                        alpha = alpha,
                        dropout = dropout)
    model = model.to(dtype=torch.float32) # changed to float32 for better numerical stability during training
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                         model.parameters()), 
                                  lr=1e-3)
    model.train()
    
    for step in range(10):
    
        optimizer.zero_grad()
        logits = model(input_ids = input_ids).logits
        labels = input_ids
            
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        elif step == 9:
            final_loss = loss.item()
            
    assert final_loss < initial_loss, f"Loss did not decrease after training with LoRA layers. Initial loss: {initial_loss}, Final loss: {final_loss}"