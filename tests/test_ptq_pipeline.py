import os
import json
import torch
import pytest
import logging
import peft.peft_model
if not hasattr(peft.peft_model, 'PEFT_TYPE_TO_MODEL_MAPPING'):
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING = {}
from awq import AutoAWQForCausalLM
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM
from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer

from src.finetuning import load_config, load_ptq_model_and_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="module")
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
    
    return tmp_dir

@pytest.fixture(scope="module")
def gptq_quantized_checkpoint(tiny_mistral_path, tmp_path_factory):
    """Quantize once, reuse across all GPTQ tests in this module."""
    
    bits = 4
    group_size = 128
    desc_act = False
    torch_dtype = torch.float16
    trust_remote_code = False
    device_map = "auto"
    checkpoint_dir = str(tmp_path_factory.mktemp("gptq_shared"))
    
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    quantize_config = BaseQuantizeConfig(bits = bits,
                                         group_size = group_size,
                                         desc_act = desc_act,
                        )
    
    model = AutoGPTQForCausalLM.from_pretrained(tiny_mistral_path,
                                                quantize_config = quantize_config,
                                                torch_dtype = torch_dtype,
                                                trust_remote_code = trust_remote_code,
                                                device_map = device_map,
                )
    
    calib_samples = [
        "The cat sat on the mat.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The weather today is sunny with a chance of rain.",
        "OpenAI's GPT models are powerful language models."
    ]
    # Tokenize calibration samples for auto_gptq
    calib_tokenized = [tokenizer(s, return_tensors = "pt", 
                                 truncation = True,
                                 max_length = 128)
                        for s in calib_samples
                        ]
    
    model.quantize(calib_tokenized)
    model.save_quantized(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    return checkpoint_dir

@pytest.fixture(scope="module")
def awq_quantized_checkpoint(tiny_mistral_path, tmp_path_factory):
    
    device_map = "auto"
    torch_dtype = torch.float16
    trust_remote_code = False
    zero_point = True
    q_group_size = 128
    w_bit = 4
    version = 'GEMM'
    checkpoint_dir = str(tmp_path_factory.mktemp("awq_shared"))
    
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    model = AutoAWQForCausalLM.from_pretrained(tiny_mistral_path, 
                                               torch_dtype = torch_dtype, 
                                               device_map = device_map, 
                                               trust_remote_code = trust_remote_code)
    
    quant_config = {"zero_point": zero_point, 
                    "q_group_size": q_group_size, 
                    "w_bit": w_bit, 
                    "version": version}
    
    calib_samples = [
        "The cat sat on the mat.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The weather today is sunny with a chance of rain.",
        "OpenAI's GPT models are powerful language models."
    ]
    
    model.quantize(tokenizer, 
                   quant_config = quant_config,
                   calib_data = calib_samples*20,
                   max_calib_seq_len=128)
    model.save_quantized(checkpoint_dir)
    
    return checkpoint_dir

def test_ptq_config_loading():
    
    ptq_config = load_config("configs/ptq_config.yaml")
    
    assert len(ptq_config['ptq']['modes']) == 4, "Expected 4 PTQ modes defined in the configuration"
    assert ptq_config['ptq']['modes'][0]['name'] == 'int8', "First PTQ mode should be 'int8'"
    assert ptq_config['ptq']['modes'][0]['load_in_8bit'] == True, "First PTQ mode should have 'load_in_8bit' set to True"
    assert ptq_config['ptq']['modes'][0]['load_in_4bit'] == False, "First PTQ mode should have 'load_in_4bit' set to False"
    assert ptq_config['ptq']['modes'][1]['name'] == 'nf4', "Second PTQ mode should be 'nf4'"
    assert ptq_config['ptq']['modes'][1]['load_in_8bit'] == False, "Second PTQ mode should have 'load_in_8bit' set to False"
    assert ptq_config['ptq']['modes'][1]['load_in_4bit'] == True, "Second PTQ mode should have 'load_in_4bit' set to True"
    assert ptq_config['ptq']['modes'][1]['bnb_4bit_quant_type'] == 'nf4', "Second PTQ mode should have 'bnb_4bit_quant_type' set to 'nf4'"
    assert ptq_config['ptq']['modes'][2]['name'] == 'gptq_int4', "Third PTQ mode should be 'gptq_int4'"
    assert ptq_config['ptq']['modes'][2]['bits'] == 4, "Third PTQ mode should have 'bits' set to 4"
    assert ptq_config['ptq']['modes'][2]['group_size'] == 128, "Third PTQ mode should have 'group_size' set to 128"
    assert 'checkpoint_dir' in ptq_config['ptq']['modes'][2], "Third PTQ mode should have 'checkpoint_dir' specified in the configuration"
    
def test_int8_model_loading(tiny_mistral_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'int8',
        'load_in_8bit': True,
        'load_in_4bit': False
    }
    
    model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'int8')
    
    assert model is not None, "Model should be loaded successfully for int8 mode"
    assert model.config is not None, "Model should have a valid configuration"
    assert any( isinstance(module, Linear8bitLt) for name, module in model.named_modules()), "Model should contain Int8Params modules for int8 quantization"
    assert tokenizer.pad_token == tokenizer.eos_token, "Tokenizer pad token should be the same as eos token for causal language models"
    
def test_nf4_model_loading(tiny_mistral_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'nf4',
        'load_in_8bit': False,
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_use_double_quant': False,
    }
    
    model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'nf4')
    
    assert model is not None, "Model should be loaded successfully for int8 mode"
    assert model.config is not None, "Model should have a valid configuration"
    assert tokenizer.pad_token == tokenizer.eos_token, "Tokenizer pad token should be the same as eos token for causal language models"
    assert any( isinstance(module, Linear4bit) for name, module in model.named_modules()), "Model should contain Linear4bit modules for nf4 quantization"
    assert model.config.quantization_config.bnb_4bit_quant_type == 'nf4', "Model quantization config should specify 'nf4' for bnb_4bit_quant_type"
    
def test_int8_forward_pass(tiny_mistral_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'int8',
        'load_in_8bit': True,
        'load_in_4bit': False
    }
    
    model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'int8')
    model.eval()
    
    inputs = tokenizer("Hello World", return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(input_ids = inputs['input_ids'],
                                 attention_mask = inputs['attention_mask'],
                                 max_length = 5)
        
        assert outputs.shape[0] == 1, "Output batch size should be 1"
        assert outputs.shape[1] > inputs['input_ids'].shape[1], "Output sequence length should be greater than input sequence length due to generation"
        assert torch.isnan(outputs).sum() == 0, "Generated output should not contain NaN values"
        
def test_nf4_forward_pass(tiny_mistral_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'nf4',
        'load_in_8bit': False,
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_use_double_quant': False,
    }
    
    model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'nf4')
    model.eval()
    
    inputs = tokenizer("Hello World", return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(input_ids = inputs['input_ids'],
                                 attention_mask = inputs['attention_mask'],
                                 max_length = 5)
        
        assert outputs.shape[0] == 1, "Output batch size should be 1"
        assert outputs.shape[1] > inputs['input_ids'].shape[1], "Output sequence length should be greater than input sequence length due to generation"
        assert torch.isnan(outputs).sum() == 0, "Generated output should not contain NaN values"
        
def test_invalid_ptq_mode(tiny_mistral_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'int8',
        'load_in_8bit': True,
        'load_in_4bit': False
    }
    ptq_mode = 'fp8'  # Invalid PTQ mode
    
    with pytest.raises(ValueError, match=f"Unsupported PTQ mode: {ptq_mode}"):
        load_ptq_model_and_tokenizer(model_config, ptq_mode_config, ptq_mode)
        
def test_gptq_quantize_config_creation():
    
    bits = 4
    group_size = 128
    desc_act = False
    
    quantize_config = BaseQuantizeConfig(bits = bits,
                                         group_size = group_size,
                                         desc_act = desc_act,
                        )
    
    assert quantize_config.bits == bits, "QuantizeConfig should have the correct number of bits"
    assert quantize_config.group_size == group_size, "QuantizeConfig should have the correct group size"
    assert quantize_config.desc_act == desc_act, "QuantizeConfig should have the correct desc_act value"
    
def test_gptq_quantize_and_save(gptq_quantized_checkpoint):
    
    bits = 4
    group_size = 128
    desc_act = False
    checkpoint_dir = gptq_quantized_checkpoint
        
    assert os.path.isdir(checkpoint_dir), "Checkpoint directory should be created after saving quantized model"
    assert os.path.isfile(os.path.join(checkpoint_dir, "quantize_config.json")), "Quantize config file should be saved in the checkpoint directory"
    assert os.path.isfile(os.path.join(checkpoint_dir, "tokenizer_config.json")), "Tokenizer config file should be saved in the checkpoint directory"
    
    with open(os.path.join(checkpoint_dir, "quantize_config.json"), "r") as f:
        saved_quantize_config = json.load(f)
        assert saved_quantize_config["bits"] == bits, "Saved quantize config should have the correct number of bits"
        assert saved_quantize_config['group_size'] == group_size, "Saved quantize config should have the correct group size"
        assert saved_quantize_config['desc_act'] == desc_act, "Saved quantize config should have the correct desc_act value"
        
    # Verify model weights file exists (safetensors or bin)
    has_weights = (os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or
                   any(f.endswith(".bin") for f in os.listdir(checkpoint_dir)) or
                   any(f.endswith(".safetensors") for f in os.listdir(checkpoint_dir)))
    assert has_weights, "Quantized model weights should be saved"
    
def test_gptq_load_from_quantized(gptq_quantized_checkpoint):
    
    trust_remote_code = False
    device_map = "auto"
    checkpoint_dir = gptq_quantized_checkpoint
    
    model = AutoGPTQForCausalLM.from_quantized(checkpoint_dir,
                                                device_map = device_map,
                                                trust_remote_code = trust_remote_code,
                    )
    
    assert model is not None, "Model should be loaded successfully from quantized checkpoint"
    assert next(model.parameters()).device.type == 'cuda', "Model parameters should be loaded on the correct device"
    
def test_gptq_forward_pass(gptq_quantized_checkpoint):

    trust_remote_code = False
    device_map = "auto"
    checkpoint_dir = gptq_quantized_checkpoint
    
    model = AutoGPTQForCausalLM.from_quantized(checkpoint_dir,
                                                device_map = device_map,
                                                trust_remote_code = trust_remote_code,
                    )
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(model.device)  # Example input IDs for generation
    
       
    outputs = model.generate(input_ids = input_ids,
                             max_length = 10)
    assert outputs.shape[0] == 1, "Output batch size should be 1"
    assert outputs.shape[1] > input_ids.shape[1], "Output sequence length should be greater than input sequence length due to generation"
    assert torch.isnan(outputs).sum() == 0, "Generated output should not contain NaN values"
    
def test_gptq_invalid_checkpoint_dir(tiny_mistral_path,tmp_path):

    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = {
        'name': 'gptq_int4',
        'checkpoint_dir': str(tmp_path / "non_existent_checkpoint"),
        'load_in_8bit': False,
        'load_in_4bit': True
    }
    checkpoint_dir_dummy = str(tmp_path / "non_existent_checkpoint")
    
    with pytest.raises(FileNotFoundError, match=f"Checkpoint directory for GPTQ quantized model not found: {checkpoint_dir_dummy}"):
        model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'gptq_int4')
        
def test_gptq_calibration_sample_format():

    tokenizer = AutoTokenizer.from_pretrained('assets/models/Mistral-7B-v0.3', 
                                              padding_side = 'left', 
                                              model_max_length = 1280,
                                              trust_remote_code = False)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default

    calib_samples = [
        "The cat sat on the mat.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The weather today is sunny with a chance of rain.",
        "OpenAI's GPT models are powerful language models."
    ]
    # Tokenize calibration samples for auto_gptq
    calib_tokenized = [tokenizer(s, return_tensors = "pt", 
                                 truncation = True,
                                 max_length = 128)
                        for s in calib_samples
                        ]

    for i, sample in enumerate(calib_tokenized):
        assert 'input_ids' in sample, f"Calibration sample {i} should contain 'input_ids' key"
        assert 'attention_mask' in sample, f"Calibration sample {i} should contain 'attention_mask' key"
        assert sample['input_ids'][0].shape[0] <= 128, f"Calibration sample {i} input_ids should be truncated to max length of 128"

def test_awq_config_loading():
    
    ptq_config = load_config("configs/ptq_config.yaml")
    
    awq_mode = ptq_config['ptq']['modes'][3]
    
    assert awq_mode['name'] == 'awq_int4', "Fourth PTQ mode should be 'awq_int4'"
    assert 'checkpoint_dir' in awq_mode, "AWQ PTQ mode should have 'checkpoint_dir' specified in the configuration"
    assert awq_mode['bits'] == 4, "AWQ PTQ mode should have 'bits' set to 4"
    assert awq_mode['group_size'] == 128, "AWQ PTQ mode should have 'group_size' set to 128"
    assert awq_mode['zero_point'] == True, "AWQ PTQ mode should have 'zero_point' set to True"
    assert awq_mode['version'] == 'GEMM', "AWQ PTQ mode should have 'version' set to 'GEMM'"
    
def test_awq_quantize_and_save(awq_quantized_checkpoint):
    
    checkpoint_dir = awq_quantized_checkpoint
    
    assert os.path.isdir(checkpoint_dir), "Checkpoint directory should be created after saving quantized model"
    assert os.path.isfile(os.path.join(checkpoint_dir, "config.json")), "Model config file should be saved in the checkpoint directory"
    assert os.path.isfile(os.path.join(checkpoint_dir, "generation_config.json")), "Generation config file should be saved in the checkpoint directory"
    
   
    # Verify model weights file exists (safetensors or bin)
    has_weights = (os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or
                   any(f.endswith(".bin") for f in os.listdir(checkpoint_dir)) or
                   any(f.endswith(".safetensors") for f in os.listdir(checkpoint_dir)))
    assert has_weights, "Quantized model weights should be saved"

def test_awq_load_from_quantized(awq_quantized_checkpoint):
    
    device_map = "auto"
    trust_remote_code = False
    checkpoint_dir = awq_quantized_checkpoint
    
    model = AutoAWQForCausalLM.from_quantized(checkpoint_dir,
                                              fuse_layers = False,
                                              device_map = device_map,
                                              trust_remote_code = trust_remote_code
                    )
    
    assert model is not None, "Model should be loaded successfully from quantized checkpoint"
    assert next(model.parameters()).device.type == 'cuda', "Model parameters should be loaded on the correct device"
    
def test_awq_forward_pass(awq_quantized_checkpoint):
    
    device_map = "auto"
    trust_remote_code = False
    checkpoint_dir = awq_quantized_checkpoint
    model = AutoAWQForCausalLM.from_quantized(checkpoint_dir,
                                              fuse_layers = False,
                                              device_map = device_map,
                                              trust_remote_code = trust_remote_code
                    )
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(next(model.parameters()).device) # Example input IDs for generation
    
       
    outputs = model.generate(input_ids = input_ids,
                             max_length = 10)
    
    assert outputs.shape[0] == 1, "Output batch size should be 1"
    assert outputs.shape[1] > input_ids.shape[1], "Output sequence length should be greater than input sequence length due to generation"
    assert torch.isnan(outputs).sum() == 0, "Generated output should not contain NaN values"
    
def test_awq_invalid_checkpoint_dir(tiny_mistral_path, tmp_path):
    
    model_config = load_config("configs/model_config.yaml")
    model_config['model']['name'] = str(tiny_mistral_path)
    
    ptq_mode_config = { 'name': 'awq_int4',
                        'checkpoint_dir': str(tmp_path / "non_existent_checkpoint"),
                        'bits': 4,
                        'group_size': 128,
                        'zero_point': True,
                        'version': 'GEMM'
    }
    
    checkpoint_dir_dummy = str(tmp_path / "non_existent_checkpoint")
    
    with pytest.raises(FileNotFoundError, match=f"Checkpoint directory for AWQ quantized model not found: {checkpoint_dir_dummy}. Please run the quantization step first to generate the quantized model checkpoint."):
        model, tokenizer = load_ptq_model_and_tokenizer(model_config, ptq_mode_config, 'awq_int4')