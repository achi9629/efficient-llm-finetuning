import torch
import pytest
import logging
from transformers import MistralConfig, MistralForCausalLM
from bitsandbytes.nn import Linear8bitLt, Linear4bit

from finetuning import load_config, load_ptq_model_and_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def tiny_mistral_path(tmp_path_factory):
    
    tmp_dir = tmp_path_factory.mktemp("tiny_mistral")
    config = MistralConfig(vocab_size = 32768,
                            hidden_size=64,
                            intermediate_size=128,
                            num_hidden_layers=2,
                            num_attention_heads=2,
                            num_key_value_heads=2,
                            max_position_embeddings=128,
                )
    model = MistralForCausalLM(config)
    model.save_pretrained(tmp_dir)
    
    return tmp_dir

def test_ptq_config_loading():
    
    ptq_config = load_config("configs/ptq_config.yaml")
    
    assert len(ptq_config['ptq']['modes']) == 2, "Expected 2 PTQ modes in the configuration"
    assert ptq_config['ptq']['modes'][0]['name'] == 'int8', "First PTQ mode should be 'int8'"
    assert ptq_config['ptq']['modes'][0]['load_in_8bit'] == True, "First PTQ mode should have 'load_in_8bit' set to True"
    assert ptq_config['ptq']['modes'][0]['load_in_4bit'] == False, "First PTQ mode should have 'load_in_4bit' set to False"
    assert ptq_config['ptq']['modes'][1]['name'] == 'nf4', "Second PTQ mode should be 'nf4'"
    assert ptq_config['ptq']['modes'][1]['load_in_8bit'] == False, "Second PTQ mode should have 'load_in_8bit' set to False"
    assert ptq_config['ptq']['modes'][1]['load_in_4bit'] == True, "Second PTQ mode should have 'load_in_4bit' set to True"
    assert ptq_config['ptq']['modes'][1]['bnb_4bit_quant_type'] == 'nf4', "Second PTQ mode should have 'bnb_4bit_quant_type' set to 'nf4'"
    
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