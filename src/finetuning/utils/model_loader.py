import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    """
    Description:
        Load a pretrained causal language model and its corresponding tokenizer.
        This function initializes a model and tokenizer from the Hugging Face Model Hub
        based on the provided configuration. The model is set to evaluation mode, and
        the tokenizer is configured with a padding token (using EOS token as fallback).
    Args:
        model_config (dict): Configuration dictionary containing model parameters with
            the following nested attributes:
            - model.name (str): The model identifier/name on Hugging Face Model Hub
            - model.torch_dtype (torch.dtype): Data type for model weights (e.g., float16, bfloat16)
            - model.device_map (str or dict): Device placement strategy for model layers
              (e.g., 'auto', 'cuda', 'cpu')
            - model.trust_remote_code (bool): Whether to execute code from the model repository
              that isn't part of the official Hugging Face transformers library
            - model.padding_side (str): Which side to pad sequences ('left' or 'right')
            - model.max_length (int): Maximum sequence length for tokenizer
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing:
            - AutoModelForCausalLM: The loaded pretrained causal language model in eval mode
            - AutoTokenizer: The loaded tokenizer configured with specified padding and
              max length parameters
    """
    
    dtype_map = {"float16": torch.float16, 
                 "bfloat16": torch.bfloat16, 
                 "float32": torch.float32}
    model_id = model_config['model']['name']
    torch_dtype = dtype_map[model_config['model']['torch_dtype']]
    device_map = model_config['model']['device_map']
    trust_remote_code = model_config['model']['trust_remote_code']
    token_name = model_config['tokenizer']['name']
    padding_side = model_config['tokenizer']['padding_side']
    max_length = model_config['tokenizer']['max_length']
    
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 torch_dtype = torch_dtype, 
                                                 device_map = device_map, 
                                                 trust_remote_code = trust_remote_code)
    
    tokenizer = AutoTokenizer.from_pretrained(token_name, 
                                              padding_side = padding_side, 
                                              model_max_length = max_length,
                                              trust_remote_code = trust_remote_code)
    
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    return model, tokenizer