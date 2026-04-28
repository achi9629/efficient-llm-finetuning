import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils import load_config, run_inference, generate_run_name, save_predictions

def load_ptq_model_and_tokenizer(model_config: dict,
                                 ptq_config: dict,
                                 ptq_mode: str,
        ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    """
    Description:
        Load a pretrained language model and tokenizer with post-training quantization (PTQ).
        This function loads a causal language model and its corresponding tokenizer with
        quantization applied. It supports two quantization modes: int8 and nf4 (4-bit Normal Float).
    Args:
        model_config (dict): Configuration dictionary containing model and tokenizer settings.
            - model.name (str): Model identifier (e.g., model ID from Hugging Face Hub)
            - model.device_map (str or dict): Device mapping for model placement
            - model.trust_remote_code (bool): Whether to trust remote code for the model
            - tokenizer.name (str): Tokenizer identifier
            - tokenizer.padding_side (str): Side to pad sequences ('left' or 'right')
            - tokenizer.max_length (int): Maximum sequence length
        ptq_config (dict): Post-training quantization configuration.
            - load_in_8bit (bool): Whether to load model in 8-bit mode
            - load_in_4bit (bool): Whether to load model in 4-bit mode
            - bnb_4bit_quant_type (str, optional): Quantization type for 4-bit (required for 'nf4' mode)
            - bnb_4bit_compute_dtype (str, optional): Compute dtype for 4-bit operations (required for 'nf4' mode)
            - bnb_4bit_use_double_quant (bool, optional): Whether to use double quantization (required for 'nf4' mode)
        ptq_mode (str): Quantization mode to apply. One of:
            - 'int8': 8-bit integer quantization
            - 'nf4': 4-bit Normal Float quantization
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing:
            - AutoModelForCausalLM: The quantized language model
            - AutoTokenizer: The configured tokenizer with pad token set to EOS token
    Raises:
        ValueError: If ptq_mode is not 'int8' or 'nf4'
    """
                                
    dtype_map = {"float16": torch.float16, 
                 "bfloat16": torch.bfloat16, 
                 "float32": torch.float32}
    
    model_id = model_config['model']['name']
    device_map = model_config['model']['device_map']
    trust_remote_code = model_config['model']['trust_remote_code']
    token_name = model_config['tokenizer']['name']
    padding_side = model_config['tokenizer']['padding_side']
    max_length = model_config['tokenizer']['max_length']
    
    load_in_8bit = ptq_config['load_in_8bit']
    load_in_4bit = ptq_config['load_in_4bit']
    
    if ptq_mode == 'int8':
        
        bnb_config = BitsAndBytesConfig(load_in_8bit = load_in_8bit,
                                        load_in_4bit = load_in_4bit,
                        )
        
    elif ptq_mode == 'nf4':
        
        bnb_4bit_quant_type = ptq_config['bnb_4bit_quant_type']
        bnb_4bit_compute_dtype = dtype_map[ptq_config['bnb_4bit_compute_dtype']]
        bnb_4bit_use_double_quant = ptq_config['bnb_4bit_use_double_quant']
        
        bnb_config = BitsAndBytesConfig(load_in_8bit = load_in_8bit,
                                        load_in_4bit = load_in_4bit,
                                        bnb_4bit_quant_type = bnb_4bit_quant_type,
                                        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
                                        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
                        )
        
    else:
        raise ValueError(f"Unsupported PTQ mode: {ptq_mode}")
    
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config = bnb_config,
                                                 device_map = device_map)
    
    tokenizer = AutoTokenizer.from_pretrained(token_name, 
                                              padding_side = padding_side, 
                                              model_max_length = max_length,
                                              trust_remote_code = trust_remote_code)
    
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    return model, tokenizer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptq_mode", type=str, choices=["int8", "nf4"], required=True)
    args = parser.parse_args()
    
    model_config = load_config('configs/model_config.yaml')
    ptq_config = load_config('configs/ptq_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    ptq_mode = args.ptq_mode
    name = f"ptq_{ptq_mode}"
    ptq_mode_name = next(m for m in ptq_config['ptq']['modes'] if m['name'] == ptq_mode)
    
    model, tokenizer = load_ptq_model_and_tokenizer(model_config, 
                                                    ptq_mode_name, 
                                                    ptq_mode)
    model.eval()
    
    predictions = run_inference(model, 
                                tokenizer, 
                                data_config, 
                                eval_config)
    
    run_name = generate_run_name(name, 
                                 model_config['model']['name'],
                                 data_config['dataset']['name'])
    save_predictions(run_name, predictions, eval_config['output']['predictions_dir'])
    
if __name__ == "__main__":
    main()