import os
import torch
import argparse
import peft.peft_model
from datasets import load_dataset
if not hasattr(peft.peft_model, 'PEFT_TYPE_TO_MODEL_MAPPING'):
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING = {}
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils import load_config, run_inference, generate_run_name, save_predictions

def quantize_gptq_and_save(model_config: dict, 
                           data_config: dict,
                           ptq_config: dict
        ) -> None:
    
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
    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    test_split = data_config['dataset']['val_split']
    input_field = data_config['dataset']['input_field']
    prompt_template = data_config['preprocessing']['prompt_template']
    bits = ptq_config['bits']
    group_size = ptq_config['group_size']
    desc_act = ptq_config['desc_act']
    checkpoint_dir = ptq_config['checkpoint_dir']
    
    tokenizer = AutoTokenizer.from_pretrained(token_name, 
                                              padding_side = padding_side, 
                                              model_max_length = max_length,
                                              trust_remote_code = trust_remote_code)
    
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    # Load your local CNN/DailyMail dataset for calibration
    calib_dataset = load_dataset(os.path.join('assets/datasets', 
                                              dataset_name, 
                                              dataset_config),
                                split = test_split
                        )

    # Take ~128 samples, format as plain text strings
    calib_samples = [
        prompt_template.format(article=sample[input_field])
        for sample in calib_dataset.select(range(128))
    ]
    
    quantize_config = BaseQuantizeConfig(bits = bits,
                                         group_size = group_size,
                                         desc_act = desc_act,
                        )
    
    model = AutoGPTQForCausalLM.from_pretrained(model_id,
                                                quantize_config = quantize_config,
                                                torch_dtype = torch_dtype,
                                                trust_remote_code = trust_remote_code,
                                                device_map = device_map,
                )
    
    # Tokenize calibration samples for auto_gptq
    calib_tokenized = [tokenizer(s, return_tensors="pt", 
                                 truncation=True,
                                 max_length=max_length)
                        for s in calib_samples
                        ]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.quantize(calib_tokenized)
    model.save_quantized(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
                                   
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
            - 'gptq_int4': 4-bit quantization using GPTQ method (model must already be quantized and saved locally)
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing:
            - AutoModelForCausalLM: The quantized language model
            - AutoTokenizer: The configured tokenizer with pad token set to EOS token
    Raises:
        ValueError: If ptq_mode is not 'int8' or 'nf4' or 'gptq_int4'
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
    
    if ptq_mode == 'int8':
        load_in_8bit = ptq_config['load_in_8bit']
        load_in_4bit = ptq_config['load_in_4bit']
        bnb_config = BitsAndBytesConfig(load_in_8bit = load_in_8bit,
                                        load_in_4bit = load_in_4bit,
                        )
        
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config = bnb_config,
                                                     device_map = device_map,
                                                     trust_remote_code = trust_remote_code)
    elif ptq_mode == 'nf4':
        load_in_8bit = ptq_config['load_in_8bit']
        load_in_4bit = ptq_config['load_in_4bit']
        bnb_4bit_quant_type = ptq_config['bnb_4bit_quant_type']
        bnb_4bit_compute_dtype = dtype_map[ptq_config['bnb_4bit_compute_dtype']]
        bnb_4bit_use_double_quant = ptq_config['bnb_4bit_use_double_quant']
        
        bnb_config = BitsAndBytesConfig(load_in_8bit = load_in_8bit,
                                        load_in_4bit = load_in_4bit,
                                        bnb_4bit_quant_type = bnb_4bit_quant_type,
                                        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
                                        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
                        )
        
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config = bnb_config,
                                                     device_map = device_map,
                                                     trust_remote_code = trust_remote_code)
    elif ptq_mode == 'gptq_int4':
        
        checkpoint_dir = ptq_config['checkpoint_dir']
        model_id = checkpoint_dir

        model = AutoGPTQForCausalLM.from_quantized(model_id,
                                                   device_map = device_map,
                                                   trust_remote_code = trust_remote_code,
                    )

    else:
        raise ValueError(f"Unsupported PTQ mode: {ptq_mode}")

    tokenizer = AutoTokenizer.from_pretrained(token_name, 
                                              padding_side = padding_side, 
                                              model_max_length = max_length,
                                              trust_remote_code = trust_remote_code)
    
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    return model, tokenizer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptq_mode", type=str, choices=["int8", "nf4", "gptq_int4"], required=True)
    args = parser.parse_args()
    
    model_config = load_config('configs/model_config.yaml')
    ptq_config = load_config('configs/ptq_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    ptq_mode = args.ptq_mode
    name = f"ptq_{ptq_mode}"
    ptq_mode_name = next(m for m in ptq_config['ptq']['modes'] if m['name'] == ptq_mode)
    
    if ptq_mode == 'gptq_int4' and not os.path.exists(ptq_mode_name['checkpoint_dir']):
        quantize_gptq_and_save(model_config, data_config, ptq_mode_name)
    
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