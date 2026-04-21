import os
import torch
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
                         DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from ..utils import load_config
from .callbacks import GPUMemoryCallback
from .lora_layer import inject_lora, save_lora_weights
from .train_lora import prepare_dataset

def load_qlora_model_and_tokenizer(model_config: dict, train_qlora: dict):
    
    dtype_map = {"float16": torch.float16, 
                 "bfloat16": torch.bfloat16, 
                 "float32": torch.float32}
    
    model_id = model_config['model']['name']
    trust_remote_code = model_config['model']['trust_remote_code']
    token_name = model_config['tokenizer']['name']
    padding_side = model_config['tokenizer']['padding_side']
    max_length = model_config['tokenizer']['max_length']
    
    load_in_4bit = train_qlora['quantization']['load_in_4bit']
    bnb_4bit_quant_type = train_qlora['quantization']['bnb_4bit_quant_type']
    bnb_4bit_use_double_quant = train_qlora['quantization']['bnb_4bit_use_double_quant']
    bnb_4bit_compute_dtype = dtype_map[train_qlora['quantization']['bnb_4bit_compute_dtype']]
    
    bnb_config = BitsAndBytesConfig(load_in_4bit = load_in_4bit,
                                    bnb_4bit_quant_type = bnb_4bit_quant_type,
                                    bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
                                    bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
                    )
    
    
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config = bnb_config,
                                                 device_map = {"": 0},
                                                 trust_remote_code = trust_remote_code,
                )
    
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(token_name, 
                                              padding_side = padding_side, 
                                              model_max_length = max_length,
                                              trust_remote_code = trust_remote_code)
    
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
    
    return model, tokenizer

def build_qlora_model(model: AutoModelForCausalLM, 
                     train_qlora: dict
    ) -> AutoModelForCausalLM:
    
    """
    Description:
        Build a QLoRA-configured model by injecting LoRA (Low-Rank Adaptation) parameters.
        This function extracts LoRA configuration from the training configuration dictionary
        and applies LoRA adaptations to the provided model using the specified target modules
        and hyperparameters.
    Args:
        model (AutoModelForCausalLM): The base causal language model to be adapted with LoRA.
        train_qlora (dict): Configuration dictionary containing LoRA parameters with the structure:
            {
                'lora': {
                    'target_modules': list,      # Modules to apply LoRA to
                    'r': int,                    # LoRA rank
                    'lora_alpha': float,         # LoRA scaling factor
                    'lora_dropout': float        # Dropout rate for LoRA layers
                }
            }
    Returns:
        AutoModelForCausalLM: The model with LoRA adaptations injected into the specified target modules.
    Example:
        >>> config = {
        ...     'lora': {
        ...         'target_modules': ['q_proj', 'v_proj'],
        ...         'r': 16,
        ...         'lora_alpha': 32,
        ...         'lora_dropout': 0.05
        ...     }
        ... }
        >>> qlora_model = build_qlora_model(base_model, config)
    """
                   
    target_modules = train_qlora['lora']['target_modules']
    r = train_qlora['lora']['r']
    alpha = train_qlora['lora']['lora_alpha']
    dropout = train_qlora['lora']['lora_dropout']
        
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = dropout
            )
    
    return model

def train(model: AutoModelForCausalLM,
          tokenizer: AutoTokenizer,
          train_dataset: Dataset, 
          valid_dataset: Dataset, 
          train_qlora: dict,) -> None:
    
    """
    Description:
        Train a language model using LoRA (Low-Rank Adaptation) fine-tuning.
        This function configures and executes the training process for a causal language model
        with LoRA adapters applied. It handles dataset preparation, training configuration,
        and saves the trained LoRA weights to disk.
    Args:
        model (AutoModelForCausalLM): The pre-trained causal language model to fine-tune.
        tokenizer (AutoTokenizer): The tokenizer associated with the model for processing text.
        train_dataset (Dataset): The training dataset containing prepared examples.
        valid_dataset (Dataset): The validation dataset for evaluating model performance during training.
        train_lora (dict): Configuration dictionary containing training parameters and output paths.
            Expected to have a 'training' key with TrainingArguments-compatible parameters
            and an 'output_dir' key for the model save location.
    Returns:
        None
    Raises:
        None explicitly, but may raise exceptions from trainer.train() or save_lora_weights().
    """
    
    training_config = train_qlora['training']
    args = TrainingArguments(**training_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True)
    avg_seq_len = sum(len(x['input_ids']) for x in train_dataset) / len(train_dataset)
    
    trainer = Trainer(model = model,
                      args = args,              
                      train_dataset = train_dataset,
                      eval_dataset = valid_dataset,
                      data_collator = data_collator,
                      callbacks = [GPUMemoryCallback(avg_seq_len = avg_seq_len)],
                )
    
    trainer.train()
    
    save_path = os.path.join(training_config['output_dir'], "lora_adapter.pt")
    save_lora_weights(model = model, save_path = save_path)

def main():
    
    os.makedirs("outputs/runs/qlora", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                                    # console
            logging.FileHandler("outputs/runs/qlora/training.log"),      # file
        ]
    )
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    train_qlora = load_config('configs/train_qlora.yaml')
    
    model, tokenizer = load_qlora_model_and_tokenizer(model_config, train_qlora)
    
    # During training, we will pad on the right side to ensure that the loss is correctly 
    # calculated on the target tokens, which are at the end of the sequence.
    tokenizer.padding_side = "right"
    # Set a very large max_length to avoid truncation by the tokenizer, 
    # since we will handle it in our tokenize_fn with separate budgets for prompt and target.
    tokenizer.model_max_length = 100_000 
    
    train_dataset, valid_dataset = prepare_dataset(tokenizer, data_config)
    model = build_qlora_model(model, train_qlora)
    train(model, tokenizer, train_dataset, valid_dataset, train_qlora)
    
if __name__ == "__main__":
    main()