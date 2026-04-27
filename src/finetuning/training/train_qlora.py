import os
import torch
import logging
from datasets import Dataset
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
                         DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from ..utils import load_config
from .train_lora import prepare_dataset
from .lora_layer import inject_lora, save_lora_weights
from .callbacks import GPUMemoryCallback, QLoRACheckpointCallback

def load_qlora_model_and_tokenizer(model_config: dict, 
                                   train_qlora: dict,
                                   use_gradient_checkpointing: bool = True
        ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    """
    Description:
        Load a quantized language model and tokenizer with QLoRA configuration.
        This function initializes a causal language model using 4-bit quantization
        via BitsAndBytes and prepares it for efficient fine-tuning with QLoRA.
        It also loads and configures the corresponding tokenizer.
    Args:
        model_config (dict): Configuration dictionary containing model and tokenizer settings.
            Expected keys:
            - model.name (str): HuggingFace model identifier
            - model.trust_remote_code (bool): Whether to trust remote code from the model repo
            - tokenizer.name (str): HuggingFace tokenizer identifier
            - tokenizer.padding_side (str): Padding side ('left' or 'right')
            - tokenizer.max_length (int): Maximum sequence length
        train_qlora (dict): QLoRA training configuration dictionary.
            Expected keys:
            - quantization.load_in_4bit (bool): Whether to load model in 4-bit
            - quantization.bnb_4bit_quant_type (str): Quantization type ('nf4' or 'fp4')
            - quantization.bnb_4bit_use_double_quant (bool): Whether to use double quantization
            - quantization.bnb_4bit_compute_dtype (str): Compute dtype ('float16', 'bfloat16', or 'float32')
        use_gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing
            for memory efficiency. Defaults to True.
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing:
            - model: 4-bit quantized causal language model prepared for training
            - tokenizer: Configured tokenizer with pad token set to eos_token
    Note:
        The tokenizer's pad_token is automatically set to eos_token to handle models
        like Mistral that don't have a default pad token.
    """
    
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
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = use_gradient_checkpointing)
    # model.gradient_checkpointing_disable() # Disable gradient checkpointing for 4-bit models to avoid issues with the custom LoRA implementation
    
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
    
    
    '''
    Since we are using custom code to inject LoRA layers, we need to set a flag 
    on the model to indicate that the PEFT config has been loaded. 
    '''
    model._hf_peft_config_loaded = True
    
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
    args.save_strategy = "no"  # We will handle saving LoRA weights manually, so disable the default saving mechanism
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True)
    avg_seq_len = sum(len(x['input_ids']) for x in train_dataset) / len(train_dataset)
    
    gpu_cb = GPUMemoryCallback(avg_seq_len = avg_seq_len)

    trainer = Trainer(model = model,
                      args = args,              
                      train_dataset = train_dataset,
                      eval_dataset = valid_dataset,
                      data_collator = data_collator,
                      callbacks = [gpu_cb,
                                   QLoRACheckpointCallback()],
                )
    
    trainer.train()
    
    save_path = os.path.join(training_config['output_dir'], "lora_adapter.pt")
    save_lora_weights(model = model, save_path = save_path)
    
    # Save training performance report
    run_name = os.path.basename(training_config['output_dir'])
    gpu_cb.save_training_report(training_config['output_dir'], run_name)

def main():
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    train_qlora = load_config('configs/train_qlora.yaml')
    
    r = train_qlora['lora']['r']
    alpha = train_qlora['lora']['lora_alpha']
    lr = train_qlora['training']['learning_rate']
    use_gradient_checkpointing = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"qlora_r{r}_a{alpha}_lr{lr}_{timestamp}" if not use_gradient_checkpointing else f"qlora_r{r}_a{alpha}_lr{lr}_gc_{timestamp}" 
    
    train_qlora['training']['output_dir'] = f'outputs/checkpoints/{tag}'
    train_qlora['training']['logging_dir'] = f'outputs/runs/{tag}'
    
    os.makedirs(f'outputs/runs/{tag}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                                  
            logging.FileHandler(os.path.join(f'outputs/runs/{tag}', "training.log")),
        ]
    )
      
    model, tokenizer = load_qlora_model_and_tokenizer(model_config, train_qlora, use_gradient_checkpointing)
    
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