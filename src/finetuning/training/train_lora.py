import os
import logging
from functools import partial
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, \
                         Trainer, DataCollatorForSeq2Seq
from .callbacks import GPUMemoryCallback
from .lora_layer import inject_lora, save_lora_weights
from ..utils import load_model_and_tokenizer, load_config


def tokenize_fn(sample: dict,
                tokenizer: AutoTokenizer, 
                data_config: dict
    ) -> dict:
    
    """
    Description:
        Tokenize a sample for fine-tuning with LoRA, separating prompt and target tokens.
        This function constructs a prompt-target pair from the input sample, tokenizes them
        in a single pass to avoid BPE boundary artifacts, and applies independent length budgets
        to prompts and targets. The resulting tensors mask prompt tokens in the labels to
        ensure only target tokens contribute to the loss.
    Args:
        sample (dict): A dictionary containing the input data with keys matching
                        'input_field' and 'target_field' from data_config.
        tokenizer (AutoTokenizer): A pretrained tokenizer (e.g., from transformers library)
                                    used to encode text and access special tokens.
        data_config (dict): Configuration dictionary with the following nested structure:
                            - dataset.input_field (str): Key name for input text in sample
                            - dataset.target_field (str): Key name for target text in sample
                            - preprocessing.prompt_template (str): Template string with {article}
                                                                    placeholder for formatting
                            - preprocessing.max_input_length (int): Maximum token budget for prompt
                            - preprocessing.max_target_length (int): Maximum token budget for target
    Returns:
        dict: A dictionary with the following keys:
                - 'input_ids' (list): Token IDs [bos_token_id, prompt_ids, target_ids],
                                    truncated according to individual budgets
                - 'labels' (list): Loss mask with -100 for bos and prompt tokens,
                                    and target token IDs for target portion
                - 'attention_mask' (list): Binary mask of 1s indicating valid tokens
    Notes:
        - Prompt and target are tokenized together to prevent BPE boundary issues
        - Character offsets are used to precisely locate the token boundary between
            prompt and target
        - Prompt tokens are masked (-100) in labels to exclude them from loss calculation
        - If prompt exceeds max_input_length, it is truncated; target remains intact up to max_target_length
    """
    
    input_field = data_config['dataset']['input_field']
    target_field = data_config['dataset']['target_field']
    prompt_template = data_config['preprocessing']['prompt_template']
    max_prompt_len = data_config['preprocessing']['max_input_length']   # 1024
    max_target_len = data_config['preprocessing']['max_target_length']  # 256
    
    prompt = prompt_template.format(article=sample[input_field])
    target = sample[target_field] + tokenizer.eos_token
    full_text = prompt + target
    
    # 1. Single tokenizer call on the full text — zero BPE boundary artifact
    enc = tokenizer(full_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True)
    ids = enc['input_ids']
    offsets = enc['offset_mapping'] # [(char_start, char_end), ...]
    
    # 2. Find exact token index where target begins, using character offset
    prompt_char_len = len(prompt)

    # Equivalent explicit loop:
    # for i, (s, e) in enumerate(offsets):
    #     if s >= prompt_char_len:
    #         prompt_end_idx = i
    #         break
    
    # Using next():
    prompt_end_idx = next(i for i, (s, e) in enumerate(offsets) if s >= prompt_char_len)
    
    # 3. Independent budget truncation — article absorbs the cut, target stays intact
    prompt_ids = ids[ : min(prompt_end_idx, max_prompt_len)]
    target_ids = ids[prompt_end_idx : prompt_end_idx + max_target_len]
    
    # 4. Prepend bos, build labels with prompt masked
    input_ids = [tokenizer.bos_token_id] + prompt_ids + target_ids
    labels = [-100] * (1 + len(prompt_ids)) + target_ids
    attention_mask = [1] * len(input_ids)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask}

def prepare_dataset(tokenizer: AutoTokenizer, 
                    data_config: dict
    ) -> tuple[Dataset, Dataset]:
    
    """
    Description:
        Prepare and tokenize a dataset for model training.
        Loads a dataset from the specified configuration and applies tokenization
        using the provided tokenizer. Processing is parallelized across multiple
        processes for efficiency.
    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for encoding text samples.
        data_config (dict): Configuration dictionary containing dataset specifications with the following structure:
            {
                'dataset': {
                    'name': str - Name/path of the dataset,
                    'config': str - Configuration identifier for the dataset,
                    'train_split': str - Dataset split to load (e.g., 'train', 'validation')
                }
            }
    Returns:
        Dataset: Tokenized dataset with original columns removed and token sequences added.
    Note:
        - Uses multiprocessing (num_proc=4) for efficient parallel tokenization
        - Original dataset columns are removed and replaced with tokenized features
        - Dataset is loaded from 'assets/datasets/{name}/{config}' directory
    """
    
    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    train_split = data_config['dataset']['train_split']
    valid_split = data_config['dataset']['val_split']
    
    dataset_train = load_dataset(os.path.join('assets/datasets', 
                                              dataset_name, 
                                              dataset_config), 
                                split = train_split
                        )
    
    dataset_valid = load_dataset(os.path.join('assets/datasets', 
                                              dataset_name,
                                              dataset_config),
                                split = valid_split
                        )
    
    # This will also work, but is less efficient due to separate tokenization calls 
    # and no multiprocessing:
    # dataset_tokenized = []
    # for sample in dataset:
    #     dataset_tokenized.append(tokenize_fn(sample, tokenizer, data_config))
    # dataset_tokenized = Dataset.from_list(dataset_tokenized)
    
    tokenize = partial(tokenize_fn, tokenizer = tokenizer, data_config = data_config)
    
    dataset_tokenized_train = dataset_train.map(tokenize,
                                          remove_columns = dataset_train.column_names,
                                          num_proc = 4
                                )
    
    dataset_tokenized_valid = dataset_valid.map(tokenize,
                                          remove_columns = dataset_valid.column_names,
                                          num_proc = 4
                                )
    
    return dataset_tokenized_train, dataset_tokenized_valid

def build_lora_model(model: AutoModelForCausalLM, 
                     train_lora: dict
    ) -> AutoModelForCausalLM:
    
    """
    Description:
        Build a LoRA (Low-Rank Adaptation) fine-tuned model by freezing all base model parameters
        and injecting LoRA adapters into specified target modules.
    Args:
        model (AutoModelForCausalLM): The base causal language model to be adapted with LoRA.
        train_lora (dict): Configuration dictionary containing LoRA hyperparameters with the following structure:
            {
                'lora': {
                    'target_modules': List[str] - Names of model modules to apply LoRA adapters to,
                    'r': int - Rank of the LoRA decomposition,
                    'lora_alpha': int - Scaling factor for LoRA updates,
                    'lora_dropout': float - Dropout probability for LoRA layers
                }
            }
    Returns:
        AutoModelForCausalLM: The model with all base parameters frozen and LoRA adapters injected
                                into the specified target modules.
    Note:
        - All original model parameters are frozen (requires_grad=False) before LoRA injection
        - Only LoRA adapter parameters will be trainable after this function returns
    """
    
    target_modules = train_lora['lora']['target_modules']
    r = train_lora['lora']['r']
    alpha = train_lora['lora']['lora_alpha']
    dropout = train_lora['lora']['lora_dropout']
    
    # Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
        
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
          train_lora: dict,) -> None:
    
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
    
    training_config = train_lora['training']
    args = TrainingArguments(**training_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True)
    avg_seq_len = sum(len(x['input_ids']) for x in train_dataset) / len(train_dataset)
    
    gpu_cb = GPUMemoryCallback(avg_seq_len = avg_seq_len)
    
    trainer = Trainer(model = model,
                      args = args,              
                      train_dataset = train_dataset,
                      eval_dataset = valid_dataset,
                      data_collator = data_collator,
                      callbacks = [gpu_cb],
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
    train_lora = load_config('configs/train_lora.yaml')
    
    r = train_lora['lora']['r']
    alpha = train_lora['lora']['lora_alpha']
    lr = train_lora['training']['learning_rate']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"lora_r{r}_a{alpha}_lr{lr}_{timestamp}"
    
    train_lora['training']['output_dir'] = f'outputs/checkpoints/{tag}'
    train_lora['training']['logging_dir'] = f'outputs/runs/{tag}'
    
    os.makedirs(f'outputs/runs/{tag}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                                    # console
            logging.FileHandler(os.path.join(f'outputs/runs/{tag}', "training.log")),      # file
        ]
    )
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.train()
    
    # During training, we will pad on the right side to ensure that the loss is correctly 
    # calculated on the target tokens, which are at the end of the sequence.
    tokenizer.padding_side = "right"
    # Set a very large max_length to avoid truncation by the tokenizer, 
    # since we will handle it in our tokenize_fn with separate budgets for prompt and target.
    tokenizer.model_max_length = 100_000 
    
    train_dataset, valid_dataset = prepare_dataset(tokenizer, data_config)
    model = build_lora_model(model, train_lora)
    train(model, tokenizer, train_dataset, valid_dataset, train_lora)
    
if __name__ == "__main__":
    main()