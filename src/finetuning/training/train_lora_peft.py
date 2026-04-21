import os
import logging
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from .train_lora import prepare_dataset
from .callbacks import GPUMemoryCallback
from ..utils import load_config, load_model_and_tokenizer

def main():
    
    os.makedirs("outputs/runs/lora_peft", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                                    # console
            logging.FileHandler("outputs/runs/lora_peft/training.log"),      # file
        ]
    )
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    train_lora = load_config('configs/train_lora.yaml')
    
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.train()
    
    # During training, we will pad on the right side to ensure that the loss is correctly 
    # calculated on the target tokens, which are at the end of the sequence.
    tokenizer.padding_side = "right"
    # Set a very large max_length to avoid truncation by the tokenizer, 
    # since we will handle it in our tokenize_fn with separate budgets for prompt and target.
    tokenizer.model_max_length = 100_000 
    
    train_dataset, valid_dataset = prepare_dataset(tokenizer, data_config)
    avg_seq_len = sum(len(x['input_ids']) for x in train_dataset) / len(train_dataset)
    
    # Configure the LoRA parameters and wrap the model with the PEFT framework
    peft_config = LoraConfig(**train_lora['lora'])
    model = get_peft_model(model, peft_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True)
    
    # Update the output and logging directories in the training configuration
    training_config = train_lora['training']
    training_config['output_dir'] = 'outputs/checkpoints/lora_peft'
    training_config['logging_dir'] = 'outputs/runs/lora_peft'
    args = TrainingArguments(**training_config)
    
    trainer = Trainer(model = model,
                      args = args,              
                      train_dataset = train_dataset,
                      eval_dataset = valid_dataset,
                      data_collator = data_collator,
                      callbacks = [GPUMemoryCallback(avg_seq_len = avg_seq_len)],
                )
    trainer.train()
    
    # Save the fine-tuned model and tokenizer to the specified output directory
    model.save_pretrained(training_config['output_dir'])
    
if __name__ == "__main__":
    main()