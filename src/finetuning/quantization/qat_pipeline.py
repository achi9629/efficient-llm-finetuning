import os
import argparse
import pandas as pd
import torch.nn as nn
from datasets import Dataset
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, \
                        Trainer, DataCollatorForSeq2Seq

from .fake_quant_layer import FakeQuantLinear
from ..training import prepare_dataset, GPUMemoryCallback
from ..utils import load_config, load_model_and_tokenizer

def load_qat_targets(qat_config: dict
        ) -> tuple[list[int], list[int]]:
    
    '''
    Description:
        Forward pass with simulated INT-N quantization on weights using symmetric
        per-channel (per-output-row) quantization scheme and STE for gradient flow.

    Args:
        x (torch.Tensor): Input activation tensor of shape (batch, seq_len, in_features).

    Returns:
        torch.Tensor: Output of shape (batch, seq_len, out_features), computed as
            F.linear(x, w_fake, bias) where w_fake uses quantized weights in the
            forward path but routes gradients through the original weights.

    Note:
        - Scale is computed per output channel: scale_i = max(|W_i|) / (2^(bits-1) - 1)
        - STE trick: w_fake = quant(W).detach() + W - W.detach()
            Forward sees quantized weights; backward sees identity gradient to W.
    '''
    
    target_path = qat_config['sensitivity_path']
    block_top_k = qat_config['block_top_k']
    module_top_k = qat_config['module_top_k']
    
    per_block_path = os.path.join(target_path, 'sensitivity_per_block.csv')
    per_module_path = os.path.join(target_path, 'sensitivity_per_module.csv')
    
    target_blocks = pd.read_csv(per_block_path, index_col=0)['block_idx'].iloc[:block_top_k].tolist()
    target_modules = pd.read_csv(per_module_path, index_col=0)['module_type'].iloc[:module_top_k].tolist()
    
    return target_blocks, target_modules

def insert_fake_quant_observers(model: AutoModelForCausalLM,
                                target_blocks: list[int],
                                target_modules: list[int],
                                bits: int
        ) -> AutoModelForCausalLM:
    
    '''
    Description:
        Forward pass with simulated INT-N quantization on weights using symmetric
        per-channel (per-output-row) quantization scheme and STE for gradient flow.

    Args:
        x (torch.Tensor): Input activation tensor of shape (batch, seq_len, in_features).

    Returns:
        torch.Tensor: Output of shape (batch, seq_len, out_features), computed as
            F.linear(x, w_fake, bias) where w_fake uses quantized weights in the
            forward path but routes gradients through the original weights.

    Note:
        - Scale is computed per output channel: scale_i = max(|W_i|) / (2^(bits-1) - 1)
        - STE trick: w_fake = quant(W).detach() + W - W.detach()
            Forward sees quantized weights; backward sees identity gradient to W.
    '''
    
    for name, module in model.named_modules():
        parts = name.split('.')
        if len(parts) > 2 and \
            int(parts[2]) in target_blocks and \
            parts[-1] in target_modules and \
                isinstance(module, nn.Linear):
            
            parent_name, child_name = '.'.join(parts[:-1]), parts[-1]
            parent_module = model.get_submodule(parent_name)
            fake_layer = FakeQuantLinear(module, bits=bits)
            setattr(parent_module, child_name, fake_layer)
            
    return model

def freeze_non_target_params(model: AutoModelForCausalLM, 
                             target_blocks: list[int], 
                             target_modules: list[int]
        ) -> AutoModelForCausalLM:
    
    '''
    Description:
        Freeze all model parameters except those in the target blocks and module types.
        Only parameters whose path contains a target block index AND a target module name
        will have requires_grad=True. All other parameters (embeddings, layernorm, lm_head,
        non-target blocks) are frozen.

    Args:
        model (AutoModelForCausalLM): The model with fake-quant observers already inserted.
        target_blocks (list[int]): Transformer block indices whose params should be trainable.
        target_modules (list[int]): Module type names whose params should be trainable.

    Returns:
        AutoModelForCausalLM: The same model with non-target parameters frozen.

    Note:
        - After FakeQuantLinear insertion, parameter paths become e.g.
          model.layers.14.mlp.down_proj.linear.weight — the 'down_proj' substring
          still appears in the path and is matched by `any(p in target_modules for p in parts)`.
        - Typical trainable param count: ~226M for 3 blocks × 2 module types on Mistral-7B.
    '''
    
    for name, param in model.named_parameters():
        parts = name.split('.')
        if len(parts) > 3 and \
            int(parts[2]) in target_blocks and \
            any(p in target_modules for p in parts):
            
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model

def train(model: AutoModelForCausalLM,
          tokenizer: AutoTokenizer,
          train_dataset: Dataset,
          qat_config: dict
        ) -> None:
    
    '''
    Description:
        Run QAT fine-tuning using HuggingFace Trainer. The model's FakeQuantLinear layers
        inject quantization noise during forward pass; the STE allows gradients to update
        the underlying weights so they become robust to INT-N rounding. After training,
        fake-quant wrappers are stripped and the adapted model is saved.

    Args:
        model (AutoModelForCausalLM): Model with fake-quant observers and frozen non-target params.
        tokenizer (AutoTokenizer): Tokenizer with padding_side='right' for training.
        train_dataset (Dataset): Tokenized training dataset with input_ids, attention_mask, labels.
        qat_config (dict): Full QAT config containing qat.training with TrainingArguments fields.

    Returns:
        None. Saves the exported model (fake-quant stripped) and training performance report
        to the output_dir specified in qat_config['qat']['training']['output_dir'].
    '''
    
    training_config = qat_config['qat']['training']
    args = TrainingArguments(**training_config)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True)
    avg_seq_len = sum(len(x['input_ids']) for x in train_dataset) / len(train_dataset)
    
    gpu_cb = GPUMemoryCallback(avg_seq_len = avg_seq_len)
    
    trainer = Trainer(model = model,
                      args = args,              
                      train_dataset = train_dataset,
                      data_collator = data_collator,
                      callbacks = [gpu_cb],
                )

    trainer.train()
    
    # Save the fine-tuned model and tokenizer to the specified output directory
    export_quantized_model(model, training_config['output_dir'])
    
    # Save training performance report
    run_name = os.path.basename(training_config['output_dir'])
    gpu_cb.save_training_report(training_config['output_dir'], run_name)
    
def export_quantized_model(model: AutoModelForCausalLM,
                           output_dir: str
        ) -> None:
    
    '''
    Description:
        Strip FakeQuantLinear wrappers and save the adapted model. After QAT training,
        the target layers' weights have learned to be robust to quantization noise.
        This function unwraps FakeQuantLinear → nn.Linear and saves the full model
        in HuggingFace format for downstream PTQ (GPTQ/AWQ) or direct deployment.

    Args:
        model (AutoModelForCausalLM): The QAT-trained model with FakeQuantLinear wrappers.
        output_dir (str): Directory to save the unwrapped model via model.save_pretrained().

    Returns:
        None. Saves model weights and config to output_dir.

    Note:
        - The saved model is still in floating point (bf16). To get actual INT4 weights,
          apply PTQ (GPTQ/AWQ) on top — the QAT-adapted weights will quantize with less
          quality loss than the original weights.
    '''
    
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinear):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, module.linear)
            
    model.save_pretrained(output_dir)
    
def calculate_steps(qat_config: dict, 
        ) -> str:
    
    '''
    Description:
        Calculate the total number of optimizer steps for the QAT training run.
        If max_steps is specified in config, uses that directly. Otherwise computes
        from dataset size, batch size, gradient accumulation, and number of epochs.

    Args:
        qat_config (dict): QAT sub-config containing:
            - training.max_steps (int, optional): Override total steps directly
            - training.per_device_train_batch_size (int): Batch size per GPU
            - training.gradient_accumulation_steps (int): Gradient accumulation factor
            - training.num_train_epochs (int): Number of training epochs
            - dataset.train_split (str): HuggingFace split string (e.g., "train[:2000]")

    Returns:
        int: Total number of optimizer steps for the training run.
    '''
    
    if 'max_steps' in qat_config['training']:
        return qat_config['training']['max_steps']
    
    split_str = qat_config['dataset']['train_split']
    per_device_train_batch_size = qat_config['training']['per_device_train_batch_size']
    gradient_accumulation_steps = qat_config['training']['gradient_accumulation_steps']
    epochs = qat_config['training']['num_train_epochs']
    num_samples = int(split_str.split(':')[1].rstrip(']'))  # 2000
    
    steps = (num_samples // per_device_train_batch_size) // gradient_accumulation_steps * epochs
    return steps
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_path", type=str, default='assets/models/lora_peft_merged_r2_mistral_7b_v0.3', help="Path to save merged LoRA model (if using LoRA PEFT)")
    args = parser.parse_args()
    
    qat_config = load_config('configs/qat_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    
    block_top_k = qat_config['qat']['block_top_k']
    module_top_k = qat_config['qat']['module_top_k']
    bits = qat_config['qat']['bits']
    steps = calculate_steps(qat_config['qat'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"qat_block{block_top_k}_module{module_top_k}_{bits}bits_steps{steps}_{timestamp}"
    
    qat_config['qat']['training']['output_dir'] = f'outputs/checkpoints/{tag}'
    qat_config['qat']['training']['logging_dir'] = f'outputs/runs/{tag}'
    
    target_blocks, target_modules = load_qat_targets(qat_config['qat'])
    
    model_config['model']['name'] = args.lora_model_path
    model_config['model']['torch_dtype'] = qat_config['qat']['model']['torch_dtype']
    model, tokenizer = load_model_and_tokenizer(model_config = model_config)
    
    # During training, we will pad on the right side to ensure that the loss is correctly 
    # calculated on the target tokens, which are at the end of the sequence.
    tokenizer.padding_side = "right"
    # Set a very large max_length to avoid truncation by the tokenizer, 
    # since we will handle it in our tokenize_fn with separate budgets for prompt and target.
    tokenizer.model_max_length = 100_000 
    
    bits = qat_config['qat']['bits']
    model = insert_fake_quant_observers(model = model, 
                                target_blocks = target_blocks, 
                                target_modules = target_modules, 
                                bits = bits)
    model = freeze_non_target_params(model = model,
                                    target_blocks = target_blocks,
                                    target_modules = target_modules)
    
    data_config['dataset']['train_split'] = qat_config['qat']['dataset']['train_split']
    data_config['dataset']['val_split'] = qat_config['qat']['dataset']['val_split']
    train_dataset, _ = prepare_dataset(tokenizer = tokenizer, 
                                        data_config = data_config
                        )
    
    train(model = model,
          tokenizer = tokenizer,
          train_dataset = train_dataset,
          qat_config = qat_config)

if __name__ == "__main__":
    main()