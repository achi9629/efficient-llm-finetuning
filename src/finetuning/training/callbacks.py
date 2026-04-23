import os
import time
import torch
import random
import logging
import numpy as np
from transformers import TrainerCallback

from .lora_layer import save_lora_weights

logger = logging.getLogger(__name__)

class QLoRACheckpointCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 1. LoRA adapter weights
        save_lora_weights(model=model, save_path=os.path.join(ckpt_dir, "lora_adapter.pt"))
        
        # 2. Optimizer state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        
        # 3. Scheduler state
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        
        # 4. RNG states (for exact reproducibility on resume)
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_states, os.path.join(ckpt_dir, "rng_state.pth"))
        
        # 5. Trainer state (global_step, epoch, best_metric, log_history, etc.)
        state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
        
        # 6. Training args
        torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
        
        logger.info(f"QLoRA checkpoint saved to {ckpt_dir}")
     
    def on_evaluate(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(ckpt_dir):
            state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))

class GPUMemoryCallback(TrainerCallback):
    
    def __init__(self, avg_seq_len = None):
        
        self.avg_seq_len = avg_seq_len
    
    def on_train_begin(self, args, state, control, **kwargs):
        
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0
                             
    def on_log(self, args, state, control, logs = None, **kwargs):
        
        if logs is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.last_log_time
        steps_since = state.global_step - self.last_log_step
        
        if steps_since > 0 and elapsed_time > 0:
            samples = (steps_since * \
                       args.per_device_train_batch_size * \
                       args.gradient_accumulation_steps * \
                       args.world_size)
            samples_per_sec = samples / elapsed_time
            logs["samples_per_sec"] = round(samples_per_sec, 2)
            if self.avg_seq_len is not None:
                logs["tokens_per_sec"] = round(samples_per_sec * self.avg_seq_len, 1)
        
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logs["peak_gpu_memory_gb"] = round(peak_memory_gb, 2)
        
        
        self.last_log_time = current_time
        self.last_log_step = state.global_step
        
    def on_train_end(self, args, state, control, **kwargs):
        
        total_time = time.time() - self.start_time
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_samples = (state.global_step * \
                         args.per_device_train_batch_size * \
                         args.gradient_accumulation_steps * \
                         args.world_size)
        avg_throughput = total_samples / total_time
        
        summary = (
            f"\n{'=' * 60}\n"
            f"TRAINING SUMMARY\n"
            f"{'=' * 60}\n"
            f"  Total time:         {total_time:.1f}s ({total_time/60:.1f}min)\n"
            f"  Peak GPU VRAM:      {peak_memory_gb:.2f} GB\n"
            f"  Total samples:      {total_samples}\n"
            f"  Avg throughput:     {avg_throughput:.2f} samples/sec\n"
            f"{'=' * 60}\n"
        )
        logger.info(summary)