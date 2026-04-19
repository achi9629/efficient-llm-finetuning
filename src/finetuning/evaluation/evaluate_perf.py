import os
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..utils.model_loader import load_model_and_tokenizer
from ..utils.experiment_io import load_config, save_metrics

def run_single_generation(model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer,
                          measure_ttft: bool,
                          inputs: dict,
                          gen_config: dict,
                          num_beams: int
    ) -> tuple[float|None, float|None]:
                
    """
    Description:
        Run a single text generation pass and measure performance metrics.
        This function generates text using a language model and measures either:
        1. Overall throughput (tokens per second) when measure_ttft is False
        2. Time to first token (TTFT) and throughput when measure_ttft is True
    Args:
        model (AutoModelForCausalLM): The language model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        measure_ttft (bool): If True, measures time to first token using a streamer in a separate thread.
                            If False, measures only overall throughput.
        inputs (dict): Dictionary containing:
                        - 'input_ids': Tensor of shape (batch_size, sequence_length) with token ids
                        - 'attention_mask': Tensor of shape (batch_size, sequence_length) indicating valid tokens
        gen_config (dict): Generation configuration dictionary containing:
                            - 'max_new_tokens': Maximum number of new tokens to generate
                            - 'early_stopping': Whether to stop beam search early
                            - 'no_repeat_ngram_size': N-gram size to prevent repetition
                            - 'temperature': Sampling temperature for randomness control
                            - 'do_sample': Whether to use sampling vs greedy decoding
        num_beams (int): Number of beams for beam search (1 for greedy, >1 for beam search).
    Returns:
        tuple[float|None, float|None]: A tuple containing:
            - tok_per_sec (float|None): Tokens generated per second, or None if no tokens generated
            - ttft_ms (float|None): Time to first token in milliseconds (None if measure_ttft is False)
    """
    
    # Generate
    with torch.inference_mode():
        
        '''
        Args for model.generate():
            - input_ids: Tensor of shape (batch_size, sequence_length) containing token ids for the input prompt.
            - attention_mask: Tensor of shape (batch_size, sequence_length) indicating which tokens should be attended to (1 for real tokens, 0 for padding).
            - max_new_tokens: Maximum number of new tokens to generate beyond the input prompt.
            - num_beams: Number of beams for beam search (higher values lead to more diverse outputs but slower generation).
            - early_stopping: Whether to stop beam search when at least num_beams sentences are finished per batch.
            - no_repeat_ngram_size: Size of n-grams that should not be repeated in the generated text (e.g., 2 means no bigrams can be repeated).
            - temperature: Sampling temperature for controlling randomness in generation (lower values make output more deterministic).
            - do_sample: Whether to use sampling instead of greedy decoding (if True, temperature and top-k/top-p sampling parameters will be applied).
        '''
        
        if not measure_ttft:
            
            start = time.perf_counter()
            
            outputs = model.generate(input_ids = inputs['input_ids'],
                                        attention_mask = inputs['attention_mask'],
                                        max_new_tokens = gen_config['max_new_tokens'],
                                        num_beams = num_beams,
                                        early_stopping = gen_config['early_stopping'],
                                        no_repeat_ngram_size = gen_config['no_repeat_ngram_size'],
                                        temperature = gen_config['temperature'],
                                        do_sample = gen_config['do_sample'],)

            end = time.perf_counter()
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Exclude prompt tokens
        
            elapsed = end - start
            tok_per_sec = len(generated_ids) / elapsed
            ttft_ms = None  # Not measured in this mode
        
        else:
            
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
            start = time.perf_counter()
        
            # Run generate in a thread so streamer can yield tokens
            thread = Thread(target = model.generate, kwargs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'max_new_tokens': gen_config['max_new_tokens'],
                'num_beams': num_beams,
                'early_stopping': False,
                'no_repeat_ngram_size': gen_config['no_repeat_ngram_size'],
                'temperature': gen_config['temperature'],
                'do_sample': gen_config['do_sample'],
                "streamer": streamer,
            })
        
            thread.start()
            
            first_token_time = None
            generated_token_count = 0
            for token_text in streamer:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_token_count += len(tokenizer.encode(token_text, add_special_tokens=False))

            end = time.perf_counter()
            thread.join()

            if first_token_time is None:
                return None, None  # No tokens were generated
            
            ttft_ms = (first_token_time - start) * 1000  # convert to ms
            elapsed = end - start
            tok_per_sec = generated_token_count / elapsed

    return tok_per_sec, ttft_ms

def measure_latency(model: AutoModelForCausalLM,
                    tokenizer: AutoTokenizer,
                    data_config: dict,
                    eval_config: dict) -> dict:

    """
    Description:
        Measure the latency and time-to-first-token (TTFT) of a causal language model.
        This function evaluates model performance by measuring inference latency across multiple
        samples and generation modes. It computes throughput (tokens per second) and optionally
        measures the time to generate the first token.
    Args:
        model (AutoModelForCausalLM): The pretrained causal language model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        data_config (dict): Configuration dictionary containing dataset and preprocessing settings.
            Expected keys:
            - dataset.name: Name of the dataset
            - dataset.config: Dataset configuration identifier
            - dataset.test_split: Dataset split to use (e.g., 'test', 'validation')
            - dataset.input_field: Field name containing input text
            - preprocessing.prompt_template: Template string for formatting prompts
            - preprocessing.max_input_length: Maximum input sequence length
        eval_config (dict): Configuration dictionary for evaluation settings.
            Expected keys:
            - evaluation.perf.latency.num_samples: Number of samples to evaluate
            - evaluation.perf.latency.warmup_runs: Number of warmup runs to skip before recording
            - evaluation.perf.latency.modes: List of generation modes with 'name', 'num_beams', 'measure_ttft'
            - evaluation.generation: Generation configuration parameters
    Returns:
        dict: A dictionary with key 'latency' containing per-mode results. Each mode includes:
            - tok_per_sec_p50: 50th percentile of tokens per second (throughput)
            - tok_per_sec_p90: 90th percentile of tokens per second
            - tok_per_sec_p95: 95th percentile of tokens per second
            - ttft_ms_avg (optional): Average time to first token in milliseconds (if measure_ttft is True)
    Raises:
        ValueError: If no latency records are collected after warmup runs.
    """

    num_samples  = eval_config['evaluation']['perf']['latency']['num_samples']
    warmup_runs = eval_config['evaluation']['perf']['latency']['warmup_runs']
    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    test_split = data_config['dataset']['test_split']
    prompt_template = data_config['preprocessing']['prompt_template']
    input_field = data_config['dataset']['input_field']
    max_input_len = data_config['preprocessing']['max_input_length']
    gen_config = eval_config['evaluation']['generation']
    
    modes = eval_config['evaluation']['perf']['latency']['modes']
    
    # Load dataset
    dataset = load_dataset(os.path.join('assets/datasets', 
                                        dataset_name, 
                                        dataset_config), 
                                        split = test_split)
    
    results = {}
    for mode in modes:
        num_beams = mode['num_beams']
        measure_ttft = mode['measure_ttft']
        mode_name = mode['name']
    

        # latency = tokens/second
        # ttft = time to first token in ms
        record_latencies = []
        record_ttft = []
        for idx in tqdm(range(num_samples), desc = 'Measuring Latency'):
        
            sample = dataset[idx]
            # Format prompt
            prompt = prompt_template.format(article=sample[input_field])
            
            # Tokenize
            '''
            Args for tokenizer():
                - text (str or List[str]): The input text(s) to tokenize.
                - return_tensors (str): The type of tensors to return ('pt' for PyTorch, 'tf' for TensorFlow, 'np' for NumPy).
                - truncation (bool or str): Whether to truncate sequences that exceed the model's maximum input length.
                                            If True, truncates to the model's max length. If 'longest_first', truncates the longest sequence first when processing batches.
                - max_length (int): The maximum length of the tokenized input sequence. If truncation is enabled, 
                                    sequences longer than this will be truncated to this length. This should typically 
                                    be set to the model's maximum input length minus the expected number of new tokens 
                                    to be generated to avoid exceeding the model's context window during generation.
            '''
        
            inputs = tokenizer(prompt, 
                            return_tensors = 'pt', 
                            truncation = True, 
                            max_length = max_input_len
                            ).to(model.device)
            
            tok_per_sec, ttft_ms = run_single_generation(model, tokenizer, measure_ttft, inputs, gen_config, num_beams)
            
            if idx >= warmup_runs and tok_per_sec is not None:
                record_latencies.append(tok_per_sec)
                if ttft_ms is not None:
                    record_ttft.append(ttft_ms)
        
        if len(record_latencies) == 0:
            raise ValueError("No latency records were collected. Please check the number of warmup runs and total samples.")
                    
        tok_per_sec_p50 = np.percentile(record_latencies, 50)
        tok_per_sec_p90 = np.percentile(record_latencies, 90)
        tok_per_sec_p95 = np.percentile(record_latencies, 95)
        if measure_ttft and len(record_ttft) > 0:
            ttft_ms_avg = np.mean(record_ttft)
        
        results[mode_name] = {
            'tok_per_sec_p50': tok_per_sec_p50,
            'tok_per_sec_p90': tok_per_sec_p90,
            'tok_per_sec_p95': tok_per_sec_p95,
        }
        if measure_ttft and len(record_ttft) > 0:
            results[mode_name]['ttft_ms_avg'] = ttft_ms_avg
    
    return results
    
def measure_memory(model: AutoModelForCausalLM,
                   tokenizer: AutoTokenizer,
                   eval_config: dict,
                   data_config: dict) -> dict:
    
    """
    Description:
        Measure GPU memory usage during model inference on a dataset sample.
        This function loads a dataset, prepares a single sample, tokenizes it according
        to the provided configuration, and generates text using the model while tracking
        GPU memory consumption metrics.
    Args:
        model (AutoModelForCausalLM): The pretrained causal language model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer associated with the model for encoding input text.
        eval_config (dict): Configuration dictionary containing evaluation parameters with structure:
            - 'evaluation': dict
                - 'generation': dict with keys like 'max_new_tokens', 'num_beams', 
                    'early_stopping', 'no_repeat_ngram_size', 'temperature', 'do_sample'
        data_config (dict): Configuration dictionary containing dataset and preprocessing parameters with structure:
            - 'dataset': dict
                - 'name': str (dataset name)
                - 'config': str (dataset configuration)
                - 'test_split': str (split to use)
                - 'input_field': str (field name containing input text)
            - 'preprocessing': dict
                - 'prompt_template': str (template string with {article} placeholder)
                - 'max_input_length': int (maximum tokenized input length)
    Returns:
        dict: A dictionary containing GPU memory metrics:
            - 'max_memory_allocated' (float): Peak GPU memory allocated in GB
            - 'memory_allocated' (float): Current GPU memory allocated in GB
            - 'memory_reserved' (float): Total GPU memory reserved in GB
    """

    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    test_split = data_config['dataset']['test_split']
    prompt_template = data_config['preprocessing']['prompt_template']
    input_field = data_config['dataset']['input_field']
    max_input_len = data_config['preprocessing']['max_input_length']
    gen_config = eval_config['evaluation']['generation']
    device = model.device
    
    # Load dataset
    dataset = load_dataset(os.path.join('assets/datasets', 
                                        dataset_name, 
                                        dataset_config), 
                                        split = test_split)
    
     # Generate
    with torch.inference_mode():
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        sample = dataset[0]
        # Format prompt
        prompt = prompt_template.format(article=sample[input_field])
        
        # Tokenize
        '''
        Args for tokenizer():
            - text (str or List[str]): The input text(s) to tokenize.
            - return_tensors (str): The type of tensors to return ('pt' for PyTorch, 'tf' for TensorFlow, 'np' for NumPy).
            - truncation (bool or str): Whether to truncate sequences that exceed the model's maximum input length.
                                        If True, truncates to the model's max length. If 'longest_first', truncates the longest sequence first when processing batches.
            - max_length (int): The maximum length of the tokenized input sequence. If truncation is enabled, 
                                sequences longer than this will be truncated to this length. This should typically 
                                be set to the model's maximum input length minus the expected number of new tokens 
                                to be generated to avoid exceeding the model's context window during generation.
        '''
    
        inputs = tokenizer(prompt, 
                        return_tensors = 'pt', 
                        truncation = True, 
                        max_length = max_input_len
                        ).to(model.device)
        
        '''
        Args for model.generate():
            - input_ids: Tensor of shape (batch_size, sequence_length) containing token ids for the input prompt.
            - attention_mask: Tensor of shape (batch_size, sequence_length) indicating which tokens should be attended to (1 for real tokens, 0 for padding).
            - max_new_tokens: Maximum number of new tokens to generate beyond the input prompt.
            - num_beams: Number of beams for beam search (higher values lead to more diverse outputs but slower generation).
            - early_stopping: Whether to stop beam search when at least num_beams sentences are finished per batch.
            - no_repeat_ngram_size: Size of n-grams that should not be repeated in the generated text (e.g., 2 means no bigrams can be repeated).
            - temperature: Sampling temperature for controlling randomness in generation (lower values make output more deterministic).
            - do_sample: Whether to use sampling instead of greedy decoding (if True, temperature and top-k/top-p sampling parameters will be applied).
        '''

        _ = model.generate(input_ids = inputs['input_ids'],
                                    attention_mask = inputs['attention_mask'],
                                    max_new_tokens = gen_config['max_new_tokens'],
                                    num_beams = gen_config['num_beams'],
                                    early_stopping = gen_config['early_stopping'],
                                    no_repeat_ngram_size = gen_config['no_repeat_ngram_size'],
                                    temperature = gen_config['temperature'],
                                    do_sample = gen_config['do_sample'],)


        max_memory_allocated = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2) if torch.cuda.is_available() else None
        memory_allocated = round(torch.cuda.memory_allocated(device) / 1024 ** 3, 2) if torch.cuda.is_available() else None
        memory_reserved = round(torch.cuda.memory_reserved(device) / 1024 ** 3, 2) if torch.cuda.is_available() else None
        
    return {
            'peak_vram_gb': max_memory_allocated,
            'current_vram_gb': memory_allocated,
            'reserved_vram_gb': memory_reserved,
            }
    
def measure_model_size(model: AutoModelForCausalLM) -> dict:
    
    """
    Description:
        Calculate the total model size and parameter count.
        This function computes the memory footprint of a model by multiplying the number of 
        parameters by their element size in bytes. The model size is converted from bytes to 
        gigabytes (GB), while the parameter count is converted to billions.
        Note: The conversion from bytes to GB uses 1024^3, while the parameter count uses 
        1,000,000,000 (1 billion). This introduces a factor difference of 1000/1024 between 
        the two conversions. For example, a model with a byte size of 7.24 GB would have 
        approximately 13.5 billion parameters (7.24 * 2 ≈ 13.5).
    Args:
        model (AutoModelForCausalLM): The model instance for which to measure size and 
                                      parameter count.
    Returns:
        dict: A dictionary containing:
            - 'model_size_gb' (float): The total model size in gigabytes (GB).
            - 'param_count_billion' (float): The total number of parameters in billions.
    """
    
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    
    # Return size in GB and parameter count in billions
    return {
            'model_size_gb': round(total_bytes / (1024 ** 3), 2),
            'param_count_billion': round(param_count / 1_000_000_000, 2)
            }

def main():
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    predictions_dir = eval_config['output']['predictions_dir']
    reports_dir = eval_config['output']['reports_dir']
    
    # Find latest predictions file
    preds_files = sorted(Path(predictions_dir).glob("*_preds.jsonl"))
    if not preds_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    preds_path = preds_files[-1]  # latest
    run_name = preds_path.stem.replace("_preds", "")
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.eval()
    
    metrics = {}
    
    metrics.update(measure_latency(model, tokenizer, data_config, eval_config))
    metrics.update(measure_memory(model, tokenizer, eval_config, data_config))
    metrics.update(measure_model_size(model))
    
    save_metrics(run_name, metrics, reports_dir, suffix = "perf")
    
if __name__ == '__main__':
    main()