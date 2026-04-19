import torch, os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.model_loader import load_model_and_tokenizer
from ..utils.experiment_io import load_config, generate_run_name, save_predictions

def run_inference(model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  data_config: dict,
                  eval_config: dict) -> list[dict]:
    
    """
    Description:
        Run inference on a model using a dataset and generate predictions.
        This function loads a dataset, processes each sample through a language model,
        and generates predictions using specified generation parameters. It handles
        prompt formatting, tokenization, and text generation for evaluation purposes.
    Args:
        model (AutoModelForCausalLM): A pretrained causal language model loaded via
            transformers.AutoModelForCausalLM. The model should be in evaluation mode
            and on the appropriate device (CPU/GPU).
        tokenizer (AutoTokenizer): A pretrained tokenizer corresponding to the model,
            used for encoding prompts and decoding generated tokens.
        data_config (dict): Configuration dictionary containing dataset and preprocessing
            settings with the following structure:
            - dataset:
                - name (str): Name or identifier of the dataset
                - config (str): Configuration variant of the dataset
                - test_split (str): Dataset split to use for inference (e.g., 'test', 'validation')
                - input_field (str): Key name for the input/article field in dataset samples
                - target_field (str): Key name for the target/reference field in dataset samples
            - preprocessing:
                - prompt_template (str): Template string for formatting prompts with
                    format placeholder 'article' (e.g., "Summarize: {article}")
                - max_input_length (int): Maximum token length for input prompts (truncation threshold)
        eval_config (dict): Configuration dictionary containing generation parameters
            with the following structure:
            - evaluation:
                - generation (dict): Generation parameters including:
                    - max_new_tokens (int): Maximum number of new tokens to generate
                    - num_beams (int): Number of beams for beam search
                    - early_stopping (bool): Whether to stop beam search early
                    - no_repeat_ngram_size (int): N-gram size to avoid repetition
                    - temperature (float): Sampling temperature for diversity control
                    - do_sample (bool): Whether to use sampling instead of greedy decoding
    Returns:
        list[dict]: A list of prediction dictionaries, each containing:
            - input (str): The formatted prompt truncated to 500 characters
            - prediction (str): The generated text output from the model
            - reference (str): The ground truth target text from the dataset
    Note:
        - The dataset is loaded from local path 'assets/datasets/{dataset_name}/{dataset_config}'
            rather than directly from Hugging Face Hub due to proxy constraints
        - Model inference runs without gradient computation for efficiency (torch.no_grad context)
        - Generated text excludes the input prompt tokens from the model output
        - Progress is displayed via tqdm progress bar during inference
    """
    
    dataset_name = data_config['dataset']['name']
    dataset_config = data_config['dataset']['config']
    test_split = data_config['dataset']['test_split']
    prompt_template = data_config['preprocessing']['prompt_template']
    max_input_len = data_config['preprocessing']['max_input_length']
    input_field = data_config['dataset']['input_field']
    target_field = data_config['dataset']['target_field']
    gen_config = eval_config['evaluation']['generation']
    
    # Load dataset
    # Due to proxy issue load_dataset direclty won't work, 
    # so we have to manually download the dataset and load it from local path
    # dataset = load_dataset(dataset_name, dataset_config, split = test_split)
    dataset = load_dataset(os.path.join('assets/datasets', 
                                        dataset_name, 
                                        dataset_config), 
                                        split = test_split)
    
    predictions = []
    
    for sample in tqdm(dataset, desc = "Running Inference"):
        
        # Format prompt
        prompt = prompt_template.format(article=sample[input_field])
        reference = sample[target_field]
        
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
            
            outputs = model.generate(input_ids = inputs['input_ids'],
                                     attention_mask = inputs['attention_mask'],
                                     max_new_tokens = gen_config['max_new_tokens'],
                                     num_beams = gen_config['num_beams'],
                                     early_stopping = gen_config['early_stopping'],
                                     no_repeat_ngram_size = gen_config['no_repeat_ngram_size'],
                                     temperature = gen_config['temperature'],
                                     do_sample = gen_config['do_sample'],)
    
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Exclude prompt tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens = True)
            
            predictions.append({"input": prompt[:500],  # truncate for storage
                                "prediction": generated_text,
                                'reference': reference
                                })
            
    return predictions
            
def main():
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.eval()
    predictions = run_inference(model, 
                                tokenizer, 
                                data_config, 
                                eval_config)
    
    run_name = generate_run_name("base", 
                                 model_config['model']['name'],
                                 data_config['dataset']['name'])
    save_predictions(run_name, predictions, eval_config['output']['predictions_dir'])
    
if __name__ == "__main__":
    main()