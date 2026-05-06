import argparse

from ..utils import run_inference, load_model_and_tokenizer, \
                    load_config, generate_run_name, save_predictions
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_type", type=str, choices=["base", "qat"], required=True)
    args = parser.parse_args()
    
    model_config = load_config('configs/model_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    if args.adapter_type == "qat":
        # qat_config = load_config('configs/qat_config.yaml')
        # block_top_k = qat_config['qat']['block_top_k']
        # module_top_k = qat_config['qat']['module_top_k']
        # bits = qat_config['qat']['bits']
        # steps = calculate_steps(qat_config['qat'])
        # tag = f"qat_block{block_top_k}_module{module_top_k}_{bits}bits_steps{steps}"
        tag = ""
        
    elif args.adapter_type == "base":
        tag = "base"
    else:
        raise ValueError(f"Unsupported adapter type: {args.adaptor_type}")
    
    model_config['model']['name'] = args.model_path
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.eval()
    predictions = run_inference(model, 
                                tokenizer, 
                                data_config, 
                                eval_config)
    
    run_name = generate_run_name(tag,
                                 model_config['model']['name'],
                                 data_config['dataset']['name'])
    save_predictions(run_name, predictions, eval_config['output']['predictions_dir'])
    
if __name__ == "__main__":
    main()