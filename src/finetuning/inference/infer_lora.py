import argparse
from peft import PeftModel

from ..training import load_lora_weights, merge_lora
from ..utils import load_config, load_model_and_tokenizer, run_inference, \
                    generate_run_name, save_predictions

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--adapter_type", type=str, choices=["scratch", "peft"], required=True)
    args = parser.parse_args()
    
    model_config = load_config('configs/model_config.yaml')
    lora_config = load_config('configs/train_lora.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    if args.adapter_type == "scratch":
        model = load_lora_weights(model = model, 
                                  path = args.adapter_path,
                                  target_modules = lora_config['lora']['target_modules']).to(model.device)
        
        model = merge_lora(model)
        name = "lora_scratch"
    elif args.adapter_type == "peft":
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        name = "lora_peft"
    else:
        name = "base"
    
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