import argparse

from ..training import load_lora_weights, build_qlora_model, load_qlora_model_and_tokenizer
from ..utils import load_config, run_inference, generate_run_name, save_predictions

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    args = parser.parse_args()

    model_config = load_config('configs/model_config.yaml')
    train_qlora = load_config('configs/train_qlora.yaml')
    data_config = load_config('configs/data_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    
    r = train_qlora['lora']['r']
    alpha = train_qlora['lora']['lora_alpha']
    lr = train_qlora['training']['learning_rate']
    use_gradient_checkpointing = False
    tag = f"qlora_r{r}_a{alpha}_lr{lr}" if not use_gradient_checkpointing else f"qlora_r{r}_a{alpha}_lr{lr}_gc" 
    
    model, tokenizer = load_qlora_model_and_tokenizer(model_config, train_qlora, use_gradient_checkpointing)
    
    model = build_qlora_model(model, train_qlora)
    model = load_lora_weights(model=model,
                              path=args.adapter_path,
                              target_modules=train_qlora['lora']['target_modules'])
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
