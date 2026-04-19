from ..utils import run_inference, load_model_and_tokenizer, \
                    load_config, generate_run_name, save_predictions
            
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