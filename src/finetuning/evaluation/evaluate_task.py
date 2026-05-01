import json
import argparse
import evaluate
from tqdm import tqdm
from pathlib import Path

from ..utils import load_config, save_metrics

def compute_rouge(predictions: list[str], 
                  references: list[str]
    ) -> dict:
    
    """
    Description:
        Compute ROUGE scores for comparing predicted text against reference text.
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics
        used to evaluate automatic summarization and machine translation by comparing
        the overlap of n-grams, word sequences, and word pairs between the generated
        and reference texts.
    Args:
        predictions (list[str]): A list of predicted/generated text strings.
        references (list[str]): A list of reference/ground truth text strings.
    Returns:
        dict: A dictionary containing ROUGE scores including rouge1, rouge2, rougeL,
                and rougeLsum. Each score typically includes 'precision', 'recall',
                and 'fmeasure' metrics.
    Raises:
        ValueError: If predictions and references lists have different lengths.
    Example:
        >>> predictions = ["The cat is on the mat.", "The dog is in the house."]
        >>> references = ["The cat is on the mat.", "The dog is in the yard."]
        >>> scores = compute_rouge(predictions, references)
        >>> print(scores)
        >>> {'rouge1': 0.88, 'rouge2': 0.75, 'rougeL': 0.88, 'rougeLsum': 0.88}
    """
    
    if len(predictions) != len(references):
        raise ValueError("The number of predictions and references must be the same.")
    
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions = predictions, references = references)
    
    return results

def compute_bertscore(predictions: list[str],
                      references: list[str]
    ) -> dict:
    
    """
    Description:
        Compute BERTScore metrics comparing predictions against references.
        BERTScore is an automatic evaluation metric for text generation that measures
        semantic similarity between generated text and reference text using contextual
        embeddings from BERT.
    Args:
        predictions: A list of predicted/generated text strings to evaluate.
        references: A list of reference text strings to compare against.
                    Must have the same length as predictions.
    Returns:
        A dictionary containing average scores across all prediction-reference pairs:
            - bertscore_precision: Average precision score (0-1).
            - bertscore_recall: Average recall score (0-1).
            - bertscore_f1: Average F1 score (0-1).
    Raises:
        ValueError: If the number of predictions and references do not match.
    """
    
    if len(predictions) != len(references):
        raise ValueError("The number of predictions and references must be the same.")
    
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions = predictions, references = references, lang="en")
    
    n = len(results['f1'])
    return {
            "bertscore_precision": sum(results["precision"]) / n,
            "bertscore_recall": sum(results["recall"]) / n,
            "bertscore_f1": sum(results["f1"]) / n,
        }

def evaluate_from_file(preds_path: str) -> dict:
    
    """
    Description:
        Evaluate predictions against references by reading from a JSONL file.
        Loads predictions and references from a JSONL file where each line contains
        a JSON object with 'prediction' and 'reference' fields. Computes and returns
        ROUGE scores for the predictions.
    Args:
        preds_path (str): Path to the JSONL file containing predictions and references.
                         Each line should be a JSON object with 'prediction' and 'reference' keys.
    Returns:
        dict: Dictionary containing ROUGE evaluation scores computed from the predictions
              and references.
    Raises:
        FileNotFoundError: If the file at preds_path does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
        KeyError: If a JSON object is missing 'prediction' or 'reference' keys.
    """
    
    predictions = []
    references = []
    
    if not Path(preds_path).exists():
        raise FileNotFoundError(f"The file {preds_path} does not exist.")
    
    with open(preds_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading predictions"):
            entry = json.loads(line)

            if "prediction" not in entry or "reference" not in entry:
                raise KeyError("Each JSON object must contain 'prediction' and 'reference' keys.")
            predictions.append(entry["prediction"])
            references.append(entry["reference"])
            
    print('Computing ROUGE scores...')      
    rouge_scores = compute_rouge(predictions, references)
    # print('Computing BertScore...')
    # bert_scores = compute_bertscore(predictions, references)
    print('Evaluation complete.')
    return {**rouge_scores}

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", type=str, choices=["lora_scratch", "lora_peft", "base", "qlora", "ptq_int8", 
                                                             "ptq_nf4", "gptq_int4"], required=True)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank (ignored for non-LoRA models)")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha (ignored for non-LoRA models)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate used during fine-tuning (for tagging purposes)")
    
    args = parser.parse_args()
    
    eval_config = load_config('configs/eval_config.yaml')

    metric_keys = eval_config['evaluation']['metrics']
    predictions_dir = eval_config['output']['predictions_dir']
    reports_dir = eval_config['output']['reports_dir']
    
    adapter_type = args.adapter_type
    if adapter_type == "qlora":
        r = args.r
        alpha = args.alpha
        lr = args.lr
        tag = f"{adapter_type}_r{r}_a{alpha}_lr{lr}" if adapter_type == 'qlora' else adapter_type
    else:
        tag = adapter_type
    
    # Find latest predictions file
    preds_files = sorted(Path(predictions_dir).glob("*" + tag + "*_preds.jsonl"))
    if not preds_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    preds_path = preds_files[-1]  # latest
    run_name = preds_path.stem.replace("_preds", "")
    
    output = evaluate_from_file(preds_path)
    metrics = {key: output[key] for key in metric_keys}
    
    save_metrics(run_name, metrics, reports_dir)
    
if __name__ == "__main__":
    main()
    
    