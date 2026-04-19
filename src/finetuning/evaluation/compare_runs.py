import csv
import json
import argparse
from pathlib import Path

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
    
def build_comparison_table(run_labels: list[str],
                           metrics_paths: list[str],
                           perf_paths: list[str]
    ) -> list[dict]:
    
    rows = []
    for label, metrics_path, perf_path in zip(run_labels, metrics_paths, perf_paths):
        metrics = load_json(metrics_path)
        perf = load_json(perf_path)
        rows.append({
            "run": label,
            "rouge1": metrics["rouge1"],
            "rouge2": metrics["rouge2"],
            "rougeL": metrics["rougeL"],
            "rougeLsum": metrics["rougeLsum"],
            "greedy_tok_per_sec_p50": perf["greedy"]["tok_per_sec_p50"],
            "beam_tok_per_sec_p50": perf["beam_search"]["tok_per_sec_p50"],
            "greedy_tok_per_sec_p90": perf["greedy"]["tok_per_sec_p90"],
            "beam_tok_per_sec_p90": perf["beam_search"]["tok_per_sec_p90"],
            "greedy_tok_per_sec_p95": perf["greedy"]["tok_per_sec_p95"],
            "beam_tok_per_sec_p95": perf["beam_search"]["tok_per_sec_p95"],
            "peak_vram_gb": perf["peak_vram_gb"],
            "ttft_ms_avg": perf["greedy"]["ttft_ms_avg"],
        })
    return rows

def print_table(rows: list[dict]) -> None:
    
    headers = list(rows[0].keys())
    col_widths = []
    for h in headers:
        max_w = len(h)
        for row in rows:
            val = row[h]
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            max_w = max(max_w, len(cell))
        col_widths.append(max_w)
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-|-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    
    for row in rows:
        cells = []
        for h, w in zip(headers, col_widths):
            val = row[h]
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            cells.append(cell.ljust(w))
        print(" | ".join(cells))

def save_csv(rows: list[dict], output_path: str) -> None:
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved comparison table to {output_path}")
    
def main():
    
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run entries as label:metrics_path:perf_path")
    parser.add_argument("--output", type=str, 
                        default="outputs/reports/comparison_table.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    
    run_labels, metrics_paths, perf_paths = [], [], []
    for run_spec in args.runs:
        parts = run_spec.split(":")
        if len(parts) != 3:
            parser.error(f"Invalid run spec '{run_spec}'. Expected label:metrics_path:perf_path")
        run_labels.append(parts[0])
        metrics_paths.append(parts[1])
        perf_paths.append(parts[2])
    
    rows = build_comparison_table(run_labels, metrics_paths, perf_paths)
    print_table(rows)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()