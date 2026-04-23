import argparse
import csv
import json
import re
from pathlib import Path


RUN_TAG_RE = re.compile(
    r"^(?P<method>[a-zA-Z0-9_]+)_r(?P<rank>\d+)_a(?P<alpha>\d+)_lr(?P<lr>[0-9eE+.\-]+)_"
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_tag(run_name: str) -> dict:
    m = RUN_TAG_RE.match(run_name)
    if not m:
        raise ValueError(f"Could not parse run_name tag fields: {run_name}")
    out = m.groupdict()
    out["rank"] = int(out["rank"])
    out["alpha"] = int(out["alpha"])
    out["lr"] = float(out["lr"])
    return out


def collect_metrics(reports_dir: Path, method_prefix: str) -> dict:
    metrics_by_run = {}
    for path in sorted(reports_dir.glob(f"{method_prefix}*_metrics.json")):
        data = load_json(path)
        run_name = data.get("run_name")
        if not run_name:
            continue
        metrics_by_run[run_name] = data
    return metrics_by_run


def collect_perf(reports_dir: Path, method_prefix: str) -> dict:
    perf_by_run = {}
    for path in sorted(reports_dir.glob(f"{method_prefix}*_perf.json")):
        data = load_json(path)
        run_name = data.get("run_name")
        if not run_name:
            continue
        perf_by_run[run_name] = data
    return perf_by_run


def build_rows(metrics_by_run: dict, perf_by_run: dict, strict: bool = False) -> list[dict]:
    rows = []

    for run_name, metrics in metrics_by_run.items():
        tag = parse_run_tag(run_name)
        perf = perf_by_run.get(run_name)

        if strict and perf is None:
            raise FileNotFoundError(f"Missing perf json for run: {run_name}")

        row = {
            "method": tag["method"],
            "rank": tag["rank"],
            "alpha": tag["alpha"],
            "lr": tag["lr"],
            "run_name": run_name,
            "rouge1": metrics.get("rouge1"),
            "rouge2": metrics.get("rouge2"),
            "rougeL": metrics.get("rougeL"),
            "rougeLsum": metrics.get("rougeLsum"),
            "greedy_tok_per_sec_p50": None,
            "beam_tok_per_sec_p50": None,
            "greedy_tok_per_sec_p90": None,
            "beam_tok_per_sec_p90": None,
            "greedy_tok_per_sec_p95": None,
            "beam_tok_per_sec_p95": None,
            "peak_vram_gb": None,
            "ttft_ms_avg": None,
        }

        if perf is not None:
            greedy = perf.get("greedy", {})
            beam = perf.get("beam_search", {})
            row.update(
                {
                    "greedy_tok_per_sec_p50": greedy.get("tok_per_sec_p50"),
                    "beam_tok_per_sec_p50": beam.get("tok_per_sec_p50"),
                    "greedy_tok_per_sec_p90": greedy.get("tok_per_sec_p90"),
                    "beam_tok_per_sec_p90": beam.get("tok_per_sec_p90"),
                    "greedy_tok_per_sec_p95": greedy.get("tok_per_sec_p95"),
                    "beam_tok_per_sec_p95": beam.get("tok_per_sec_p95"),
                    "peak_vram_gb": perf.get("peak_vram_gb"),
                    "ttft_ms_avg": greedy.get("ttft_ms_avg"),
                }
            )

        rows.append(row)

    rows.sort(key=lambda x: x["rank"])
    return rows


def save_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"Collected {len(rows)} runs")
    best = max(rows, key=lambda r: (r["rougeL"] if r["rougeL"] is not None else float("-inf")))
    print(
        "Best by rougeL: "
        f"rank={best['rank']}, alpha={best['alpha']}, lr={best['lr']}, "
        f"rougeL={best['rougeL']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rank sweep CSV from metrics/perf JSON files.")
    parser.add_argument("--reports-dir", type=str, default="outputs/reports")
    parser.add_argument("--method-prefix", type=str, default="qlora_")
    parser.add_argument("--output", type=str, default="outputs/reports/rank_sweep.csv")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any metrics run is missing a matching perf file.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output)

    metrics_by_run = collect_metrics(reports_dir, args.method_prefix)
    if not metrics_by_run:
        raise FileNotFoundError(
            f"No metrics files found in {reports_dir} for prefix {args.method_prefix}"
        )

    perf_by_run = collect_perf(reports_dir, args.method_prefix)
    rows = build_rows(metrics_by_run, perf_by_run, strict=args.strict)
    save_csv(rows, output_path)
    print_summary(rows)
    print(f"Saved rank sweep csv to {output_path}")


if __name__ == "__main__":
    main()