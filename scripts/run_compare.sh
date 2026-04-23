python -m src.finetuning.evaluation.compare_runs \
  --runs \
    "base:outputs/reports/base_mistral7bv03_cnndaily_20260416_211313_metrics.json:outputs/reports/base_mistral7bv03_cnndaily_20260416_211313_perf.json" \
    "lora_scratch:outputs/reports/lora_scratch_mistral7bv03_cnndaily_20260420_061534_metrics.json:outputs/reports/lora_scratch_mistral7bv03_cnndaily_20260420_061534_perf.json" \
    "lora_peft:outputs/reports/lora_peft_mistral7bv03_cnndaily_20260420_073304_metrics.json:outputs/reports/lora_peft_mistral7bv03_cnndaily_20260420_073304_perf.json" \
    "qlora:outputs/reports/qlora_r2_a4_lr0.0002_mistral7bv03_cnndaily_20260423_164506_metrics.json:outputs/reports/qlora_r2_a4_lr0.0002_mistral7bv03_cnndaily_20260423_164506_perf.json" \
  --output outputs/reports/day6_4way_comparison.csv