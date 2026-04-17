# Project Folder Structure

Recommended directory layout for the efficient fine-tuning and quantization project.

```bash
efficient-llm-finetuning/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── model_config.yaml              # base model, dtype, max_seq_len, precision mode
│   ├── data_config.yaml               # dataset paths/splits, max lengths, preprocessing options
│   ├── train_lora.yaml                # LoRA hyperparams (r, alpha, dropout, target modules)
│   ├── train_qlora.yaml               # QLoRA hyperparams + 4-bit settings (nf4, double quant)
│   ├── ptq_config.yaml                # PTQ calibration settings and backend configs
│   ├── qat_config.yaml                # selective-QAT settings and training schedule
│   └── tensorrt_config.yaml           # TensorRT profiles, precision modes, workspace
├── docs/
│   ├── progress_finetuning.md
│   ├── folder_structure_finetuning.md
│   ├── career_strategy.md             # optional: copy/symlink from P1
│   └── concepts/
│       ├── lora.md                    # low-rank adaptation notes
│       ├── qlora.md                   # quantized adapters and memory tradeoffs
│       ├── ptq.md                     # calibration flow and tradeoffs
│       ├── qat.md                     # fake-quant flow and selective-QAT strategy
│       ├── tensorrt_deployment.md     # ONNX export to TensorRT engine lifecycle
│       └── failure_analysis.md        # common quantization and deployment failure patterns
├── src/
│   └── finetuning/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset_loader.py      # train/val/test loading and split handling
│       │   ├── preprocess.py          # tokenization and prompt formatting
│       │   └── calibration_sampler.py # representative set builder for PTQ
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_lora.py          # From Scratch LoRA training entrypoint
│       │   ├── lora_layers.py         # From Scratch LoRALinear class implementation
│       │   ├── train_lora_peft.py     # LoRA training entrypoint
│       │   ├── train_qlora.py         # QLoRA training entrypoint
│       │   ├── train_qat.py           # selective-QAT training entrypoint
│       │   ├── trainer_utils.py       # optimizer/scheduler/seed/checkpoint helpers
│       │   └── callbacks.py           # memory + throughput logging hooks
│       ├── quantization/
│       │   ├── __init__.py
│       │   ├── ptq_pipeline.py        # PTQ workflow (calibrate, convert, validate)
│       │   ├── qat_pipeline.py        # QAT setup + conversion helpers
│       │   └── sensitivity.py         # layer/head sensitivity and fallback analysis
│       ├── deployment/
│       │   ├── __init__.py
│       │   ├── export_onnx.py         # export models to ONNX with deployment constraints
│       │   ├── build_trt_engine.py    # build FP16/INT8 TensorRT engines
│       │   ├── calibrate_trt.py       # TensorRT INT8 calibration runner
│       │   └── validate_trt.py        # runtime validation on target hardware
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluate_task.py       # task metric computation across variants
│       │   ├── evaluate_perf.py       # latency/throughput/memory evaluation
│       │   └── compare_runs.py        # summary table generation
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── infer_base.py          # base model inference
│       │   ├── infer_lora.py          # LoRA checkpoint inference
│       │   ├── infer_qlora.py         # QLoRA checkpoint inference
│       │   └── infer_trt.py           # TensorRT engine inference wrapper
│       └── utils/
│           ├── __init__.py
│           ├── metrics.py             # accuracy/f1/rouge wrappers
│           ├── memory_monitor.py      # GPU memory capture (peak and step-wise)
│           ├── speed_monitor.py       # tokens/sec or samples/sec capture
│           └── experiment_io.py       # run IDs and artifact save/load helpers
├── benchmarks/
│   ├── commands.txt
│   ├── metrics.py                     # shared benchmark metric helpers
│   ├── benchmark_lora_qlora.py        # base vs lora vs qlora comparisons
│   ├── benchmark_ptq.py               # PTQ impact on quality and performance
│   ├── benchmark_qat.py               # selective-QAT recovery and cost
│   ├── benchmark_tensorrt.py          # FP32/FP16/INT8 TensorRT benchmark suite
│   └── benchmark_gguf.py              # Convert fine-tuned HF checkpoint to GGUF format at multiple quant levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0)
├── scripts/
│   ├── commands.txt
│   ├── run_lora.sh                    # LoRA train + eval command
│   ├── run_qlora.sh                   # QLoRA train + eval command
│   ├── run_ptq.sh                     # PTQ calibration + conversion + eval
│   ├── run_qat.sh                     # selective-QAT train + conversion + eval
│   ├── run_export_onnx.sh             # export ONNX artifacts
│   ├── run_build_trt.sh               # build TensorRT engines
│   ├── run_compare.sh                 # generate final comparison matrix
│   └── run_gguf_export.sh             # Benchmark GGUF variants against TensorRT INT8 and QLoRA (quality, latency, model size)
├── tests/
│   ├── commands.txt
│   ├── test_data_pipeline.py
│   ├── test_train_lora_smoke.py
│   ├── test_train_qlora_smoke.py
│   ├── test_ptq_pipeline.py
│   ├── test_qat_pipeline.py
│   ├── test_onnx_export.py
│   └── test_tensorrt_inference.py
├── notebooks/
│   ├── finetuning_plots.ipynb
│   └── quantization_analysis.ipynb
├── integration/
│   ├── README.md                      # how to plug best checkpoint/engine into serving stack
│   └── adapter_loader.py              # helper to load adapter or TensorRT engine variants
├── outputs/
│   ├── runs/                          # run-wise logs and scalar metrics
│   ├── checkpoints/                   # adapters, QAT checkpoints, merged models
│   ├── onnx/                          # exported ONNX files
│   ├── engines/                       # TensorRT serialized engines
│   ├── predictions/                   # model outputs for qualitative analysis
│   └── reports/                       # final comparison tables and plots
└── assets/
    ├── files/
    │   ├── sample_prompts.txt
    │   └── calibration_manifest.json
    └── plots/
```
