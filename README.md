# Memory-Efficient Quantized GPT-2 Training in Raw C/CUDA

Weight quantization, optimizer-state quantization, and activation quantization for GPT-2
training built on top of Karpathy's `llm.c`.

## Setup

**Prerequisites:** CUDA toolkit, cuBLAS, Python 3.

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py   # download and tokenize dataset
python dev/data/fineweb.py           # (optional) larger dataset
```

Download pretrained weights for weight initialization:

```bash
python dev/download_starter_pack.sh
```

## Build

```bash
make train_gpt2cu
```

This produces `./train_gpt2cu`.

## Running Experiments

### Baseline (no quantization)

```bash
./train_gpt2cu -x 300 -s 100 -v 100
```

### Weight Quantization

```bash
# FP8 weights
./train_gpt2cu --ptq 1 --ptq_precision fp8 -x 300 -s 100 -v 100

# INT8 weights
./train_gpt2cu --ptq 1 --ptq_precision int8 -x 300 -s 100 -v 100

# INT4 weights (group size sweep)
./train_gpt2cu --ptq 1 --ptq_precision int4 --ptq_group_size 128 -x 300 -s 100 -v 100
./train_gpt2cu --ptq 1 --ptq_precision int4 --ptq_group_size 16  -x 300 -s 100 -v 100
./train_gpt2cu --ptq 1 --ptq_precision int4 --ptq_group_size 4   -x 300 -s 100 -v 100
```

### Optimizer-State Quantization

```bash
# FP8 moments (COAT-style)
./train_gpt2cu --optim_quant fp8 --optim_group_size 64 -x 300 -s 100 -v 100

# INT8 moments
./train_gpt2cu --optim_quant int8 --optim_group_size 64 -x 300 -s 100 -v 100

# INT4 moments
./train_gpt2cu --optim_quant int4 --optim_group_size 64 -x 300 -s 100 -v 100

# Dynamic range expansion on vs off
./train_gpt2cu --optim_quant fp8 --optim_group_size 64 --coat_expansion 1 -x 300 -s 100 -v 100
./train_gpt2cu --optim_quant fp8 --optim_group_size 64 --coat_expansion 0 -x 300 -s 100 -v 100
```

### Activation Quantization

```bash
# Naive grid quantization (diverges)
./train_gpt2cu --aq 1 --aq_type fp8 --aq_group_size 4 -x 300 -s 100 -v 100

# COAT-style row quantization (stable)
./train_gpt2cu --aq 1 --aq_type fp8 --aq_group_size 64 -x 300 -s 100 -v 100
```

### Combined (full low-precision config)

```bash
./train_gpt2cu --ptq 1 --ptq_precision int4 --ptq_group_size 4 \
               --optim_quant fp8 --optim_group_size 64 \
               --aq 1 --aq_type fp8 --aq_group_size 64 \
               -x 300 -s 100 -v 100
```

## Running Multiple Experiments and Plotting

Use `run_experiments.py` to run a batch of commands and compare loss curves:

```bash
python run_experiments.py
```

Paste your `./train_gpt2cu ...` commands one per line, then press Enter on a blank line.
Results are saved under `logs/` and a comparison plot is generated automatically.

To plot from existing logs:

```bash
python plot_train_losses.py
```

## Key Flags

| Flag | Options | Description |
|------|---------|-------------|
| `--ptq` | `0`, `1` | Enable weight quantization |
| `--ptq_precision` | `int8`, `fp8`, `int4` | Weight precision |
| `--ptq_group_size` | integer | Weight group size (0 = per-row) |
| `--optim_quant` | `fp32`, `fp8`, `int8`, `int4` | Optimizer moment precision |
| `--optim_group_size` | integer | Optimizer group size |
| `--coat_expansion` | `0`, `1` | COAT dynamic range expansion |
| `--aq` | `0`, `1` | Enable activation quantization |
| `--aq_type` | `fp8`, `int8`, `int4` | Activation precision |
| `--aq_group_size` | integer | Activation group size (columns per segment) |
| `-x` | integer | Number of training steps |
| `-s` | integer | Log interval |
| `-v` | integer | Validation interval |
