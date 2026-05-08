# Training Arguments

| Flag | What it controls | Options |
|------|-----------------|---------|
| `--optim_quant` | Optimizer precision | `fp32`, `fp8`, `int8`, `int4` |
| `--optim_group_size` | Optimizer group size | power of 2 in [4, 1024] |
| `--coat_expansion` | Dynamic range expansion | `1` (on, default), `0` (off = plain absmax) |
| `--ptq` | Weight quantization | `0` (off), `1` (on) |
| `--ptq_precision` | Weight precision | `int8`, `fp8`, `int4` |
| `--ptq_group_size` | Weight group size | any integer |
| `--aq` | Activation quantization | `0` (off), `1` (on) |
| `--aq_type` | Activation precision | `fp8`, `int8`, `int4` |
| `--aq_group_size` | Activation group size | positive integer (square 2D groups) |

```bash
./train_gpt2cu --optim_quant int4 --optim_group_size 64 --ptq 1 --ptq_precision int4 --ptq_group_size 64 --aq 1 --aq_type int4 --aq_group_size 4
```

---

# Running Experiments

```bash
python run_experiments.py
```

1. Paste your `./train_gpt2cu ...` commands — one per line, or all at once
2. Press **Enter** on a blank line when done
3. It runs each experiment sequentially
4. At the end, choose **1** (train loss) or **2** (val loss) to plot
5. SVG comparison graph is saved to `logs/comparison_graphs/`

**Example (paste all at once):**
```
./train_gpt2cu --optim_quant int4 --optim_group_size 64 --ptq 1 --ptq_precision int4 --ptq_group_size 64 --aq 1 --aq_type fp8 --aq_group_size 16 -x 300 -s 100 -v 100
./train_gpt2cu --optim_quant int4 --optim_group_size 64 --ptq 1 --ptq_precision int4 --ptq_group_size 64 --aq 1 --aq_type int8 --aq_group_size 16 -x 300 -s 100 -v 100
./train_gpt2cu --optim_quant int4 --optim_group_size 64 --ptq 1 --ptq_precision int4 --ptq_group_size 64 --aq 1 --aq_type int4 --aq_group_size 16 -x 300 -s 100 -v 100

```
(blank line at the end to submit)
