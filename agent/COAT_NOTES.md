# COAT — Paper Notes and Implementation Context

> Reference for implementing COAT ideas in this codebase (train_gpt2.cu / llmc/).
> Paper: "COAT: Compressing Optimizer States and Activations for Memory-Efficient FP8 Training"
> Authors: Xi et al., NVIDIA + UC Berkeley + MIT, 2025.

---

## 1. What COAT is (one paragraph)

Standard FP8 training (TransformerEngine) only runs the matrix multiplications in FP8. The
optimizer states (m and v in AdamW), activations stored for the backward pass, and gradients all
stay in BF16 or FP32. COAT fixes this by quantizing **both optimizer states and activations** into
FP8, getting a 1.54× total memory reduction and 1.43× speedup vs BF16, with nearly zero
accuracy loss across LLM pretraining, fine-tuning, and VLM training.

---

## 2. The two ideas COAT contributes

### Idea 1 — Dynamic Range Expansion (for optimizer states)

**The problem.** FP8 E4M3 can represent values from ~2⁻⁹ to 448 (a range ratio of ~2×10⁵).
But optimizer states (m and v in AdamW) are highly peaked — most values are small and clustered,
with only ~1% outliers. So the actual dynamic range of a quantization group is:
- First-order momentum m: typically < 10⁴
- Second-order momentum v: typically < 10¹

Both are **far below** what FP8 can represent. Most of FP8's bits are wasted.

**The fix.** Before quantizing, apply an expand function:

```
f(x) = sign(x) * |x|^k
```

where `k = log_R(R_E4M3)` and `R` is the actual dynamic range of the group.
This "stretches" the distribution to fill FP8's range, reducing quantization error by ~1.63×.

After dequantizing, apply the inverse:

```
f⁻¹(x) = sign(x) * |x|^(1/k)
```

`k` is computed per-group, per-step. It is stored alongside the scale factor.

**Key numbers:**
- First-order m: k ≈ 1–3 (modest expansion)
- Second-order v: k ≈ 5–15 (larger expansion, because v has smaller dynamic range)

**What this replaces in our codebase.** Currently `m_memory` and `v_memory` are stored as
FP32. COAT compresses them to FP8 + one float scale + one float k per group of 128 elements.
Memory for optimizer states drops from ~8 bytes/param (FP32 m + FP32 v) to ~2 bytes/param
(FP8 m + FP8 v + BF16 scales).

---

### Idea 2 — Mixed-Granularity Activation Quantization (for activations)

**The problem.** During the forward pass, activations must be saved for the backward pass. For a
Llama-style model, this is ~22.7U of memory (U = B×T×C×2 bytes). That's more than weights
in many training configs. Non-linear layers (LayerNorm, activation functions) account for ~50%
of activation memory; linear layer inputs account for ~25%.

**The fix.** Save activations in FP8 instead of BF16. The trick is choosing the right quantization
granularity per layer type:

| Layer type | Quantization scheme | Why |
|---|---|---|
| Linear layers (QKV, MLP, etc.) | Per-tensor FP8 | Efficient on TensorCores; per-group kills speed |
| Non-linear layers (LayerNorm, SiLU) | Per-group FP8 (group size 1×16, along hidden dim) | Token-axis quantization causes large error; must be per-hidden-group |

Result: activation memory drops to ~13.3U, a 1.65× reduction vs BF16.

**Group Scaling.** For per-tensor quantization you need the max of the whole tensor (expensive).
COAT splits this into two stages:
1. Compute max over each 1×G slice (fuseable into the previous kernel).
2. Take max over the G× smaller intermediate tensor.

This avoids the full-tensor max reduction overhead without using delayed scaling.

---

## 3. What COAT does NOT do (relevant gaps for our project)

- COAT does not change which weights are quantized — weights are handled separately by the
  weight quantization path (INT8/FP8 weights, same as what this codebase already does).
- COAT's backward gradients stay in BF16 (not quantized). So gradient buffers (`grads_memory`)
  stay the same size.
- COAT does not use INT8; it is FP8-only (E4M3 for weights/activations, E4M3 or E5M2 for
  optimizer states).
- COAT does not address the master weight copy — it stays FP32 in their setup too.

---

## 4. Memory breakdown: before and after COAT

For Llama-2-7B on 4 GPUs (what the paper reports):

| Component | BF16 | COAT |
|---|---|---|
| Optimizer states (m + v) | 13.1 GB | 3.2 GB |
| Activations | 25.8 GB | 16.9 GB |
| Peak total | 55.1 GB | 35.6 GB |
| Max batch size | 2 | 4 |
| Throughput | 7730 tok/s | 11257 tok/s |

Optimizer states go from 4 bytes/param × 2 = 8 bytes/param (FP32 m + v) down to ~1 byte/param
each (FP8 + scale overhead amortized over group size 128).

---

## 5. How the optimizer step works in COAT

This is the full COAT AdamW step (Appendix A of the paper):

```
# Start of step t:
m_t-1 = dequantize(m_fp8_t-1, scale_m, k_m)   # FP8 → FP32, undo expand
v_t-1 = dequantize(v_fp8_t-1, scale_v, k_v)   # FP8 → FP32, undo expand

# Standard AdamW update in FP32:
m_t = β1 * m_t-1 + (1 - β1) * g_t
v_t = β2 * v_t-1 + (1 - β2) * g_t²
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
w_t+1 = w_t - lr * (m_hat / (sqrt(v_hat) + ε) + λ * w_t)

# Quantize updated states back to FP8 with dynamic range expansion:
m_fp8_t, scale_m = quantize(expand(m_t, k_m))
v_fp8_t, scale_v = quantize(expand(v_t, k_v))
# k is recomputed per group from current dynamic range
```

Compare to our current codebase: `adamw_update` in `llmc/adamw.cuh` does the same AdamW
math but stores m and v in FP32 (`m_memory`, `v_memory`). To add COAT, you would:
1. Store m and v in FP8 + (scale, k) per group of 128.
2. Dequantize at the start of `adamw_update`, run the same math, requantize at the end.

---

## 6. How activation quantization works in COAT

The key insight: instead of saving BF16 activations for backward, save FP8 activations.
The quantization happens at the *output* of each layer (the tensor that would be saved for backward
is saved in FP8 instead of BF16).

In the backward pass, the saved FP8 activation is dequantized back to BF16 before use.

For non-linear layers (LayerNorm, SiLU), use per-group quantization with group size 16 along
the hidden dimension (not across the token dimension — that causes large error).

For linear layers, use per-tensor quantization (single scale for the whole input tensor).

**This is not yet implemented in this codebase at all.** Currently all activations in `acts.*`
are stored in floatX (BF16 when `PRECISION_MODE=BF16`).

---

## 7. Implementation priority for this project

Ranked by value and complexity:

| Priority | COAT feature | Memory saved | Complexity |
|---|---|---|---|
| 1 | FP8 optimizer states (Dynamic Range Expansion) | ~4× on m+v | Medium — new FP8 storage for m/v in adamw |
| 2 | Activation quantization for non-linear layers | ~2× on LN/activation memory | High — need to save/restore FP8 in forward |
| 3 | Activation quantization for linear layers | ~0.5× on linear activation memory | High — touch matmul_forward paths |

For Phase 2 of this project (COAT-style FP8 optimizer states), only priority #1 is in scope.
Do not start #2 or #3 without explicit user instruction.

---

## 8. Relevant files to touch for Phase 2 (optimizer state compression)

| File | What changes |
|---|---|
| `llmc/adamw.cuh` | The `adamw_update` kernel — add FP8 m/v storage with expand/compress |
| `train_gpt2.cu` | `m_memory` and `v_memory` allocation — change from FP32 to FP8 layout |
| `train_gpt2.cu` | Checkpoint save/load — new format for FP8 optimizer states |
| `llmc/` (new file) | `coat_fp8_optim.cuh` — expand kernel, compress kernel, per-group k computation |

Do NOT touch the weight quantization path (`ptq_*` functions) when implementing optimizer
state compression. They are independent.

---

## 9. Things to verify before implementing

Per the QAT_REPORT.md known issues, fix these first (Phase 1) before adding COAT:
1. Master weight init bug (init from already-quantized weights).
2. INT8 + FP8 codec round-trip unit test.
3. Baseline val loss numbers measured and recorded in STATUS.md.

COAT optimizer state compression is Phase 2. Do not start until Phase 1 exit criteria are met.
