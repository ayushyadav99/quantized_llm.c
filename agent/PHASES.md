# PHASES.md — The four-phase plan

This is the agreed sequencing for QAT work in this repo. Each phase has an
**entry criterion** (what must be true before the phase starts) and an **exit
criterion** (what must be measurably true before the phase is declared done).

Do not start a phase before its entry criterion is met. Do not declare a phase
done before its exit criterion is met. The current phase is recorded in
`STATUS.md`.

---

## Phase 1 — Weight-only QAT, clean

**Goal.** The four large transformer weight matrices (`qkvw`, `attprojw`,
`fcw`, `fcprojw`) are stored quantized (INT8 or FP8 E4M3), the forward pass
sees the dequantized values (so the model experiences quantization rounding
during training), the optimizer keeps an FP32 master and re-quantizes after
every step. Convergence matches the BF16 baseline within a small tolerance.

**Entry criterion.** Code compiles. `--ptq 1 --ptq_precision int8` runs without
crashing on GPT-2 124M.

**Exit criteria.**

1. Round-trip unit test for the INT8 and FP8 codecs in `dev/test/` passes
   (lives next to other tests, runs in under a minute, prints PASS/FAIL).
2. FP32 master is initialized from the **original** weights, not from the
   already-quantized values.
3. The first call to quantize-from-scratch and the per-step re-quantization
   both run from the FP32 master (consistent input precision).
4. A `quantization_noise` metric (per-tensor mean
   `|W_master - dequant(quant(W_master))| / |W_master|`) is logged each
   `val_loss_every` steps.
5. On GPT-2 124M, validation loss curve with INT8 weight QAT is within `0.05`
   of the BF16-no-quant baseline at step 1000. Numbers logged in `STATUS.md`.
6. Hellaswag accuracy (if available) does not regress by more than 1 absolute
   point.

**Risks / what could go wrong.** Master init bug propagating quantization
error from step 0. FP8 E4M3 codec being silently incorrect (it is hand-rolled
— must be tested against `cuda_fp8.h`). Re-quantization scale being computed
from a stale source.

**Out of scope for this phase.** Activation quantization. Optimizer state
compression. INT8 GEMM kernels. Touching `wte` / `wpe`.

---

## Phase 2 — COAT-style optimizer state compression

**Goal.** AdamW's `m` and `v` are stored in FP8 (E4M3 + E4M3, or E4M3 + E5M2)
using COAT's per-group dynamic-range expansion. Master weights stay FP32.
Memory footprint of optimizer state drops by roughly 4×. Convergence still
matches the FP32-state baseline.

**Entry criterion.** Phase 1 exit criteria are all met. Branch is clean.

**Exit criteria.**

1. `expand` / `contract` per-group transforms implemented and unit-tested
   against synthetic Adam-state distributions (paper figures reproduced
   qualitatively).
2. `adamw_kernel3_fp8` (or equivalent) wraps the existing AdamW math with
   read-decompress / write-compress around `m` and `v`.
3. New flag `--fp8_optstate 1` toggles the path. Default off.
4. Allocation in `gpt2_allocate_state` and the load/save in `load_state` /
   `save_state` updated for the new layout. Checkpoint version bumped.
5. On GPT-2 124M, validation loss curve with FP8 optimizer states is within
   `0.05` of the FP32-optstate baseline at step 1000. Numbers in `STATUS.md`.
6. Memory measurement before/after logged in `STATUS.md`.

**Risks.** `v`'s narrow dynamic range (≈1e1) collapses to a single E4M3 grid
point without expansion. ZeRO-1 sharding changing the group boundaries. The
existing FP8 codec being subtly wrong (must finish Phase 1 exit #1 first).

**Out of scope.** Quantizing master weights. Activation quantization.

---

## Phase 3 — Activation quantization

**Goal.** Saved activations between layers (the things in `acts_memory`) are
stored in FP8 with mixed-granularity quantization (per-tensor or per-group as
appropriate). This is the largest memory win for long-context / large-batch
training.

**Entry criterion.** Phase 2 exit criteria are all met.

**Exit criteria.**

1. A clear written design choosing the granularity per activation tensor (LN
   inputs, attention outputs, residuals, FC up/down activations).
2. Forward path inserts quant-then-dequant on saved activations; backward path
   reads the dequantized values.
3. Tests: per-layer activation round-trip error logged on a fixed input,
   bounded.
4. On GPT-2 124M, validation loss curve with FP8 activations is within `0.10`
   of the BF16 baseline at step 1000.
5. Memory savings on activation memory measured and logged.

**Risks.** Numerical instability in attention or layernorm when activations
are FP8. Backward pass needing higher precision than forward.

**Out of scope.** Real INT8/FP8 GEMM kernels (Phase 4).

---

## Phase 4 — Real low-precision GEMM kernels (optional / stretch)

**Goal.** The matmuls themselves run in INT8 (with INT32 accumulation) or FP8
(with FP32 accumulation), not just on dequantized weights. This is where
real *speedup* (not just memory savings) comes from.

**Entry criterion.** Phase 3 exit criteria are met OR user explicitly chooses
to skip Phase 3.

**Exit criteria.**

1. cuBLASLt configured with `CUBLAS_COMPUTE_32I` (INT8) or FP8 compute mode.
2. Wrapper around `matmul_forward_cublaslt` and `matmul_backward` that handles
   the new compute dtype.
3. Wall-clock per-step measurement showing >= 1.2× speedup over Phase 1
   baseline on the same hardware.
4. Numerical accuracy preserved within tolerances established in earlier
   phases.

**Risks.** Hardware-specific (only Hopper / Ada / Blackwell support FP8 well;
INT8 GEMM has its own tensor-core requirements). cuBLASLt configuration
quirks. Loss of precision in attention path.

**Out of scope.** Anything not directly tied to the GEMM compute dtype.

---

## What success looks like end-to-end

After all four phases, this codebase trains GPT-2 124M in a configuration
where:

- The four large weight matrices live in INT8 / FP8 storage on device.
- The optimizer's `m` and `v` live in FP8 with COAT-style expansion.
- Saved activations are FP8 with appropriate granularity.
- The matmul tensor-core compute itself runs in INT8/FP8.
- The FP32 master copy is the only thing keeping convergence on track, and it
  is sharded via ZeRO-1 so its per-GPU cost is reduced.

The expected memory profile, for the four large weight tensors at GPT-2 124M
scale, drops from roughly 14 bytes/param (BF16 + FP32 master + FP32 m + FP32
v) to roughly 3 bytes/param (INT8 weight + FP8 m + FP8 v + FP32 master /
sharded), a >4× reduction on the dominant memory term, without measurable
accuracy loss vs. the BF16 baseline.
