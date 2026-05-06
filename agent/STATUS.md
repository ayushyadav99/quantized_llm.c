# STATUS.md — Where the project is right now

> Update this file every time something material changes. One-line entries are
> fine. The next agent / next conversation reads this first.

**Last updated:** 2026-04-30

**Active branch:** `FP8-fix` (tracks `origin/pragnay/FP8_QAT_fixes_plotting_train_losses`)

**Current phase:** Phase 1 — Weight-only QAT, clean. See [`PHASES.md`](./PHASES.md).

---

## What is done

- Per-row symmetric absmax INT8 weight quantization is wired into the four
  large transformer matrices (`qkvw`, `attprojw`, `fcw`, `fcprojw`).
- FP8 E4M3 weight quantization codec is implemented as an alternative path
  (`--ptq_precision fp8`).
- Forward and backward paths dequantize-on-demand into a per-layer scratch
  buffer (`scratch_dequant`).
- Optimizer keeps FP32 master weights, runs AdamW in FP32, and re-quantizes
  the master into INT8 / FP8 storage after every step.
- Beast-mode checkpoint format (versions 7-10) saves and restores the
  quantized layout.
- A baseline run on GPT-2 124M trains end-to-end without crashing in this
  configuration.

## What is in progress

- Nothing is currently in progress. Phase 1 cleanup tasks below have not been
  started.

## Phase 1 cleanup — outstanding work

Ranked. Top of the list is the highest-value next thing to do.

1. **FP32 master init from original weights, not from the quantized version.**
   Today the master is initialized from the dequantized INT8, which means it
   starts already-rounded. Fix in `gpt2_update`'s init-state branch (or move
   master allocation before `gpt2_prepare_ptq` collapses params memory).
2. **INT8 + FP8 codec round-trip unit test in `dev/test/`.** Must run in under
   a minute, must produce a clear PASS / FAIL line. Compare FP8 codec against
   `cuda_fp8.h` reference values across a sweep of inputs.
3. **Make the first quantization consistent with later re-quantizations.** The
   first call to `gpt2_prepare_ptq` quantizes from BF16 source; every later
   call quantizes from FP32 master. Make both paths use the FP32 master when
   one exists.
4. **Log per-tensor quantization noise** (mean
   `|W_master - dequant(quant(W_master))| / |W_master|`) every
   `val_loss_every` steps. This is the QAT equivalent of a loss curve and
   should appear in the run log.
5. **Establish baseline numbers** on GPT-2 124M: BF16-no-quant val loss at
   steps 0/100/500/1000, hellaswag accuracy if available, wall-clock per
   step. Record here under "Baseline numbers" once measured.

## What is next (after Phase 1 closes)

- Phase 2: COAT-style FP8 optimizer state compression. Do not start until
  Phase 1 exits cleanly. See `PHASES.md` for entry/exit criteria.

---

## Baseline numbers (fill in when measured)

| Configuration | val loss @ 100 | val loss @ 500 | val loss @ 1000 | hellaswag | s/step |
| --- | --- | --- | --- | --- | --- |
| BF16, no quant (baseline) | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| BF16 + INT8 weight QAT | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| BF16 + FP8 weight QAT | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |

Until these are filled in, "does it work?" cannot be answered numerically.
Filling these in is part of Phase 1 exit criterion #5.

---

## Open questions / decisions pending

- Whether to also quantize `wte` (currently kept BF16 because it's used as a
  gather *and* as the classifier weight — but it's the largest unquantized
  matmul weight in the model). Listed in `TODO.md` as item 2. Defer until
  Phase 1 closes.
- Whether to ship the codec test as a CI step or as a manual one. Probably
  manual to start; CI later.
- ZeRO-1 master sharding: already supported by the optimizer code path; not
  yet exercised in the QAT runs. Worth turning on once Phase 1 baseline
  numbers exist.

---

## Recent changes log

> One-line entries with date. Most recent at top.

- 2026-04-30 — Added `agent/ARCHITECTURE.md`: Mermaid flow-charts of the
  training step, quantization data flow, optimizer with re-quant, codecs,
  and checkpoint layout, plus a §13 update protocol. Wired references from
  `AGENTS.md` (priority list, "Files you must keep up to date", Pointers).
  No code changes.
- 2026-04-30 — Reorganized agent docs: kept `AGENTS.md` at repo root (auto-
  discovery entry point), moved `PHASES.md`, `STATUS.md`, `AGENT_GUIDE.md`,
  and `QAT_REPORT.md` into `agent/` subfolder. Updated cross-references.
- 2026-04-30 — Created `AGENTS.md`, `PHASES.md`, `STATUS.md`, `AGENT_GUIDE.md`
  for agent orientation. No code changes.
- 2026-04-30 — Added `QAT_REPORT.md` (deep-dive analysis of current
  quantization implementation). No code changes.
- *(earlier history is in `git log`)*
