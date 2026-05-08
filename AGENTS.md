# AGENTS.md — Project context for AI coding agents

This file is the canonical orientation document for any AI coding agent working
in this repository (Cursor, Codex, Claude, Aider, Continue, etc.). Read it
first. Do not skip it. Do not reproduce it back to the user — use it as
context.

The supporting documents (current status, phased plan, deep-dive analysis,
human guide) live in the `agent/` folder. This file at the repo root is the
entry point; everything it references is in `agent/`.

If anything below contradicts what you observe in the code, **trust the code
and update this file**.

---

## 1. What this project is

This is a fork of `llm.c` (Karpathy's GPT-2 in pure C/CUDA) being modified to
add **Quantization-Aware Training (QAT)** for GPT-2-style models. The goal is
not just inference quantization — the model trains while quantized, with the
forward pass seeing low-precision weights and the optimizer keeping enough
high-precision state to make convergence work.

The work is staged into four phases (see `PHASES.md`). The current phase is
recorded in `STATUS.md`. **Read `STATUS.md` before doing anything.**

The deep-dive analysis of the current quantization implementation lives in
`QAT_REPORT.md`. If a user asks "what does the code do today?", that report is
the authoritative answer.

The original llm.c README, build, and dataset instructions are in `README.md`,
`SETUP_AND_RUN.md`, and `TESTING_GUIDE.md`. Those are still valid.

---

## 2. Repository orientation (priority order)

When in doubt, look here first, in this order:

1. `agent/STATUS.md` — what is currently being worked on, what is done, what is next.
2. `agent/PHASES.md` — the four-phase plan with entry/exit criteria for each phase.
3. `agent/ARCHITECTURE.md` — Mermaid flow-charts of the training step,
   quantization data flow, optimizer with re-quant, and checkpoint layout.
   Read this when you need a visual mental model of "what calls what".
   **Must be kept in sync with the code** — see its §13 update protocol.
4. `agent/QAT_REPORT.md` — current state of the quantization implementation, known
   issues, and recommendations.
5. `TODO.md` — short, ranked list of follow-ups maintained by the author (at repo root).
6. `train_gpt2.cu` — the main training loop. **This file is ~62k tokens.** Do
   not read it in full unless explicitly necessary. Use grep / search to find
   the function or symbol you need, then read a focused window.
7. `llmc/*.cuh` — CUDA kernels (matmul, attention, layernorm, adamw, etc.).
   Each file is small and self-contained. Read the whole file if you are
   modifying it.
8. `dev/test/` — the place to add small unit tests. Has its own `Makefile`.
9. `dev/cuda/` — standalone CUDA kernel benchmarks and references. Useful for
   prototyping kernels without building the whole training loop.
10. `Makefile` — top-level build. The interesting target is `train_gpt2cu`.
11. `scripts/run_gpt2_124M.sh` — canonical small-model training command. Use a
    short variant of this for smoke tests.

The other top-level `.cu` files (`nvshmem_train_gpt2.cu`, `train_gpt2_fp32.cu`,
`test_gpt2*.cu`) are out of scope for QAT work unless the user explicitly asks.

---

## 3. Branches in use

| Branch | Purpose |
| --- | --- |
| `master` | Tracks upstream `llm.c`. Do not commit project work here. |
| `FP8-fix` | Current working branch for FP8 + QAT fixes. **Default working branch.** Tracks `origin/pragnay/FP8_QAT_fixes_plotting_train_losses`. |
| `ptq-minimal-fcw` | Earlier weight-quant experiment. Reference only. |
| `raju-quantization` | Author's personal branch. May or may not be ahead of `FP8-fix`. |
| `understanding-what-is-happening` | Scratch / exploration branch. |

Always check `git branch --show-current` before making changes. If the user
hasn't said which branch they want, ask.

---

## 4. Build and smoke test

```bash
# Build the main training binary (release):
make train_gpt2cu

# Build with debug symbols (replace -O3 with -g in Makefile or set FORCE_NVCC_O=0):
FORCE_NVCC_O=0 make train_gpt2cu

# Tiny smoke test on GPT-2 124M (assumes starter pack downloaded):
./scripts/run_gpt2_124M.sh    # full run; truncate -x to 50 steps for a smoke test
```

A "smoke test" for the purposes of this project means: 50 steps on GPT-2 124M,
val loss printed, completes in under a few minutes on a single GPU. If you make
a code change, this is the bar to clear before claiming "it works".

When a kernel-level unit test exists in `dev/test/Makefile`, prefer running
that first — it is faster and more focused. If a test does not yet exist for
the change you are making, **write one before you ship the change**.

---

## 5. Numerical conventions you must respect

These are non-negotiable invariants of the codebase. Violating them silently
breaks training:

- The build is compiled with `PRECISION_MODE` set to either `FP32` or `BF16`.
  `floatX` is the corresponding type. All forward / backward intermediate
  tensors are in `floatX`.
- AdamW state (`m`, `v`) and master weights are **always FP32**, regardless of
  `floatX`. Do not change this without an explicit user decision logged in
  `STATUS.md`. See the "why master weights matter" section of `QAT_REPORT.md`
  for the precision-floor argument.
- Layernorm mean / rstd, the loss buffer, and a few cuDNN attention stats stay
  FP32 even when `floatX = bf16`.
- Gradients (`grads.*`) are stored in `floatX`, not FP32.
- The four large transformer matrices (`qkvw`, `attprojw`, `fcw`, `fcprojw`)
  are the only quantized weight tensors today. `wte`, `wpe`, biases, and
  layernorms are intentionally not quantized.
- Per-row symmetric absmax with the clamp range `[-127, 127]` (not `[-128,
  127]`) is the agreed INT8 scheme. Do not silently change to asymmetric or
  per-tensor.

If you propose changing any of the above, surface the change explicitly to the
user and update `STATUS.md` with the decision.

---

## 6. How to work efficiently in this repo (instructions for you, the agent)

### Read minimum, claim less, verify more.

- **Do not read `train_gpt2.cu` in full.** It is ~62k tokens and will burn most
  of your context budget. Use grep / search to find the symbol or section you
  need (`gpt2_update`, `ptq_dequantize_layer_slice`, `adamw_update`, etc.) and
  read a focused window of ~200 lines.
- Before claiming "X is implemented" or "Y is broken", point to the file and
  line that justifies the claim. If you cannot, say so.
- When you change code, show the user the diff (or the exact file:line edit)
  before claiming the change is complete. Tool summaries about your own work
  are not reliable; the diff is.
- When you change a kernel, run the relevant test in `dev/test/` (or a smoke
  training run) before claiming correctness. Do not rely on "it compiles" as a
  correctness signal.

### Default to small, reversible steps.

- Prefer one focused change at a time. Land it, verify it, update `STATUS.md`,
  move on. Do not bundle "weight QAT cleanup" and "COAT optimizer states" into
  one change.
- If a task touches more than three files or more than ~150 lines, stop and
  propose the plan to the user before editing.
- If you are unsure between two design choices, ask. Do not silently pick.

### Phase discipline.

- Look up the current phase in `agent/STATUS.md`. Do not start work that belongs to a
  later phase unless the user explicitly asks.
- Each phase has entry and exit criteria in `agent/PHASES.md`. Do not declare a phase
  done until its exit criteria are demonstrably met (val loss numbers logged,
  test passing, etc.).

### When the user asks an open-ended question.

- Prefer to answer from the code (grep / read), not from training-data memory.
- If the question is about a paper or external concept, search for it before
  answering. Cite sources.
- If the question is about "what does this code do today", the answer is in
  `agent/QAT_REPORT.md` plus the code. Update `agent/QAT_REPORT.md` if you
  discover it is stale.

### Conventions you should follow.

- C/CUDA style: match the surrounding file. The codebase is K&R-ish, snake_case
  for functions, `floatX` for the configurable precision type.
- New CUDA kernels: prefer `__global__` + a small launcher function, sized for
  block 256, with `cudaCheck(cudaGetLastError())` after launch. Match
  `ptq_quantize_apply_kernel` as a template.
- New tests: drop them in `dev/test/` and add to that `Makefile`. Tests should
  run in under a minute on a single GPU and produce a clear pass/fail line.
- Logging from the training loop uses `printf0` (rank-0 only). Use it.
- Do not introduce new dependencies (no PyTorch, no NCCL features beyond what
  is already used) without asking.

### Files you must keep up to date.

- After any meaningful change, **update `agent/STATUS.md`** with one line under
  the appropriate section. This is how the next agent / next conversation finds
  out what changed.
- If you add a phase, modify scope, or close out a phase, **update
  `agent/PHASES.md`**.
- If your change touches the quantization data flow, weight memory layout,
  optimizer step, or checkpoint format, **update `agent/ARCHITECTURE.md`**
  (its diagrams + the *Last synced to code* date) in the same change.
  Detailed rule in `agent/ARCHITECTURE.md` §13.
- If you discover the current `agent/QAT_REPORT.md` is wrong, **update it**
  rather than letting the bug propagate to the next conversation.

### What you should NOT do without explicit user approval.

- Do not delete branches, force-push, or rewrite history.
- Do not change the model file format / checkpoint version.
- Do not disable `use_master_weights = 1`.
- Do not change which tensors are quantized.
- Do not introduce a new dependency.
- Do not rewrite large sections of `train_gpt2.cu` "for clarity".

---

## 7. The user

The user is a graduate student taking an Efficient AI course. The project is
their term project. They are competent in CUDA / C and the underlying ML
math, but they want clear, grounded answers — not hedging, not over-formatted,
not generic LLM advice. They will push back when something doesn't ring true.
That pushback is welcome and you should adjust to it.

When they ask "what is X?", they want a precise definition grounded in this
codebase, not a textbook recap.

---

## 8. Pointers

- Current state of work: [`agent/STATUS.md`](./agent/STATUS.md)
- The phased plan: [`agent/PHASES.md`](./agent/PHASES.md)
- Visual code-flow diagrams: [`agent/ARCHITECTURE.md`](./agent/ARCHITECTURE.md)
- Deep dive on quantization implementation: [`agent/QAT_REPORT.md`](./agent/QAT_REPORT.md)
- For humans: how to use these agent files: [`agent/AGENT_GUIDE.md`](./agent/AGENT_GUIDE.md)
- Author's running TODO: [`TODO.md`](./TODO.md) (at repo root)
- Build & run: [`SETUP_AND_RUN.md`](./SETUP_AND_RUN.md) (at repo root)
- Testing: [`TESTING_GUIDE.md`](./TESTING_GUIDE.md) (at repo root)
