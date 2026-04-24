# `train_gpt2.cu` Precision And Quantization Map

This note is meant to answer one question before you start quantizing:

Where is the code already doing low-precision work, and where would quantization need to hook in?

Current status:

- `train_gpt2.cu` now has an optional row-wise PTQ shadow behind `--ptq`
- all learnable parameter tensors are quantized to `int8` or `fp8`
- forward and backward read a dequantized mirror of that shadow
- optimizer state and optional master weights remain in FP32
- the PTQ shadow is regenerated after optimizer updates and after restore-from-master flows

## 1. What "mixed precision" means in this codebase

This file does not use PyTorch AMP or runtime autocast. Precision is chosen at compile time through `floatX` in [`llmc/cuda_common.h`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/llmc/cuda_common.h).

- `ENABLE_FP32` -> `floatX = float`
- `ENABLE_FP16` -> `floatX = half`
- default -> `floatX = __nv_bfloat16`

The current training path is effectively:

- Parameters stored as `floatX`
- Activations stored mostly as `floatX`
- Some reduction/statistics tensors stored as `float`
- GEMMs executed through cuBLASLt with FP32 accumulation
- AdamW moments `m` and `v` stored as `float`
- Optional FP32 `master_weights` used as the authoritative optimizer copy

So the important idea is:

Low precision is mainly used for storage/bandwidth. Critical reductions and optimizer math are still FP32.

## 2. Where the important precision decisions happen

### Type selection

[`llmc/cuda_common.h`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/llmc/cuda_common.h)

- `floatX` is defined here.
- `PRECISION_MODE` is also defined here.
- This choice propagates into model weights, activations, gradients, and many kernels.

### cuBLAS matmul compute type

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu) in `common_start()`

- `cublas_compute` is set here.
- FP32 builds can use TF32 on supported GPUs.
- BF16 builds still route GEMMs through cuBLASLt with FP32 accumulation configured in [`llmc/matmul.cuh`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/llmc/matmul.cuh).

### Parameter and activation storage

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu)

- `ParameterTensors` stores model weights as `floatX`
- `ActivationTensors` stores most activations as `floatX`
- layernorm stats and losses stay `float`

This is the first place to inspect if you want quantized storage formats later.

### Optimizer precision

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu) in `gpt2_allocate_state()` and `gpt2_update()`

- `m_memory` and `v_memory` are always `float`
- `master_weights` is optional but enabled by default
- update math reads gradients, promotes to float, updates FP32 optimizer state, updates an FP32 master parameter, then rounds back into `floatX`

That is the strongest existing pattern you want to preserve when adding quantization: keep the update path numerically safer than the forward storage path.

## 3. Forward pass: where quantization would matter most

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu) in `gpt2_forward()`

The performance-critical points are the repeated calls to:

- `matmul_forward_cublaslt(...)`
- `attention_forward_cudnn(...)` or `attention_forward(...)`
- `fused_residual_forward5(...)`
- `layernorm_forward(...)`

The most important quantization-sensitive tensors are still the linear layers:

- `qkvw`
- `attprojw`
- `fcw`
- `fcprojw`
- final projection using `wte`

Why these matter most:

- They dominate FLOPs
- They already go through a narrow abstraction (`matmul_forward_cublaslt`)
- Even though the current PTQ path quantizes all learnable tensors, these are still the dominant accuracy/performance seams

## 4. Backward pass: what gets harder

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu) in `gpt2_backward_and_reduce()`

Backward depends on:

- the runtime weights used by forward
- forward activations
- gradients in `floatX`
- `matmul_backward(...)`

The current PTQ implementation avoids custom quantized backward kernels by making both forward and backward read the same dequantized runtime mirror through `gpt2_get_active_params()`.

That keeps the training path simple:

1. update canonical trainable weights in the usual path
2. rebuild the quantized shadow
3. dequantize that shadow into the runtime mirror
4. use that mirror consistently in forward and backward

## 5. Optimizer/update path: what already resembles quantization workflow

[`llmc/adamw.cuh`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/llmc/adamw.cuh)

Current update path:

1. Read `grads_memory[idx]` and convert to `float`
2. Update `m` and `v` in `float`
3. Read old parameter from `master_weights` if available, otherwise from low precision storage
4. Compute new parameter in `float`
5. Stochastically round back into `params_memory` (`floatX`)

This is already conceptually similar to quantization-aware storage:

- one higher-precision source of truth
- one cheaper runtime representation

If you later add int8/int4 weights, you will likely want the same split:

- master/trainable FP32 or BF16 weights
- quantized packed weights for forward compute

## 6. Checkpointing constraints you need to know before quantizing

[`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu) in `gpt2_build_from_checkpoint()`, `save_state()`, and `load_state()`

Important facts:

- model checkpoint `.bin` stores raw `floatX` weights
- optimizer state file stores `m`, `v`, and optional `master_weights` as `float`
- reload assumes raw bytes match the compiled precision mode

The current implementation does not serialize the PTQ shadow.

Instead:

- checkpoints keep storing the canonical `floatX` parameters
- optimizer state still stores `m`, `v`, and optional `master_weights`
- PTQ data is regenerated after load when needed

## 7. Concrete quantization entry points

The implemented PTQ entry points are:

1. [`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu)
   `gpt2_prepare_ptq()` quantizes every parameter tensor and rebuilds the dequantized runtime mirror.
2. [`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu)
   `gpt2_get_active_params()` selects the dequantized PTQ mirror for forward/backward when PTQ is enabled.
3. [`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu)
   `gpt2_update()` regenerates the PTQ shadow after parameter updates.
4. [`train_gpt2.cu`](/Users/kvlnraju/College/courses/semester-4/efficient-ai/project/quantized_llm.c/train_gpt2.cu)
   `load_state()` reuses the normal parameter restore flow, and PTQ is rebuilt from canonical weights rather than checkpointed separately.

## 8. Recommended implementation order

If you want to extend this further without turning it into a research detour, the order should be:

1. Keep the current row-wise shadow path as the single source of PTQ behavior
2. Optimize shadow rebuild/dequantization cost if it becomes a bottleneck
3. Decide whether biases and norm parameters should remain quantized or be exempted
4. Only after that, consider quantized activations or direct quantized GEMMs

That keeps the design aligned with what this code already does well: low-precision storage with higher-precision update math.
