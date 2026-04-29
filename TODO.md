# Quantization Roadmap: Extreme Memory Efficiency

This document outlines the next steps for pushing the `quantized_llm.c` project beyond "Beast-mode PTQ" into a fully quantized training and inference regime.

## Phase 1: Parameter & Weight Optimization
- [ ] **Remove FP32 Master Weights**
  - Currently, `use_master_weights = 1` maintains a full FP32 copy for updates.
  - *Goal:* Perform updates directly on `floatX` (BF16/FP16) or quantized weights to save 4 bytes per parameter.
- [ ] **Quantize Embedding Weights (`wte` & `wpe`)**
  - Currently excluded in `ptq_should_quantize_tensor` due to random-access lookup patterns.
  - *Goal:* Implement INT8 quantization for embeddings and update `encoder_forward` to handle dequantization.

## Phase 2: Optimizer State Compression
- [ ] **8-bit Optimizer States**
  - Currently, AdamW moments ($m$ and $v$) are stored in FP32.
  - *Goal:* Change optimizer buffers to INT8/FP8 with dynamic scaling to reduce optimizer memory overhead by 75%.
- [ ] **Stochastic Rounding for Updates**
  - Ensure that updates to low-precision weights don't "vanish" by implementing stochastic rounding.

## Phase 3: Activation & Flow Quantization
- [ ] **Quantize Activations**
  - Currently, activations like `ln1`, `atty`, and `fch` are stored in `floatX`.
  - *Goal:* Implement INT8/FP8 quantization for all intermediate activations to significantly reduce VRAM usage during the backward pass.
- [ ] **Fused Quantization Kernels**
  - Merge quantization/dequantization steps into existing kernels (e.g., fused LayerNorm + Quantize) to minimize memory bandwidth overhead.

## Phase 4: Architectural Improvements
- [ ] **Type-Agnostic Core**
  - Refactor the codebase to support arbitrary precision modes via template parameters or dynamic dispatch.
  - *Goal:* Easily switch between FP32, BF16, FP16, FP8, and INT8 without modifying core logic.
- [ ] **Validation & Correctness Suite**
  - Implement a rigorous verification script to compare low-precision training stability against FP32 baselines.
  - Monitor loss divergence and gradient flow health.
