# Quantization TODO List

1. **Remove master weights**
   - Currently, `use_master_weights = 1` maintains a full FP32 copy for updates.
   - *Task:* Perform updates directly on `floatX` (BF16/FP16) or quantized weights to save 4 bytes per parameter.

2. **Change embedding weights to INT8**
   - Currently excluded in `ptq_should_quantize_tensor` due to random-access lookup patterns.
   - *Task:* Implement INT8 quantization for `wte` and `wpe` and update `encoder_forward` to handle dequantization.

3. **Make it type agnostic and see if code is correct**
   - Refactor the codebase to support arbitrary precision modes (FP32, BF16, FP16, FP8, INT8).
   - *Task:* Ensure switching precision doesn't require logic changes and verify loss convergence against FP32 baselines.

4. **Change the optimizer states to INT8/FP8**
   - AdamW moments ($m$ and $v$) are currently stored in FP32.
   - *Task:* Compress optimizer buffers to 8-bit with dynamic scaling to reduce VRAM overhead by 75%.

5. **Change the Activations to INT8/FP8**
   - Activations (e.g., `ln1`, `atty`, `fch`) are currently stored in `floatX`.
   - *Task:* Implement quantization for intermediate activations to save memory during the backward pass.
