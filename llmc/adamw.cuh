/*
AdamW kernel
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // fetch the old value of this parameter as a float, from either source
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // update this parameter
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    // update our low precision version of the parameters using stochastic rounding
    // this will be used in the next forward pass
    stochastic_rounding(param, &params_memory[idx], seed);
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                          ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }
    params_memory += blockIdx.y * w_stride; // adjust for layer offset
    master_params_memory += blockIdx.y * s_stride;
    stochastic_rounding(master_params_memory[idx], &params_memory[idx], seed);
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                         m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                         learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                         grad_scale, seed);
    cudaCheck(cudaGetLastError());
}

template <typename Tp>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream) {
    int block_size = 512; // must match block size of adamw_update so that RNG also matches
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                             (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// COAT FP8 optimizer state kernel (Phase 3)
// One CUDA block per group of COAT_GROUP_SIZE parameters.
// Each thread handles one parameter: dequantize → AdamW math → group reduce → requantize.
// ----------------------------------------------------------------------------
#ifdef COAT_OPTIM
#include "coat_fp8_optim.cuh"

// Compute group k and scale from the group's max and min absolute values.
// Mirrors coat_quantize_group but is callable per-block after the shared-memory reduction.
__device__ inline void coat_group_meta(float max_abs, float min_abs,
                                        float* out_k, float* out_scale) {
    if (max_abs == 0.0f) { *out_k = 1.0f; *out_scale = 1.0f; return; }
    if (min_abs >= max_abs) min_abs = max_abs;
    float k = coat_compute_k(max_abs / min_abs);
    if (max_abs < 1.0f) {
        // Guard: cap k so max_abs^k doesn't underflow float32 (see coat_fp8_optim.cuh).
        k = fminf(k, -43.668f / logf(max_abs));
    }
    float max_expanded = coat_expand(max_abs, k);
    *out_k    = k;
    *out_scale = (max_expanded > 0.0f) ? COAT_FP8_MAX / max_expanded : 1.0f;
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3_coat(
    Tp*      params_memory,
    float*   master_params_memory,
    Tg*      grads_memory,
    uint8_t* m_fp8,   uint8_t* v_fp8,
    float*   m_scales, float*  v_scales,
    float*   m_kfactors, float* v_kfactors,
    size_t   num_parameters,
    ptrdiff_t w_stride, ptrdiff_t g_stride,
    ptrdiff_t s_stride, ptrdiff_t meta_stride,
    float learning_rate, float beta1, float beta2,
    float beta1_correction, float beta2_correction,
    float eps, float weight_decay, float grad_scale, unsigned int seed
) {
    // blockIdx.x = group index within this slice
    // blockIdx.y = slice index (layer in multi-layer tensors)
    // threadIdx.x = parameter index within the group  [0, COAT_GROUP_SIZE)
    size_t group_idx = blockIdx.x;
    int    local_idx = threadIdx.x;
    size_t param_idx = group_idx * (size_t)COAT_GROUP_SIZE + local_idx;
    bool in_bounds   = (param_idx < num_parameters);

    // Apply per-slice pointer offsets (same role as blockIdx.y * stride in adamw_kernel3).
    params_memory  += blockIdx.y * w_stride;
    grads_memory   += blockIdx.y * g_stride;
    m_fp8          += blockIdx.y * s_stride;
    v_fp8          += blockIdx.y * s_stride;
    m_scales       += blockIdx.y * meta_stride;
    v_scales       += blockIdx.y * meta_stride;
    m_kfactors     += blockIdx.y * meta_stride;
    v_kfactors     += blockIdx.y * meta_stride;
    if (master_params_memory) master_params_memory += blockIdx.y * s_stride;

    // --- 1. Dequantize m and v for this thread ---
    // scale=0 means uninitialised (first step) → treat as zero moment.
    float m = 0.0f, v = 0.0f;
    if (in_bounds) {
        float sc_m = m_scales[group_idx],   k_m = m_kfactors[group_idx];
        float sc_v = v_scales[group_idx],   k_v = v_kfactors[group_idx];
        if (sc_m > 0.0f)
            m = coat_unexpand(coat_fp8_decode(m_fp8[param_idx]) / sc_m, k_m);
        if (sc_v > 0.0f)
            v = coat_unexpand(coat_fp8_decode(v_fp8[param_idx]) / sc_v, k_v);
    }

    // --- 2. AdamW math in FP32 (identical to existing kernel) ---
    if (in_bounds) {
        float grad = grad_scale * (float)grads_memory[param_idx];
        m = lerp(grad, m, beta1);
        v = lerp(grad * grad, v, beta2);
        float m_hat    = m / beta1_correction;
        float v_hat    = v / beta2_correction;
        float old_param = master_params_memory
                          ? master_params_memory[param_idx]
                          : (float)params_memory[param_idx];
        float param = old_param - learning_rate *
                      (m_hat / (sqrtf(v_hat) + eps) + weight_decay * old_param);
        stochastic_rounding(param, &params_memory[param_idx], seed);
        if (master_params_memory) master_params_memory[param_idx] = param;
    }

    // Shared memory for two sequential reductions (reused for m then v).
    __shared__ float sm_max[COAT_GROUP_SIZE];
    __shared__ float sm_min[COAT_GROUP_SIZE];

    // --- 3. Quantize updated m back to FP8 ---
    {
        float abs_m = in_bounds ? fabsf(m) : 0.0f;

        // Max reduction across the group.
        sm_max[local_idx] = abs_m;
        __syncthreads();
        for (int s = COAT_GROUP_SIZE / 2; s > 0; s >>= 1) {
            if (local_idx < s)
                sm_max[local_idx] = fmaxf(sm_max[local_idx], sm_max[local_idx + s]);
            __syncthreads();
        }
        float max_m = sm_max[0];

        // Min reduction (out-of-bounds and zero threads contribute the sentinel max_m).
        sm_min[local_idx] = (abs_m > 0.0f) ? abs_m : max_m;
        __syncthreads();
        for (int s = COAT_GROUP_SIZE / 2; s > 0; s >>= 1) {
            if (local_idx < s)
                sm_min[local_idx] = fminf(sm_min[local_idx], sm_min[local_idx + s]);
            __syncthreads();
        }
        float min_m = sm_min[0];

        float new_k_m, new_scale_m;
        coat_group_meta(max_m, min_m, &new_k_m, &new_scale_m);

        if (in_bounds) {
            float expanded = coat_expand(m, new_k_m) * new_scale_m;
            expanded = fmaxf(fminf(expanded, COAT_FP8_MAX), -COAT_FP8_MAX);
            m_fp8[param_idx] = coat_fp8_encode(expanded);
        }
        if (local_idx == 0) {
            m_scales[group_idx]   = new_scale_m;
            m_kfactors[group_idx] = new_k_m;
        }
    }
    __syncthreads(); // free shared memory before reuse for v

    // --- 4. Quantize updated v back to FP8 (same pattern as m) ---
    {
        float abs_v = in_bounds ? fabsf(v) : 0.0f;

        sm_max[local_idx] = abs_v;
        __syncthreads();
        for (int s = COAT_GROUP_SIZE / 2; s > 0; s >>= 1) {
            if (local_idx < s)
                sm_max[local_idx] = fmaxf(sm_max[local_idx], sm_max[local_idx + s]);
            __syncthreads();
        }
        float max_v = sm_max[0];

        sm_min[local_idx] = (abs_v > 0.0f) ? abs_v : max_v;
        __syncthreads();
        for (int s = COAT_GROUP_SIZE / 2; s > 0; s >>= 1) {
            if (local_idx < s)
                sm_min[local_idx] = fminf(sm_min[local_idx], sm_min[local_idx + s]);
            __syncthreads();
        }
        float min_v = sm_min[0];

        float new_k_v, new_scale_v;
        coat_group_meta(max_v, min_v, &new_k_v, &new_scale_v);

        if (in_bounds) {
            float expanded = coat_expand(v, new_k_v) * new_scale_v;
            expanded = fmaxf(fminf(expanded, COAT_FP8_MAX), -COAT_FP8_MAX);
            v_fp8[param_idx] = coat_fp8_encode(expanded);
        }
        if (local_idx == 0) {
            v_scales[group_idx]   = new_scale_v;
            v_kfactors[group_idx] = new_k_v;
        }
    }
}

// Host-side wrapper — same call convention as adamw_update but with FP8 m/v storage.
template <typename Tp, typename Tg>
void adamw_update_coat(
    Tp*      params_memory, float*   master_params_memory, Tg* grads_memory,
    uint8_t* m_fp8,         uint8_t* v_fp8,
    float*   m_scales,      float*   v_scales,
    float*   m_kfactors,    float*   v_kfactors,
    size_t   num_parameters,
    ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
    float learning_rate, float beta1, float beta2, int t,
    float eps, float weight_decay, float grad_scale, unsigned int seed,
    cudaStream_t stream
) {
    size_t    num_groups   = CEIL_DIV(num_parameters, (size_t)COAT_GROUP_SIZE);
    ptrdiff_t meta_stride  = (ptrdiff_t)CEIL_DIV(s_stride, COAT_GROUP_SIZE);
    float     beta1_corr   = 1.0f - powf(beta1, t);
    float     beta2_corr   = 1.0f - powf(beta2, t);
    adamw_kernel3_coat<<<dim3(num_groups, num_slices), COAT_GROUP_SIZE, 0, stream>>>(
        params_memory, master_params_memory, grads_memory,
        m_fp8, v_fp8, m_scales, v_scales, m_kfactors, v_kfactors,
        num_parameters, w_stride, g_stride, s_stride, meta_stride,
        learning_rate, beta1, beta2, beta1_corr, beta2_corr,
        eps, weight_decay, grad_scale, seed
    );
    cudaCheck(cudaGetLastError());
}
#endif // COAT_OPTIM
