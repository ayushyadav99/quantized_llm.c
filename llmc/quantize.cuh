/*
floatLOAD -> floatX weight quantization (W8A16).

Weights are stored as floatX with a per-row floatLOAD scale factor:
    scale[row] = max(|w[row, :]|) / target range
    q[row, col] = round(w[row, col] / scale[row])   clamped to target range

Activations remain in floatX.
*/
#pragma once
#include <assert.h>
#include <stdint.h>
#include <limits>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// Kernel 1: compute per-row scale = max(|w[row,:]|) / 127
// Launch: <<<rows, block_size>>> where block_size covers cols with stride loops.
__global__ void quantize_scale_kernel(float* scales, const floatX* w, int cols) {
    int row = blockIdx.x;
    const floatX* row_ptr = w + (size_t)row * cols;

    float thread_max = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf((float)row_ptr[c]));
    }
    float row_max = blockReduce<warpReduceMax>(thread_max, false, 0.0f);

    if (threadIdx.x == 0) {
        // guard against zero rows (e.g. freshly initialised weights)
        scales[row] = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    }
}

// Kernel 2: quantize floatX → int8 using the precomputed per-row scales.
// Must be launched after quantize_scale_kernel on the same stream.
__global__ void quantize_weights_kernel(int8_t* out, const floatX* in,
                                        const float* scales, int cols) {
    int row = blockIdx.x;
    float inv_scale = 1.0f / scales[row];
    const floatX* in_row  = in  + (size_t)row * cols;
    int8_t*       out_row = out + (size_t)row * cols;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = (float)in_row[c] * inv_scale;
        val = fmaxf(-127.0f, fminf(127.0f, val));
        out_row[c] = (int8_t)__float2int_rn(val);
    }
}

// Kernel 3: dequantize int8 → floatX using per-row scales.
// Called once per matmul during the forward pass; output goes into a scratch buffer.
__global__ void dequantize_weights_kernel(floatX* out, const int8_t* in,
                                          const float* scales, int cols) {
    int row = blockIdx.x;
    float scale = scales[row];
    const int8_t* in_row  = in  + (size_t)row * cols;
    floatX*       out_row = out + (size_t)row * cols;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out_row[c] = (floatX)((float)in_row[c] * scale);
    }
}

// Returns the maximum finite value of type T as a float.
// Uses std::numeric_limits so there are no hardcoded constants.
// float is directly supported; fp16/bf16 don't specialise std::numeric_limits,
// so we round-trip via float: cast FLT_MAX into T (which clamps to the type's
// actual maximum due to IEEE rounding), then back to float.
template<typename T>
__host__ __device__ inline float type_max_finite() {
    return (float)(std::numeric_limits<T>::max());
}

// Kernel 4: compute per-row scale so that max(|row|) maps to the floatX maximum.
// Launch: <<<rows, block_size>>> where block_size covers cols with stride loops.
__global__ void quantize_scale_kernel_load(float* scales, const floatLOAD* w, int cols) {
    const float target_max = type_max_finite<floatX>();
    int row = blockIdx.x;
    const floatLOAD* row_ptr = w + (size_t)row * cols;

    float thread_max = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = (float)row_ptr[c];
        // skip non-finite values that may exist in a corrupt checkpoint
        if (isfinite(v)) {
            thread_max = fmaxf(thread_max, fabsf(v));
        }
    }
    float row_max = blockReduce<warpReduceMax>(thread_max, false, 0.0f);

    if (threadIdx.x == 0) {
        // guard against zero rows (e.g. freshly initialised weights)
        scales[row] = (row_max > 0.0f) ? row_max / target_max : 1.0f;
    }
}

// Kernel 5: scale floatLOAD values into floatX range and cast.
// Must be launched after quantize_scale_kernel_load on the same stream.
__global__ void quantize_weights_kernel_load(floatX* out, const floatLOAD* in,
                                             const float* scales, int cols) {
    const float target_max = type_max_finite<floatX>();
    int row = blockIdx.x;
    float inv_scale = 1.0f / scales[row];
    const floatLOAD* in_row = in  + (size_t)row * cols;
    floatX*         out_row = out + (size_t)row * cols;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = (float)in_row[c] * inv_scale;
        // clamp to floatX representable range before casting
        val = fmaxf(-target_max, fminf(target_max, val));
        out_row[c] = (floatX)val;
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// Quantize a weight matrix in two serialised passes on the same stream:
//   pass 1 - compute scales,  pass 2 - write int8 values.
// `out`    : int8_t device buffer, same element count as `in`
// `scales` : float device buffer of length `rows`
// `in`     : floatX weight matrix, shape (rows, cols), row-major
void quantize_weights(int8_t* out, float* scales, const floatX* in,
                      int rows, int cols, cudaStream_t stream) {
    const int block = 256;
    quantize_scale_kernel<<<rows, block, 0, stream>>>(scales, in, cols);
    cudaCheck(cudaGetLastError());
    quantize_weights_kernel<<<rows, block, 0, stream>>>(out, in, scales, cols);
    cudaCheck(cudaGetLastError());
}

// Dequantize a weight matrix from int8 back to floatX.
// `out` must point to a floatX scratch buffer of size rows*cols elements.
void dequantize_weights(floatX* out, const int8_t* in, const float* scales,
                        int rows, int cols, cudaStream_t stream) {
    const int block = 256;
    dequantize_weights_kernel<<<rows, block, 0, stream>>>(out, in, scales, cols);
    cudaCheck(cudaGetLastError());
}

void quantize_loaded_weights(floatX* out, float* scales, const floatLOAD* in,
                      int rows, int cols, cudaStream_t stream) {
    const int block = 256;
    quantize_scale_kernel_load<<<rows, block, 0, stream>>>(scales, in, cols);
    cudaCheck(cudaGetLastError());
    quantize_weights_kernel_load<<<rows, block, 0, stream>>>(out, in, scales, cols);
    cudaCheck(cudaGetLastError());
}