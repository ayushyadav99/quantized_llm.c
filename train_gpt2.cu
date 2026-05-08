/*
GPT-2 Transformer Neural Net training loop. See README.md for usage.
*/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_fp8.h>
// ----------- CPU utilities -----------
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
// defines: evalloader_init, evalloader_reset, evalloader_next_batch, evalloader_free
#include "llmc/dataloader.h"
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "llmc/rand.h"
// defines: lr_scheduler_init, get_learning_rate
#include "llmc/schedulers.h"
// defines: sample_softmax, random_f32
#include "llmc/sampler.h"
// defines: logger_init, logger_log_eval, logger_log_val, logger_log_train
#include "llmc/logger.h"
// defines: get_flops_promised
#include "llmc/mfu.h"
// defines: OutlierDetector, init_detector, update_detector
#include "llmc/outlier_detector.h"
// ----------- GPU utilities -----------
// defines:
// WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE
// NVTX_RANGE_FN
#include "llmc/cuda_common.h"
// defines:
// Packed128, f128, x128
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged
#include "llmc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "llmc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"
// defines: matmul_cublaslt, matmul_forward, matmul_backward, gelu_forward, gelu_backward_inplace
#include "llmc/matmul.cuh"
#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "llmc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "llmc/attention.cuh"
#endif
// defines: fused_classifier
#include "llmc/fused_classifier.cuh"
// defines: adamw_kernel3
#include "llmc/adamw.cuh"
// defines: coat_quantize_group, coat_dequantize_group, coat_expand, coat_unexpand
// pass --optim_quant fp8|int8|int4 at runtime to enable quantized optimizer states
#include "llmc/coat_fp8_optim.cuh"
// defines: global_norm_squared
#include "llmc/global_norm.cuh"
// ----------- Multi-GPU support -----------
// defines: ncclFloatX, ncclCheck, MultiGpuConfig, ShardInfo
// defines: printf0, multi_gpu_config
// defines: multi_gpu_config_init, multi_gpu_config_free
// defines: set_zero_configs, multi_gpu_cpu_float_sum, multi_gpu_barrier
// defines: multi_gpu_get_shard_offset, multi_gpu_async_reduce_gradient
#include "llmc/zero.cuh"

// ----------------------------------------------------------------------------
// global vars for I/O
char filename_buffer[512];

// ----------------------------------------------------------------------------
// global vars containing information about the GPU this process is running on
cudaDeviceProp deviceProp; // fills in common_start()
cudaStream_t main_stream;
// buffer size to use for device <-> disk io
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

enum PTQPrecision {
    PTQ_PRECISION_NONE = 0,
    PTQ_PRECISION_INT8 = 1,
    PTQ_PRECISION_FP8 = 2,
    PTQ_PRECISION_INT4 = 3,
};

constexpr const int NUM_PARAMETER_TENSORS = 16;

typedef struct {
    uint8_t* qvalues; // quantized payload, row-major. int4 stores two logical values per byte.
    // Scales layout: [num_layers, rows_per_layer, num_groups] in row-major float.
    // num_groups = cols / group_size. When group_size == cols this collapses to one
    // scale per row (the default for int8/fp8). For int4 we typically use group_size < cols.
    float* scales;
    int num_layers;
    int rows_per_layer;
    int cols;
    int group_size;       // grouping along cols. Always in [1, cols] and divides cols.
    size_t qvalue_bytes;
    size_t scale_count;   // total scales stored = num_layers * rows_per_layer * (cols/group_size)
    bool initialized;
} QuantizedTensor;

constexpr float FP8_E4M3_MAX = 448.0f;

const char* ptq_precision_to_string(PTQPrecision precision) {
    switch (precision) {
        case PTQ_PRECISION_INT8: return "int8";
        case PTQ_PRECISION_FP8: return "fp8";
        case PTQ_PRECISION_INT4: return "int4";
        default: return "none";
    }
}

PTQPrecision ptq_precision_from_string(const char* value) {
    if (strcmp(value, "none") == 0) { return PTQ_PRECISION_NONE; }
    if (strcmp(value, "int8") == 0) { return PTQ_PRECISION_INT8; }
    if (strcmp(value, "fp8") == 0) { return PTQ_PRECISION_FP8; }
    if (strcmp(value, "int4") == 0) { return PTQ_PRECISION_INT4; }
    fprintf(stderr, "Unsupported PTQ precision '%s'. Expected one of: int8, fp8, int4.\n", value);
    exit(EXIT_FAILURE);
}

__host__ __device__ inline bool ptq_precision_is_supported(PTQPrecision precision) {
    return precision == PTQ_PRECISION_INT8 || precision == PTQ_PRECISION_FP8 ||
           precision == PTQ_PRECISION_INT4;
}

__host__ __device__ inline float ptq_quant_max(PTQPrecision precision) {
    if (precision == PTQ_PRECISION_INT8) { return 127.0f; }
    if (precision == PTQ_PRECISION_INT4) { return 7.0f; }
    return FP8_E4M3_MAX;
}

__host__ __device__ inline size_t ptq_qvalue_bytes(size_t logical_elements, PTQPrecision precision) {
    return precision == PTQ_PRECISION_INT4 ? (logical_elements + 1) / 2 : logical_elements;
}

// Resolve a user-supplied group size (0 = unset = per-row) against the actual cols
// dimension of a tensor. Returns a value in [1, cols] that divides cols.
inline int ptq_resolve_group_size(int requested_group_size, int cols) {
    if (requested_group_size <= 0 || requested_group_size > cols) { return cols; }
    if (cols % requested_group_size != 0) { return cols; }
    return requested_group_size;
}

__host__ __device__ inline int ptq_num_groups(int cols, int group_size) {
    return cols / group_size;
}

__host__ __device__ inline float ptq_decode_fp8_e4m3(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return (float)value;
}

__global__ void ptq_write_scales_kernel(float* scales, const float* group_maxes,
                                        size_t total_groups, float quant_max);
__host__ __device__ uint8_t ptq_encode_fp8_e4m3(float value);

__host__ __device__ inline size_t aq_scale_idx_2d(int row, int col, int num_group_cols, int group_m, int group_n) {
    return (size_t)(row / group_m) * num_group_cols + (size_t)(col / group_n);
}

__global__ void aq_find_group_max_kernel(float* __restrict__ group_maxes,
                                               const floatX* __restrict__ src,
                                               int rows, int cols, int group_m, int group_n,
                                               int num_group_cols) {
    int gr = blockIdx.y;
    int gc = blockIdx.x;
    int row_begin = gr * group_m;
    int col_begin = gc * group_n;
    float local_max = 0.0f;
    for (int r = threadIdx.x; r < group_m * group_n; r += blockDim.x) {
        int rr = row_begin + (r / group_n);
        int cc = col_begin + (r % group_n);
        if (rr < rows && cc < cols) {
            local_max = fmaxf(local_max, fabsf((float)src[(size_t)rr * cols + cc]));
        }
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        group_maxes[(size_t)gr * num_group_cols + gc] = sdata[0];
    }
}

__global__ void aq_quantize_fp8_apply_kernel(uint8_t* __restrict__ dst,
                                             const floatX* __restrict__ src,
                                             const float* __restrict__ scales,
                                             int rows, int cols,
                                             int group_m, int group_n, int num_group_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * cols;
    if (idx >= count) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float scale = scales[aq_scale_idx_2d(row, col, num_group_cols, group_m, group_n)];
    float v = (float)src[idx] / scale;
    dst[idx] = ptq_encode_fp8_e4m3(v);
}

__global__ void aq_dequantize_fp8_kernel(floatX* __restrict__ dst,
                                         const uint8_t* __restrict__ src,
                                         const float* __restrict__ scales,
                                         int rows, int cols,
                                         int group_m, int group_n, int num_group_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * cols;
    if (idx >= count) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float scale = scales[aq_scale_idx_2d(row, col, num_group_cols, group_m, group_n)];
    float q = ptq_decode_fp8_e4m3(src[idx]);
    dst[idx] = (floatX)(scale * q);
}

void aq_quantize_fp8_rows_gpu(uint8_t* qdst, float* scales, float* group_maxes_scratch,
                              const floatX* src, int rows, int cols, int group_m, int group_n,
                              cudaStream_t stream) {
    const int ngr = CEIL_DIV(rows, group_m);
    const int ngc = CEIL_DIV(cols, group_n);
    const size_t total_groups = (size_t)ngr * ngc;
    dim3 grid(ngc, ngr);
    aq_find_group_max_kernel<<<grid, 256, 0, stream>>>(group_maxes_scratch, src, rows, cols,
                                                        group_m, group_n, ngc);
    ptq_write_scales_kernel<<<CEIL_DIV(total_groups, (size_t)256), 256, 0, stream>>>(
        scales, group_maxes_scratch, total_groups, FP8_E4M3_MAX);
    int count = rows * cols;
    aq_quantize_fp8_apply_kernel<<<CEIL_DIV(count, 256), 256, 0, stream>>>(
        qdst, src, scales, rows, cols, group_m, group_n, ngc);
}

void aq_dequantize_fp8_rows_gpu(floatX* dst, const uint8_t* src, const float* scales,
                                int rows, int cols, int group_m, int group_n,
                                cudaStream_t stream) {
    const int ngc = CEIL_DIV(cols, group_n);
    int count = rows * cols;
    aq_dequantize_fp8_kernel<<<CEIL_DIV(count, 256), 256, 0, stream>>>(
        dst, src, scales, rows, cols, group_m, group_n, ngc);
}

constexpr float INT8_QUANT_MAX = 127.0f;

__global__ void aq_quantize_int8_apply_kernel(uint8_t* __restrict__ dst,
                                              const floatX* __restrict__ src,
                                              const float* __restrict__ scales,
                                              int rows, int cols,
                                              int group_m, int group_n, int num_group_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float scale = scales[aq_scale_idx_2d(row, col, num_group_cols, group_m, group_n)];
    float v = (float)src[idx] / scale;
    int8_t q = (int8_t)fmaxf(-INT8_QUANT_MAX, fminf(INT8_QUANT_MAX, rintf(v)));
    dst[idx] = (uint8_t)q;
}

__global__ void aq_dequantize_int8_kernel(floatX* __restrict__ dst,
                                          const uint8_t* __restrict__ src,
                                          const float* __restrict__ scales,
                                          int rows, int cols,
                                          int group_m, int group_n, int num_group_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float scale = scales[aq_scale_idx_2d(row, col, num_group_cols, group_m, group_n)];
    dst[idx] = (floatX)((int8_t)src[idx] * scale);
}

void aq_quantize_int8_rows_gpu(uint8_t* qdst, float* scales, float* group_maxes_scratch,
                               const floatX* src, int rows, int cols, int group_m, int group_n,
                               cudaStream_t stream) {
    const int ngr = CEIL_DIV(rows, group_m);
    const int ngc = CEIL_DIV(cols, group_n);
    const size_t total_groups = (size_t)ngr * ngc;
    dim3 grid(ngc, ngr);
    aq_find_group_max_kernel<<<grid, 256, 0, stream>>>(group_maxes_scratch, src, rows, cols,
                                                       group_m, group_n, ngc);
    ptq_write_scales_kernel<<<CEIL_DIV(total_groups, (size_t)256), 256, 0, stream>>>(
        scales, group_maxes_scratch, total_groups, INT8_QUANT_MAX);
    int count = rows * cols;
    aq_quantize_int8_apply_kernel<<<CEIL_DIV(count, 256), 256, 0, stream>>>(
        qdst, src, scales, rows, cols, group_m, group_n, ngc);
}

void aq_dequantize_int8_rows_gpu(floatX* dst, const uint8_t* src, const float* scales,
                                 int rows, int cols, int group_m, int group_n,
                                 cudaStream_t stream) {
    const int ngc = CEIL_DIV(cols, group_n);
    int count = rows * cols;
    aq_dequantize_int8_kernel<<<CEIL_DIV(count, 256), 256, 0, stream>>>(
        dst, src, scales, rows, cols, group_m, group_n, ngc);
}

constexpr float INT4_QUANT_MAX = 7.0f;

__host__ __device__ inline int ptq_decode_int4_nibble(uint8_t nibble);
__host__ __device__ inline uint8_t ptq_encode_int4_nibble(int value);
__host__ __device__ inline uint8_t ptq_pack_int4_pair(int low, int high);

// Each thread handles one packed byte (two logical INT4 elements).
// Pairs are taken in flat row-major order; each element independently looks up its 2D group scale.
__global__ void aq_quantize_int4_apply_kernel(uint8_t* __restrict__ dst,
                                              const floatX* __restrict__ src,
                                              const float* __restrict__ scales,
                                              int rows, int cols,
                                              int group_m, int group_n, int num_group_cols) {
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * cols;
    const int idx0 = byte_idx * 2;
    if (idx0 >= count) return;

    const int row0 = idx0 / cols, col0 = idx0 - row0 * cols;
    const float val0 = (float)src[idx0] / scales[aq_scale_idx_2d(row0, col0, num_group_cols, group_m, group_n)];
    const int q0 = (int)lrintf(fmaxf(-INT4_QUANT_MAX, fminf(INT4_QUANT_MAX, val0)));

    int q1 = 0;
    const int idx1 = idx0 + 1;
    if (idx1 < count) {
        const int row1 = idx1 / cols, col1 = idx1 - row1 * cols;
        const float val1 = (float)src[idx1] / scales[aq_scale_idx_2d(row1, col1, num_group_cols, group_m, group_n)];
        q1 = (int)lrintf(fmaxf(-INT4_QUANT_MAX, fminf(INT4_QUANT_MAX, val1)));
    }
    dst[byte_idx] = ptq_pack_int4_pair(q0, q1);
}

__global__ void aq_dequantize_int4_kernel(floatX* __restrict__ dst,
                                          const uint8_t* __restrict__ src,
                                          const float* __restrict__ scales,
                                          int rows, int cols,
                                          int group_m, int group_n, int num_group_cols) {
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * cols;
    const int idx0 = byte_idx * 2;
    if (idx0 >= count) return;

    const uint8_t packed = src[byte_idx];

    const int row0 = idx0 / cols, col0 = idx0 - row0 * cols;
    const float scale0 = scales[aq_scale_idx_2d(row0, col0, num_group_cols, group_m, group_n)];
    dst[idx0] = (floatX)(ptq_decode_int4_nibble(packed & 0x0F) * scale0);

    const int idx1 = idx0 + 1;
    if (idx1 < count) {
        const int row1 = idx1 / cols, col1 = idx1 - row1 * cols;
        const float scale1 = scales[aq_scale_idx_2d(row1, col1, num_group_cols, group_m, group_n)];
        dst[idx1] = (floatX)(ptq_decode_int4_nibble(packed >> 4) * scale1);
    }
}

void aq_quantize_int4_rows_gpu(uint8_t* qdst, float* scales, float* group_maxes_scratch,
                               const floatX* src, int rows, int cols, int group_m, int group_n,
                               cudaStream_t stream) {
    const int ngr = CEIL_DIV(rows, group_m);
    const int ngc = CEIL_DIV(cols, group_n);
    const size_t total_groups = (size_t)ngr * ngc;
    dim3 grid(ngc, ngr);
    aq_find_group_max_kernel<<<grid, 256, 0, stream>>>(group_maxes_scratch, src, rows, cols,
                                                       group_m, group_n, ngc);
    ptq_write_scales_kernel<<<CEIL_DIV(total_groups, (size_t)256), 256, 0, stream>>>(
        scales, group_maxes_scratch, total_groups, INT4_QUANT_MAX);
    const int qbytes = ((size_t)rows * cols + 1) / 2;
    aq_quantize_int4_apply_kernel<<<CEIL_DIV(qbytes, 256), 256, 0, stream>>>(
        qdst, src, scales, rows, cols, group_m, group_n, ngc);
}

void aq_dequantize_int4_rows_gpu(floatX* dst, const uint8_t* src, const float* scales,
                                 int rows, int cols, int group_m, int group_n,
                                 cudaStream_t stream) {
    const int ngc = CEIL_DIV(cols, group_n);
    const int qbytes = ((size_t)rows * cols + 1) / 2;
    aq_dequantize_int4_kernel<<<CEIL_DIV(qbytes, 256), 256, 0, stream>>>(
        dst, src, scales, rows, cols, group_m, group_n, ngc);
}

__host__ __device__ inline int ptq_decode_int4_nibble(uint8_t nibble) {
    constexpr int INT4_BITS = 4;
    constexpr uint8_t INT4_MASK = (1u << INT4_BITS) - 1u;
    return (int)((int8_t)((nibble & INT4_MASK) << INT4_BITS) >> INT4_BITS);
}

__host__ __device__ inline uint8_t ptq_encode_int4_nibble(int value) {
    return (uint8_t)(value & 0x0F);
}

__host__ __device__ inline uint8_t ptq_pack_int4_pair(int low, int high) {
    return ptq_encode_int4_nibble(low) | (ptq_encode_int4_nibble(high) << 4);
}

__host__ __device__ inline float ptq_unpack_decode_value(const uint8_t* qvalues, size_t logical_idx,
                                                         PTQPrecision precision) {
    if (precision == PTQ_PRECISION_INT8) {
        const uint8_t raw = qvalues[logical_idx];
        return (float)((int8_t)raw);
    }
    if (precision == PTQ_PRECISION_FP8) {
        const uint8_t raw = qvalues[logical_idx];
        return ptq_decode_fp8_e4m3(raw);
    }
    if (precision == PTQ_PRECISION_INT4) {
        const uint8_t packed = qvalues[logical_idx >> 1];
        const uint8_t nibble = (logical_idx & 1) ? (packed >> 4) : (packed & 0x0F);
        return (float)ptq_decode_int4_nibble(nibble);
    }
    return 0.0f;
}

__host__ __device__ uint8_t ptq_encode_fp8_e4m3(float value) {
    return __nv_fp8_e4m3(value).__x;
}

// Host-side reference quantizer. group_size in [1, cols] and must divide cols.
// scales has shape [rows, num_groups] with num_groups = cols/group_size.
// Pass group_size == cols for per-row behavior (degenerate single-group case).
void ptq_quantize_rows_host(uint8_t* dst, float* scales, const float* src, int rows, int cols,
                            int group_size, PTQPrecision precision) {
    assert(ptq_precision_is_supported(precision));
    assert(group_size >= 1 && group_size <= cols && cols % group_size == 0);
    const float quant_max = ptq_quant_max(precision);
    const int num_groups = ptq_num_groups(cols, group_size);
    for (int row = 0; row < rows; ++row) {
        const float* row_src = src + row * cols;
        for (int g = 0; g < num_groups; ++g) {
            const int col_begin = g * group_size;
            float max_abs = 0.0f;
            for (int k = 0; k < group_size; ++k) {
                max_abs = fmaxf(max_abs, fabsf(row_src[col_begin + k]));
            }
            const float scale = max_abs > 0.0f ? max_abs / quant_max : 1.0f;
            scales[(size_t)row * num_groups + g] = scale;
        }
    }

    if (precision == PTQ_PRECISION_INT4) {
        memset(dst, 0, ptq_qvalue_bytes((size_t)rows * cols, precision));
    }

    for (int row = 0; row < rows; ++row) {
        const float* row_src = src + row * cols;
        for (int col = 0; col < cols; ++col) {
            const int g = col / group_size;
            const float scale = scales[(size_t)row * num_groups + g];
            const float scaled = row_src[col] / scale;
            if (precision == PTQ_PRECISION_INT8) {
                const int q = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, scaled)));
                dst[(size_t)row * cols + col] = (uint8_t)((int8_t)q);
            } else if (precision == PTQ_PRECISION_FP8) {
                dst[(size_t)row * cols + col] = ptq_encode_fp8_e4m3(scaled);
            } else {
                const size_t idx = (size_t)row * cols + col;
                const int q = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, scaled)));
                uint8_t* packed = dst + (idx >> 1);
                if (idx & 1) {
                    *packed = (*packed & 0x0F) | (ptq_encode_int4_nibble(q) << 4);
                } else {
                    *packed = (*packed & 0xF0) | ptq_encode_int4_nibble(q);
                }
            }
        }
    }
}

void ptq_dequantize_rows_host(float* dst, const uint8_t* src, const float* scales, int rows, int cols,
                              int group_size, PTQPrecision precision) {
    assert(ptq_precision_is_supported(precision));
    assert(group_size >= 1 && group_size <= cols && cols % group_size == 0);
    const int num_groups = ptq_num_groups(cols, group_size);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int g = col / group_size;
            const float scale = scales[(size_t)row * num_groups + g];
            const size_t idx = (size_t)row * cols + col;
            const float q = ptq_unpack_decode_value(src, idx, precision);
            dst[idx] = scale * q;
        }
    }
}

// Dequantize with group-wise scales. scales has shape [rows, num_groups] where
// num_groups = cols / group_size. Pass group_size == cols for the per-row case.
__global__ void ptq_dequantize_rows_kernel(floatX* dst, const uint8_t* src, const float* scales,
                                           int rows, int cols, int group_size, PTQPrecision precision) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * cols;
    if (idx >= count) { return; }
    const int row = idx / cols;
    const int col = idx - row * cols;
    const int num_groups = cols / group_size;
    const int g = col / group_size;
    const float q = ptq_unpack_decode_value(src, (size_t)idx, precision);
    dst[idx] = (floatX)(scales[(size_t)row * num_groups + g] * q);
}

void ptq_dequantize_rows(floatX* dst, const uint8_t* src, const float* scales, int rows, int cols,
                         int group_size, PTQPrecision precision, cudaStream_t stream) {
    const int count = rows * cols;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(count, block_size);
    ptq_dequantize_rows_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, scales, rows, cols,
                                                                     group_size, precision);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPU-side group-wise quantization (beast mode: no host round-trip).
// "Per-row" is the degenerate case group_size == cols, num_groups == 1, and
// every kernel below collapses to identical work to the original row-wise path.

// Pass 1a: find max |value| per (row, group), floatX source.
// One block per group: grid = (num_groups, rows). 256 threads stride over group_size
// values and reduce in shared memory.
__global__ void ptq_find_group_max_kernel(float* __restrict__ group_maxes,
                                          const floatX* __restrict__ src,
                                          int rows, int cols, int group_size) {
    const int g = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int num_groups = cols / group_size;
    const int col_begin = g * group_size;
    const floatX* row_src = src + (size_t)row * cols;
    float local_max = 0.0f;
    for (int k = threadIdx.x; k < group_size; k += blockDim.x) {
        local_max = fmaxf(local_max, fabsf((float)row_src[col_begin + k]));
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) group_maxes[(size_t)row * num_groups + g] = sdata[0];
}

// Pass 1b: same but from FP32 source (used when re-quantizing from master_weights).
__global__ void ptq_find_group_max_fp32_kernel(float* __restrict__ group_maxes,
                                                const float* __restrict__ src,
                                                int rows, int cols, int group_size) {
    const int g = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int num_groups = cols / group_size;
    const int col_begin = g * group_size;
    const float* row_src = src + (size_t)row * cols;
    float local_max = 0.0f;
    for (int k = threadIdx.x; k < group_size; k += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_src[col_begin + k]));
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) group_maxes[(size_t)row * num_groups + g] = sdata[0];
}

// Pass 1c: convert group_maxes -> scales (one element per group).
__global__ void ptq_write_scales_kernel(float* scales, const float* group_maxes,
                                        size_t total_groups, float quant_max) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;
    float m = group_maxes[idx];
    scales[idx] = (m > 0.0f) ? (m / quant_max) : 1.0f;
}

// Index helper: scale index for a given (row, col) under group_size.
// scales has layout [rows, num_groups] with num_groups = cols / group_size.
__host__ __device__ inline size_t ptq_scale_idx(int row, int col, int cols, int group_size) {
    const int num_groups = cols / group_size;
    return (size_t)row * num_groups + (col / group_size);
}

// Pass 2a: quantize from floatX using precomputed group-wise scales.
__global__ void ptq_quantize_apply_kernel(uint8_t* __restrict__ dst,
                                          const floatX* __restrict__ src,
                                          const float* __restrict__ scales,
                                          int rows, int cols, int group_size,
                                          PTQPrecision precision) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float val = (float)src[idx] / scales[ptq_scale_idx(row, col, cols, group_size)];
    if (precision == PTQ_PRECISION_INT8) {
        int q = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, val)));
        dst[idx] = (uint8_t)((int8_t)q);
    } else {
        dst[idx] = ptq_encode_fp8_e4m3(val);
    }
}

__global__ void ptq_quantize_apply_int4_kernel(uint8_t* __restrict__ dst,
                                               const floatX* __restrict__ src,
                                               const float* __restrict__ scales,
                                               int rows, int cols, int group_size) {
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * cols;
    const int idx0 = byte_idx * 2;
    if (idx0 >= count) return;

    const int row0 = idx0 / cols;
    const int col0 = idx0 - row0 * cols;
    const float val0 = (float)src[idx0] / scales[ptq_scale_idx(row0, col0, cols, group_size)];
    const int q0 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, val0)));

    int q1 = 0;
    const int idx1 = idx0 + 1;
    if (idx1 < count) {
        const int row1 = idx1 / cols;
        const int col1 = idx1 - row1 * cols;
        const float val1 = (float)src[idx1] / scales[ptq_scale_idx(row1, col1, cols, group_size)];
        q1 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, val1)));
    }
    dst[byte_idx] = ptq_pack_int4_pair(q0, q1);
}

// Pass 2b: quantize from FP32 using precomputed group-wise scales.
__global__ void ptq_quantize_apply_fp32_kernel(uint8_t* __restrict__ dst,
                                               const float* __restrict__ src,
                                               const float* __restrict__ scales,
                                               int rows, int cols, int group_size,
                                               PTQPrecision precision) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    int col = idx - row * cols;
    float val = src[idx] / scales[ptq_scale_idx(row, col, cols, group_size)];
    if (precision == PTQ_PRECISION_INT8) {
        int q = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, val)));
        dst[idx] = (uint8_t)((int8_t)q);
    } else {
        dst[idx] = ptq_encode_fp8_e4m3(val);
    }
}

__global__ void ptq_quantize_apply_int4_fp32_kernel(uint8_t* __restrict__ dst,
                                                    const float* __restrict__ src,
                                                    const float* __restrict__ scales,
                                                    int rows, int cols, int group_size) {
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * cols;
    const int idx0 = byte_idx * 2;
    if (idx0 >= count) return;

    const int row0 = idx0 / cols;
    const int col0 = idx0 - row0 * cols;
    const float val0 = src[idx0] / scales[ptq_scale_idx(row0, col0, cols, group_size)];
    const int q0 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, val0)));

    int q1 = 0;
    const int idx1 = idx0 + 1;
    if (idx1 < count) {
        const int row1 = idx1 / cols;
        const int col1 = idx1 - row1 * cols;
        const float val1 = src[idx1] / scales[ptq_scale_idx(row1, col1, cols, group_size)];
        q1 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, val1)));
    }
    dst[byte_idx] = ptq_pack_int4_pair(q0, q1);
}

// GPU quantize from floatX. group_maxes_scratch: caller-owned device buffer with capacity for
// `rows * (cols/group_size)` floats. group_size must divide cols and be in [1, cols].
void ptq_quantize_rows_gpu(uint8_t* dst, float* scales, float* group_maxes_scratch,
                            const floatX* src, int rows, int cols, int group_size,
                            PTQPrecision precision, cudaStream_t stream) {
    assert(group_size >= 1 && group_size <= cols && cols % group_size == 0);
    const float quant_max = ptq_quant_max(precision);
    const int num_groups = ptq_num_groups(cols, group_size);
    const size_t total_groups = (size_t)rows * num_groups;
    dim3 max_grid(num_groups, rows);
    ptq_find_group_max_kernel<<<max_grid, 256, 0, stream>>>(group_maxes_scratch, src,
                                                            rows, cols, group_size);
    cudaCheck(cudaGetLastError());
    ptq_write_scales_kernel<<<CEIL_DIV(total_groups, (size_t)256), 256, 0, stream>>>(
        scales, group_maxes_scratch, total_groups, quant_max);
    cudaCheck(cudaGetLastError());
    if (precision == PTQ_PRECISION_INT4) {
        const int qbytes = (int)ptq_qvalue_bytes((size_t)rows * cols, precision);
        ptq_quantize_apply_int4_kernel<<<CEIL_DIV(qbytes, 256), 256, 0, stream>>>(
            dst, src, scales, rows, cols, group_size);
    } else {
        ptq_quantize_apply_kernel<<<CEIL_DIV(rows * cols, 256), 256, 0, stream>>>(
            dst, src, scales, rows, cols, group_size, precision);
    }
    cudaCheck(cudaGetLastError());
}

// GPU quantize from FP32 master weights. Same group_maxes_scratch sizing as above.
void ptq_quantize_rows_gpu_fp32(uint8_t* dst, float* scales, float* group_maxes_scratch,
                                const float* src, int rows, int cols, int group_size,
                                PTQPrecision precision, cudaStream_t stream) {
    assert(group_size >= 1 && group_size <= cols && cols % group_size == 0);
    const float quant_max = ptq_quant_max(precision);
    const int num_groups = ptq_num_groups(cols, group_size);
    const size_t total_groups = (size_t)rows * num_groups;
    dim3 max_grid(num_groups, rows);
    ptq_find_group_max_fp32_kernel<<<max_grid, 256, 0, stream>>>(group_maxes_scratch, src,
                                                                  rows, cols, group_size);
    cudaCheck(cudaGetLastError());
    ptq_write_scales_kernel<<<CEIL_DIV(total_groups, (size_t)256), 256, 0, stream>>>(
        scales, group_maxes_scratch, total_groups, quant_max);
    cudaCheck(cudaGetLastError());
    if (precision == PTQ_PRECISION_INT4) {
        const int qbytes = (int)ptq_qvalue_bytes((size_t)rows * cols, precision);
        ptq_quantize_apply_int4_fp32_kernel<<<CEIL_DIV(qbytes, 256), 256, 0, stream>>>(
            dst, src, scales, rows, cols, group_size);
    } else {
        ptq_quantize_apply_fp32_kernel<<<CEIL_DIV(rows * cols, 256), 256, 0, stream>>>(
            dst, src, scales, rows, cols, group_size, precision);
    }
    cudaCheck(cudaGetLastError());
}

// Dequantize one layer-slice from a QuantizedTensor into dst (device floatX).
// For non-layered tensors pass layer=0; rows_per_layer should equal total rows.
void ptq_dequantize_layer_slice(floatX* dst, const QuantizedTensor* qt,
                                int layer, PTQPrecision precision, cudaStream_t stream) {
    const int rows = qt->rows_per_layer;
    const int cols = qt->cols;
    const int group_size = qt->group_size;
    const int num_groups = ptq_num_groups(cols, group_size);
    assert(precision != PTQ_PRECISION_INT4 || (((size_t)rows * cols) % 2 == 0));
    const size_t elem_offset  = (size_t)layer * rows * cols;
    const size_t scale_offset = (size_t)layer * rows * num_groups;
    ptq_dequantize_rows(dst,
                        qt->qvalues + ptq_qvalue_bytes(elem_offset, precision),
                        qt->scales  + scale_offset,
                        rows, cols, group_size, precision, stream);
}

typedef struct {
    // All learnable weights are stored in floatX on device:
    // - FP32 build: float
    // - BF16 build: __nv_bfloat16
    // - FP16 build: half (not fully wired up for checkpoint loading yet)
    // Precision-sensitive math often still accumulates in FP32 later.
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

typedef struct {
    QuantizedTensor tensors[NUM_PARAMETER_TENSORS];
    size_t original_weight_bytes;
    size_t quantized_weight_bytes;
    int num_quantized_tensors;
    bool initialized;
} QuantizedParameters;

enum AQType {
    AQ_TYPE_NONE = 0,
    AQ_TYPE_FP8  = 1,
    AQ_TYPE_INT8 = 2,
    AQ_TYPE_INT4 = 3,
};

typedef struct {
    uint8_t* qvalues; // one FP8 value per element
    float* scales;    // one scale per matrix group
    int rows;
    int cols;
    int group_m;
    int group_n;
    int num_group_rows;
    int num_group_cols;
    size_t qvalue_count;
    size_t scale_count;
    bool initialized;
} QuantizedActivationTensor;

constexpr int NUM_AQ_TENSORS = 7;
typedef struct {
    QuantizedActivationTensor tensors[NUM_AQ_TENSORS];
    bool initialized;
} QuantizedActivations;

typedef struct {
    int num_layers;
    int rows_per_layer;
    int cols;
} PTQTensorLayout;

enum AQTensorId {
    AQ_TENSOR_LN1 = 0,
    AQ_TENSOR_ATTY = 1,
    AQ_TENSOR_RESIDUAL2 = 2,
    AQ_TENSOR_LN2 = 3,
    AQ_TENSOR_FCH = 4,
    AQ_TENSOR_FCH_GELU = 5,
    AQ_TENSOR_QKVR = 6,
};

typedef struct GPT2 GPT2;

void get_parameter_tensor_ptrs(ParameterTensors* params, floatX** ptrs[NUM_PARAMETER_TENSORS]) {
    ptrs[0] = &params->wte;
    ptrs[1] = &params->wpe;
    ptrs[2] = &params->ln1w;
    ptrs[3] = &params->ln1b;
    ptrs[4] = &params->qkvw;
    ptrs[5] = &params->qkvb;
    ptrs[6] = &params->attprojw;
    ptrs[7] = &params->attprojb;
    ptrs[8] = &params->ln2w;
    ptrs[9] = &params->ln2b;
    ptrs[10] = &params->fcw;
    ptrs[11] = &params->fcb;
    ptrs[12] = &params->fcprojw;
    ptrs[13] = &params->fcprojb;
    ptrs[14] = &params->lnfw;
    ptrs[15] = &params->lnfb;
}

const char* parameter_tensor_name(int tensor_id) {
    static const char* names[NUM_PARAMETER_TENSORS] = {
        "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
        "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"
    };
    if (tensor_id < 0 || tensor_id >= NUM_PARAMETER_TENSORS) {
        return "unknown";
    }
    return names[tensor_id];
}

bool ptq_should_quantize_tensor(int tensor_id) {
    // Beast mode: only quantize the large transformer weight matrices.
    // wte and wpe stay as floatX: encoder_forward needs random-access gather,
    // and wte is also the weight-tied classifier — both are accessed lookup-style.
    // Biases and LayerNorm weights are tiny; quantizing them adds error for no gain.
    switch (tensor_id) {
        case 4:  // qkvw   (L, 3C, C)
        case 6:  // attprojw (L, C, C)
        case 10: // fcw    (L, 4C, C)
        case 12: // fcprojw (L, C, 4C)
            return true;
        default:
            return false;
    }
}

PTQTensorLayout ptq_tensor_layout_for_index(GPT2Config config, int tensor_id) {
    const int L = config.num_layers;
    const int C = config.channels;
    switch (tensor_id) {
        case 0: return {1, config.padded_vocab_size, C};
        case 1: return {1, config.max_seq_len, C};
        case 2: return {L, 1, C};
        case 3: return {L, 1, C};
        case 4: return {L, 3 * C, C};
        case 5: return {L, 1, 3 * C};
        case 6: return {L, C, C};
        case 7: return {L, 1, C};
        case 8: return {L, 1, C};
        case 9: return {L, 1, C};
        case 10: return {L, 4 * C, C};
        case 11: return {L, 1, 4 * C};
        case 12: return {L, C, 4 * C};
        case 13: return {L, 1, C};
        case 14: return {1, 1, C};
        case 15: return {1, 1, C};
        default:
            fprintf(stderr, "Invalid PTQ tensor id %d\n", tensor_id);
            exit(EXIT_FAILURE);
    }
}

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // calculate the total number of parameters and bytes across all tensors
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

constexpr int NUM_ACTIVATION_TENSORS = 21;
typedef struct {
    // Activations that participate in the main model dataflow are stored in floatX.
    // Statistics that need extra numeric stability (layernorm mean/rstd, losses, some
    // cuDNN attention metadata) stay in FP32 even in BF16 mode.
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    // cuDNN saves only some statistics information
#if ENABLE_CUDNN
    float* att;  // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif

    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;

    // some additional scratch buffers
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
} ActivationTensors;


struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};


#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute, bool aq_enabled) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T);
    #else
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    #endif
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2 && !aq_enabled) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? (aq_enabled ? 0 : L * B * T * 4*C) : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C);
    tensors[18] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));

    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t aq_extra_bytes = 0) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }

    const size_t total_bytes = bytes + aq_extra_bytes;
    if (aq_extra_bytes > 0) {
        printf0("allocating %d MiB for activations (+%d MiB AQ, total %d MiB)\n",
                (int)round(bytes / (1024 * 1024)),
                (int)round(aq_extra_bytes / (1024 * 1024)),
                (int)round(total_bytes / (1024 * 1024)));
    } else {
        printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));
    }

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));

    // cudaMalloc does not guarantee initial memory values so we memset the allocation here
    // this matters because e.g. non-cuDNN attention assumes the attention buffer is zeroed
    // todo - up to ~100ms on slow GPUs, could theoretically be more selective, but this is safer
    cudaCheck(cudaMemset(acts_memory, 0, bytes));

    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

typedef struct {
    int rows;
    int cols;
} AQTensorLayout;

size_t aq_estimate_activation_bytes(const GPT2* model);
AQTensorLayout aq_tensor_layout_for_id(const GPT2* model, int aq_id);
floatX* aq_tensor_ptr_for_id(const GPT2* model, int aq_id);
void gpt2_prepare_aq(GPT2* model);
void aq_quantize_tensor_slice(GPT2* model, int aq_id, int layer_or_zero, const floatX* src, cudaStream_t stream);
void aq_dequantize_tensor_slice(GPT2* model, int aq_id, int layer_or_zero, floatX* dst, cudaStream_t stream);

struct GPT2 {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // AdamW optimizer state.
    // optim_quant=0 (fp32): FP32 first (m) and second (v) moments — standard mixed precision.
    // optim_quant=1/2/3 (fp8/int8/int4): m_memory/v_memory are always nullptr; quantized layout is active.
    // optim_quant: 0=fp32, 1=fp8(COAT), 2=int8, 3=int4
    int optim_quant;
    int optim_group_size; // group size for quantized moments (all modes)
    int coat_expansion;   // 1=use COAT k-factor dynamic range expansion, 0=plain absmax
    float* m_memory;      // FP32 moments (optim_quant==0 only)
    float* v_memory;
    // Quantized moment storage (optim_quant 1/2/3). Reused across formats:
    //   fp8/int8: 1 byte/param   int4: 0.5 bytes/param (nibble-packed)
    uint8_t* m_qstate;
    uint8_t* v_qstate;
    float*   m_scales;    // absmax scale, one float per group
    float*   v_scales;
    float*   m_kfactors;  // COAT expansion exponent k, one float per group (all quantized modes)
    float*   v_kfactors;
    float* master_weights;     // optional FP32 copy of params used for numerically safer updates
    // the activations of the model, and their sizes
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // RNG state used by stochastic rounding and other kernels
    unsigned long long rng_state_last_update; // saved so checkpoint restore can reproduce the same low-precision rounding
    int use_master_weights; // keep a FP32 master copy for the optimizer/update path? 0|1
    bool init_state;   // set to true if master weights need to be initialized
    int gelu_fusion; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute; // recompute gelu | layernorm forward during model backward? 0|1|2
    int aq_enabled;
    AQType aq_type;
    int aq_group_size;
    QuantizedActivations aq;
    floatX* aq_scratch;
    size_t aq_scratch_elems;
    float* aq_group_maxes_scratch;
    size_t aq_group_maxes_scratch_elems;
    // Beast-mode PTQ: quantized storage for large transformer weight matrices.
    // params.qkvw / attprojw / fcw / fcprojw are NULL at runtime;
    // they live in ptq.tensors[i] as (qvalues + scales) and are dequantized
    // on-demand per-layer into scratch_dequant during forward/backward.
    int ptq_enabled;
    PTQPrecision ptq_precision;
    // Requested group size along the cols axis for grouped quantization (int4 in particular).
    // 0 = "unset" => per-row. Per-tensor effective group size is resolved against cols and
    // stored in QuantizedTensor::group_size.
    int ptq_group_size;
    QuantizedParameters ptq;
    // Per-layer dequant scratch: large enough for the biggest single-layer quantized weight
    // (fcw/fcprojw = 4*C*C floatX). Reused every layer, every step.
    floatX* scratch_dequant;
    size_t  scratch_dequant_elems; // capacity in elements
    // Row-max scratch for GPU quantization (used in prepare and after each update).
    // Sized for the maximum number of rows across all quantized tensors.
    float*  row_maxes_scratch;
    size_t  row_maxes_scratch_elems;
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
};

size_t aq_estimate_activation_bytes(const GPT2* model) {
    if (!model->aq_enabled || model->aq_type != AQ_TYPE_FP8 || model->aq_group_size <= 0) {
        return 0;
    }
    size_t qvalue_bytes = 0;
    size_t scale_bytes = 0;
    size_t max_elems = 0;
    size_t max_groups = 0;
    for (int i = 0; i < NUM_AQ_TENSORS; ++i) {
        AQTensorLayout lo = aq_tensor_layout_for_id(model, i);
        if (lo.cols % model->aq_group_size != 0) {
            return 0;
        }
        const size_t elems = (size_t)lo.rows * lo.cols;
        // COAT-style 1×group_n row quantization: one group per row-slice of group_n cols
        const size_t groups = (size_t)lo.rows * (size_t)(lo.cols / model->aq_group_size);
        qvalue_bytes += elems * sizeof(uint8_t);
        scale_bytes += groups * sizeof(float);
        if (elems > max_elems) max_elems = elems;
        if (groups > max_groups) max_groups = groups;
    }
    const size_t scratch_bytes = max_elems * sizeof(floatX);
    const size_t group_scratch_bytes = max_groups * sizeof(float);
    return qvalue_bytes + scale_bytes + scratch_bytes + group_scratch_bytes;
}

AQTensorLayout aq_tensor_layout_for_id(const GPT2* model, int aq_id) {
    const int L = model->config.num_layers;
    const int B = model->batch_size;
    const int T = model->seq_len;
    const int C = model->config.channels;
    switch (aq_id) {
        case AQ_TENSOR_LN1:       return {(model->recompute < 2) ? L * B * T : B * T, C};
        case AQ_TENSOR_ATTY:      return {L * B * T, C};
        case AQ_TENSOR_RESIDUAL2: return {L * B * T, C};
        case AQ_TENSOR_LN2:       return {(model->recompute < 2) ? L * B * T : B * T, C};
        case AQ_TENSOR_FCH:       return {L * B * T, 4 * C};
        case AQ_TENSOR_FCH_GELU:  return {(model->recompute < 1) ? L * B * T : B * T, 4 * C};
        case AQ_TENSOR_QKVR:      return {L * B * T, 3 * C};
        default:
            fprintf(stderr, "Invalid AQ tensor id %d\n", aq_id);
            exit(EXIT_FAILURE);
    }
}

floatX* aq_tensor_ptr_for_id(const GPT2* model, int aq_id) {
    switch (aq_id) {
        case AQ_TENSOR_LN1: return model->acts.ln1;
        case AQ_TENSOR_ATTY: return model->acts.atty;
        case AQ_TENSOR_RESIDUAL2: return model->acts.residual2;
        case AQ_TENSOR_LN2: return model->acts.ln2;
        case AQ_TENSOR_FCH: return model->acts.fch;
        case AQ_TENSOR_FCH_GELU: return model->acts.fch_gelu;
        case AQ_TENSOR_QKVR: return model->acts.qkvr;
        default: return nullptr;
    }
}

void gpt2_prepare_aq(GPT2* model) {
    if (!model->aq_enabled) {
        for (int i = 0; i < NUM_AQ_TENSORS; ++i) {
            cudaFreeCheck(&model->aq.tensors[i].qvalues);
            cudaFreeCheck(&model->aq.tensors[i].scales);
            model->aq.tensors[i] = {};
        }
        model->aq.initialized = false;
        return;
    }
    if (model->aq_type != AQ_TYPE_FP8 && model->aq_type != AQ_TYPE_INT8 && model->aq_type != AQ_TYPE_INT4) {
        fprintf(stderr, "AQ type not supported. Expected fp8, int8, or int4.\n");
        exit(EXIT_FAILURE);
    }
    if (model->aq_group_size <= 0) {
        fprintf(stderr, "AQ group size must be positive.\n");
        exit(EXIT_FAILURE);
    }
    size_t max_elems = 0;
    size_t max_groups = 0;
    for (int i = 0; i < NUM_AQ_TENSORS; ++i) {
        AQTensorLayout lo = aq_tensor_layout_for_id(model, i);
        if (lo.cols % model->aq_group_size != 0) {
            fprintf(stderr, "AQ group_n %d must divide tensor %d cols (%d)\n",
                    model->aq_group_size, i, lo.cols);
            exit(EXIT_FAILURE);
        }
        size_t elems = (size_t)lo.rows * lo.cols;
        if (model->aq_type == AQ_TYPE_INT4 && elems % 2 != 0) {
            fprintf(stderr, "AQ INT4: tensor %d has odd element count (%zu) — cannot nibble-pack\n", i, elems);
            exit(EXIT_FAILURE);
        }
        // COAT-style 1×group_n: one group per row-slice of group_n cols
        int ngr = lo.rows;
        int ngc = lo.cols / model->aq_group_size;
        size_t groups = (size_t)ngr * ngc;
        if (elems > max_elems) max_elems = elems;
        if (groups > max_groups) max_groups = groups;

        QuantizedActivationTensor* qt = &model->aq.tensors[i];
        if (qt->initialized && (qt->qvalue_count != elems || qt->scale_count != groups)) {
            cudaFreeCheck(&qt->qvalues);
            cudaFreeCheck(&qt->scales);
            *qt = {};
        }
        if (!qt->initialized) {
            // INT4 packs two logical values per byte
            size_t qbytes = (model->aq_type == AQ_TYPE_INT4) ? (elems + 1) / 2 : elems;
            cudaCheck(cudaMalloc((void**)&qt->qvalues, qbytes * sizeof(uint8_t)));
            cudaCheck(cudaMalloc((void**)&qt->scales, groups * sizeof(float)));
            qt->rows = lo.rows;
            qt->cols = lo.cols;
            qt->group_m = 1;                    // COAT-style: one row per group
            qt->group_n = model->aq_group_size;
            qt->num_group_rows = ngr;
            qt->num_group_cols = ngc;
            qt->qvalue_count = elems;
            qt->scale_count = groups;
            qt->initialized = true;
        }
    }
    if (model->aq_scratch == nullptr || model->aq_scratch_elems < max_elems) {
        cudaFreeCheck(&model->aq_scratch);
        cudaCheck(cudaMalloc((void**)&model->aq_scratch, max_elems * sizeof(floatX)));
        model->aq_scratch_elems = max_elems;
    }
    if (model->aq_group_maxes_scratch == nullptr || model->aq_group_maxes_scratch_elems < max_groups) {
        cudaFreeCheck(&model->aq_group_maxes_scratch);
        cudaCheck(cudaMalloc((void**)&model->aq_group_maxes_scratch, max_groups * sizeof(float)));
        model->aq_group_maxes_scratch_elems = max_groups;
    }
    model->aq.initialized = true;

    size_t aq_qvalue_bytes = 0;
    size_t aq_scale_bytes = 0;
    for (int i = 0; i < NUM_AQ_TENSORS; ++i) {
        const QuantizedActivationTensor* qt = &model->aq.tensors[i];
        size_t qbytes = (model->aq_type == AQ_TYPE_INT4) ? (qt->qvalue_count + 1) / 2 : qt->qvalue_count;
        aq_qvalue_bytes += qbytes * sizeof(uint8_t);
        aq_scale_bytes += qt->scale_count * sizeof(float);
    }
    const size_t aq_scratch_bytes = model->aq_scratch_elems * sizeof(floatX);
    const size_t aq_group_scratch_bytes = model->aq_group_maxes_scratch_elems * sizeof(float);
    const size_t aq_total_bytes = aq_qvalue_bytes + aq_scale_bytes + aq_scratch_bytes + aq_group_scratch_bytes;
    printf0("AQ memory: qvalues=%zu MiB, scales=%zu MiB, scratch=%zu MiB, group_scratch=%zu MiB, total=%zu MiB\n",
            aq_qvalue_bytes >> 20, aq_scale_bytes >> 20, aq_scratch_bytes >> 20,
            aq_group_scratch_bytes >> 20, aq_total_bytes >> 20);
}

void aq_quantize_tensor_slice(GPT2* model, int aq_id, int layer_or_zero, const floatX* src, cudaStream_t stream) {
    if (!model->aq_enabled || !model->aq.initialized) return;
    QuantizedActivationTensor* qt = &model->aq.tensors[aq_id];
    int layer_rows = (aq_id == AQ_TENSOR_FCH_GELU && model->recompute >= 1) ? (model->batch_size * model->seq_len)
                                                                              : (model->batch_size * model->seq_len);
    if (aq_id == AQ_TENSOR_LN1 || aq_id == AQ_TENSOR_LN2 || aq_id == AQ_TENSOR_FCH_GELU) {
        if ((aq_id == AQ_TENSOR_LN1 || aq_id == AQ_TENSOR_LN2) && model->recompute >= 2) layer_rows = model->batch_size * model->seq_len;
        if (aq_id == AQ_TENSOR_FCH_GELU && model->recompute >= 1) layer_rows = model->batch_size * model->seq_len;
    }
    size_t elem_offset = (size_t)layer_or_zero * layer_rows * qt->cols;
    size_t scale_offset = (size_t)layer_or_zero * (layer_rows / qt->group_m) * qt->num_group_cols;
    // INT4 packs two logical elements per byte, so the byte offset is half the element offset
    size_t byte_offset = (model->aq_type == AQ_TYPE_INT4) ? elem_offset / 2 : elem_offset;
    if (model->aq_type == AQ_TYPE_INT4) {
        aq_quantize_int4_rows_gpu(qt->qvalues + byte_offset, qt->scales + scale_offset,
                                  model->aq_group_maxes_scratch, src, layer_rows, qt->cols,
                                  qt->group_m, qt->group_n, stream);
    } else if (model->aq_type == AQ_TYPE_INT8) {
        aq_quantize_int8_rows_gpu(qt->qvalues + byte_offset, qt->scales + scale_offset,
                                  model->aq_group_maxes_scratch, src, layer_rows, qt->cols,
                                  qt->group_m, qt->group_n, stream);
    } else {
        aq_quantize_fp8_rows_gpu(qt->qvalues + byte_offset, qt->scales + scale_offset,
                                 model->aq_group_maxes_scratch, src, layer_rows, qt->cols,
                                 qt->group_m, qt->group_n, stream);
    }
}

void aq_dequantize_tensor_slice(GPT2* model, int aq_id, int layer_or_zero, floatX* dst, cudaStream_t stream) {
    if (!model->aq_enabled || !model->aq.initialized) return;
    QuantizedActivationTensor* qt = &model->aq.tensors[aq_id];
    int layer_rows = model->batch_size * model->seq_len;
    size_t elem_offset = (size_t)layer_or_zero * layer_rows * qt->cols;
    size_t scale_offset = (size_t)layer_or_zero * (layer_rows / qt->group_m) * qt->num_group_cols;
    size_t byte_offset = (model->aq_type == AQ_TYPE_INT4) ? elem_offset / 2 : elem_offset;
    if (model->aq_type == AQ_TYPE_INT4) {
        aq_dequantize_int4_rows_gpu(dst, qt->qvalues + byte_offset, qt->scales + scale_offset,
                                    layer_rows, qt->cols, qt->group_m, qt->group_n, stream);
    } else if (model->aq_type == AQ_TYPE_INT8) {
        aq_dequantize_int8_rows_gpu(dst, qt->qvalues + byte_offset, qt->scales + scale_offset,
                                    layer_rows, qt->cols, qt->group_m, qt->group_n, stream);
    } else {
        aq_dequantize_fp8_rows_gpu(dst, qt->qvalues + byte_offset, qt->scales + scale_offset,
                                   layer_rows, qt->cols, qt->group_m, qt->group_n, stream);
    }
}

void gpt2_clear_ptq(GPT2 *model) {
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        cudaFreeCheck(&model->ptq.tensors[i].qvalues);
        cudaFreeCheck(&model->ptq.tensors[i].scales);
        model->ptq.tensors[i] = {};
    }
    model->ptq.original_weight_bytes = 0;
    model->ptq.quantized_weight_bytes = 0;
    model->ptq.num_quantized_tensors = 0;
    model->ptq.initialized = false;
}

void gpt2_clear_aq(GPT2 *model) {
    for (int i = 0; i < NUM_AQ_TENSORS; ++i) {
        cudaFreeCheck(&model->aq.tensors[i].qvalues);
        cudaFreeCheck(&model->aq.tensors[i].scales);
        model->aq.tensors[i] = {};
    }
    model->aq.initialized = false;
}

void gpt2_init_common(GPT2 *model) {
    // common inits outside of the model weights
    // memory lazily initialized in forward()
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    // the B,T params are determined and set, fixed on first batch in forward()
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f designates no loss, set at end of forward()
    model->params_memory = NULL;
    // memory lazily initialized in backward()
    model->grads_memory = NULL;
    model->workload_indices = NULL; // on cpu, for encoder_backward
    model->bucket_info = NULL; // on cpu, for encoder_backward
    // memory lazily initialized in update()
    model->optim_quant      = 0;
    model->optim_group_size = COAT_GROUP_SIZE;
    model->coat_expansion   = 1;
    model->m_memory    = NULL;
    model->v_memory    = NULL;
    model->m_qstate    = nullptr;
    model->v_qstate    = nullptr;
    model->m_scales    = nullptr;
    model->v_scales    = nullptr;
    model->m_kfactors  = nullptr;
    model->v_kfactors  = nullptr;
    model->master_weights = NULL;
    // beast-mode PTQ scratch (allocated in gpt2_prepare_ptq and gpt2_allocate_state)
    model->scratch_dequant = NULL;
    model->scratch_dequant_elems = 0;
    model->row_maxes_scratch = NULL;
    model->row_maxes_scratch_elems = 0;
    model->aq_enabled = 0;
    model->aq_type = AQ_TYPE_NONE;
    model->aq_group_size = 32;
    model->aq = {};
    model->aq_scratch = NULL;
    model->aq_scratch_elems = 0;
    model->aq_group_maxes_scratch = NULL;
    model->aq_group_maxes_scratch_elems = 0;
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
    // In BF16 mode this is the important safeguard: update the FP32 master copy,
    // then re-round back down into floatX for the next forward pass.
    model->use_master_weights = 1; // safe default: do keep master weights in fp32
    model->init_state = true;
    model->recompute = 1; // good default: recompute gelu but not layernorm
    model->gelu_fusion = 0; //deviceProp.major >= 9 ? 2 : 0; // default: off for now (default must match main())
    model->ptq_enabled = 0;
    model->ptq_precision = PTQ_PRECISION_NONE;
    model->ptq_group_size = 0;
    model->ptq = {};
}

// Beast-mode PTQ setup.
// Called once after weights are loaded into params_memory.
// For each quantized tensor (qkvw, attprojw, fcw, fcprojw):
//   1. GPU-quantizes the floatX weights into qvalues + scales (no host copy).
//   2. Builds a compact new params_memory containing only the unquantized tensors.
//   3. Frees the original large params_memory block.
//   4. Sets the quantized params.* pointers to NULL so forward/backward
//      must dequantize on-demand into scratch_dequant.
void gpt2_prepare_ptq(GPT2 *model) {
    if (!model->ptq_enabled || model->ptq_precision == PTQ_PRECISION_NONE) {
        gpt2_clear_ptq(model);
        return;
    }
    if (!ptq_precision_is_supported(model->ptq_precision)) {
        fprintf(stderr, "PTQ precision '%s' is not supported.\n", ptq_precision_to_string(model->ptq_precision));
        exit(EXIT_FAILURE);
    }
    if (model->params_memory == nullptr) {
        fprintf(stderr, "PTQ requires model weights to be initialized first.\n");
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------ //
    // Step 1: figure out scratch sizes needed across all quantized tensors.
    // row_maxes_scratch stores per-(row, group) max-abs values, so its capacity
    // must hold the largest per-tensor (total_rows * num_groups) we'll see.
    // ------------------------------------------------------------------ //
    size_t max_group_maxes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i)) continue;
        PTQTensorLayout layout = ptq_tensor_layout_for_index(model->config, i);
        const int gs = ptq_resolve_group_size(model->ptq_group_size, layout.cols);
        const size_t total_rows = (size_t)layout.num_layers * layout.rows_per_layer;
        const size_t group_maxes = total_rows * (size_t)ptq_num_groups(layout.cols, gs);
        if (group_maxes > max_group_maxes) max_group_maxes = group_maxes;
    }
    if (model->row_maxes_scratch == nullptr || model->row_maxes_scratch_elems < max_group_maxes) {
        cudaFreeCheck(&model->row_maxes_scratch);
        cudaCheck(cudaMalloc((void**)&model->row_maxes_scratch, max_group_maxes * sizeof(float)));
        model->row_maxes_scratch_elems = max_group_maxes;
    }

    // ------------------------------------------------------------------ //
    // Step 2: GPU-quantize each transformer weight tensor into qvalues/scales
    // ------------------------------------------------------------------ //
    floatX** src_ptrs[NUM_PARAMETER_TENSORS];
    get_parameter_tensor_ptrs(&model->params, src_ptrs);

    model->ptq.original_weight_bytes = 0;
    model->ptq.quantized_weight_bytes = 0;
    model->ptq.num_quantized_tensors  = 0;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i)) continue;
        PTQTensorLayout layout = ptq_tensor_layout_for_index(model->config, i);
        const size_t total_rows     = (size_t)layout.num_layers * layout.rows_per_layer;
        const size_t total_elements = total_rows * layout.cols;
        const size_t qbytes         = ptq_qvalue_bytes(total_elements, model->ptq_precision);
        const int    group_size     = ptq_resolve_group_size(model->ptq_group_size, layout.cols);
        const size_t scale_count    = total_rows * (size_t)ptq_num_groups(layout.cols, group_size);
        assert(total_elements == model->param_elements[i]);

        QuantizedTensor* qt = &model->ptq.tensors[i];
        // Re-allocate buffers if shape (qbytes/scale_count) changed across runs.
        if (qt->initialized && (qt->qvalue_bytes != qbytes || qt->scale_count != scale_count)) {
            cudaFreeCheck(&qt->qvalues);
            cudaFreeCheck(&qt->scales);
            qt->qvalue_bytes = 0;
            qt->scale_count  = 0;
            qt->initialized = false;
        }
        if (!qt->initialized) {
            cudaCheck(cudaMalloc((void**)&qt->qvalues, qbytes));
            cudaCheck(cudaMalloc((void**)&qt->scales,  scale_count * sizeof(float)));
            qt->num_layers     = layout.num_layers;
            qt->rows_per_layer = layout.rows_per_layer;
            qt->cols           = layout.cols;
            qt->group_size     = group_size;
            qt->qvalue_bytes   = qbytes;
            qt->scale_count    = scale_count;
            qt->initialized    = true;
            model->ptq.num_quantized_tensors += 1;
        }
        model->ptq.original_weight_bytes += total_elements * sizeof(floatX);
        model->ptq.quantized_weight_bytes += qbytes + scale_count * sizeof(float);
        // GPU-quantize: floatX params -> uint8 qvalues + float (group-wise) scales
        ptq_quantize_rows_gpu(qt->qvalues, qt->scales, model->row_maxes_scratch,
                              *src_ptrs[i], (int)total_rows, layout.cols, group_size,
                              model->ptq_precision, main_stream);
    }
    cudaCheck(cudaStreamSynchronize(main_stream));
    model->ptq.initialized = true;

    // ------------------------------------------------------------------ //
    // Step 3: Build a compact params_memory holding ONLY unquantized tensors.
    // ------------------------------------------------------------------ //
    // Calculate size of the compact block
    size_t compact_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i))
            compact_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    void* compact_memory;
    cudaCheck(cudaMalloc(&compact_memory, compact_bytes));

    // Copy each unquantized tensor from the old block into the new compact block,
    // and update the params.* pointer to point into the new block.
    floatX** ptrs[NUM_PARAMETER_TENSORS];
    get_parameter_tensor_ptrs(&model->params, ptrs);
    char* compact_iter = (char*)compact_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        size_t tensor_bytes = model->param_elements[i] * model->param_sizeof[i];
        if (ptq_should_quantize_tensor(i)) {
            // Quantized tensor: clear the pointer so any accidental access is caught.
            *(ptrs[i]) = nullptr;
        } else {
            cudaCheck(cudaMemcpyAsync(compact_iter, *(ptrs[i]), tensor_bytes,
                                     cudaMemcpyDeviceToDevice, main_stream));
            *(ptrs[i]) = (floatX*)compact_iter;
            compact_iter += tensor_bytes;
        }
    }
    cudaCheck(cudaStreamSynchronize(main_stream));

    // Free the original large params_memory and install the compact one
    cudaCheck(cudaFree(model->params_memory));
    model->params_memory = compact_memory;

    // ------------------------------------------------------------------ //
    // Step 4: allocate scratch_dequant (max single-layer quantized weight size)
    // ------------------------------------------------------------------ //
    // Largest single-layer tensor: fcw = 4*C*C, fcprojw = C*4*C (same size)
    size_t max_layer_elems = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i)) continue;
        PTQTensorLayout layout = ptq_tensor_layout_for_index(model->config, i);
        size_t layer_elems = (size_t)layout.rows_per_layer * layout.cols;
        if (layer_elems > max_layer_elems) max_layer_elems = layer_elems;
    }
    if (model->scratch_dequant == nullptr || model->scratch_dequant_elems < max_layer_elems) {
        cudaFreeCheck(&model->scratch_dequant);
        cudaCheck(cudaMalloc((void**)&model->scratch_dequant,
                             max_layer_elems * sizeof(floatX)));
        model->scratch_dequant_elems = max_layer_elems;
    }
    // ------------------------------------------------------------------ //
    // Detailed per-tensor weight memory report
    // ------------------------------------------------------------------ //
    // Total original weight bytes (all tensors, floatX)
    size_t total_original_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i)
        total_original_bytes += model->param_elements[i] * sizeof(floatX);

    // Total new weight bytes = compact floatX + qvalues(uint8) + scales(float) + scratch
    size_t scratch_bytes = max_layer_elems * sizeof(floatX);
    size_t total_new_bytes = compact_bytes                         // unquantized floatX
                           + model->ptq.quantized_weight_bytes     // qvalues + scales
                           + scratch_bytes;                        // scratch_dequant

    printf0("\n");
    printf0("[beast-ptq] Weight memory breakdown\n");
    printf0("+-----------------+----------+-----------+-----------+-----------+\n");
    printf0("| Tensor          | Elements |  Orig MiB |   New MiB |  Saved MB |\n");
    printf0("+-----------------+----------+-----------+-----------+-----------+\n");
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        size_t elems  = model->param_elements[i];
        double orig   = (double)(elems * sizeof(floatX)) / (1024.0 * 1024.0);
        double newmib, saved;
        if (ptq_should_quantize_tensor(i)) {
            PTQTensorLayout lo = ptq_tensor_layout_for_index(model->config, i);
            const int gs = ptq_resolve_group_size(model->ptq_group_size, lo.cols);
            const size_t total_rows = (size_t)lo.num_layers * lo.rows_per_layer;
            const size_t scale_count = total_rows * (size_t)ptq_num_groups(lo.cols, gs);
            double qval_mib    = (double)ptq_qvalue_bytes(elems, model->ptq_precision) / (1024.0 * 1024.0);
            double scale_mib   = (double)(scale_count * sizeof(float)) / (1024.0 * 1024.0);
            newmib = qval_mib + scale_mib;
            saved  = orig - newmib;
        } else {
            newmib = orig;   // kept as-is
            saved  = 0.0;
        }
        const char* marker = ptq_should_quantize_tensor(i) ? " *" : "  ";
        printf0("| %-13s%s | %8zu | %9.2f | %9.2f | %9.2f |\n",
                parameter_tensor_name(i), marker, elems, orig, newmib, saved);
    }
    printf0("+-----------------+----------+-----------+-----------+-----------+\n");
    printf0("  * = quantized (%s qvalues + float scales stored in beast ptq)\n",
            ptq_precision_to_string(model->ptq_precision));
    printf0("\n");
    printf0("  scratch_dequant  (reused per layer, NOT persistent)  : %6.2f MiB\n",
            (double)scratch_bytes / (1024.0 * 1024.0));
    printf0("\n");
    printf0("  Original weight memory (floatX, all %d tensors)      : %6.1f MiB\n",
            NUM_PARAMETER_TENSORS, (double)total_original_bytes / (1024.0 * 1024.0));
    printf0("  New weight memory      (compact floatX + qvalues)    : %6.1f MiB\n",
            (double)(compact_bytes + model->ptq.quantized_weight_bytes) / (1024.0 * 1024.0));
    printf0("  ----------------------------------------------------------\n");
    printf0("  Net weight memory saved                               : %6.1f MiB (%.1f%%)\n",
            (double)(total_original_bytes - compact_bytes - model->ptq.quantized_weight_bytes) / (1024.0 * 1024.0),
            100.0 * (1.0 - (double)(compact_bytes + model->ptq.quantized_weight_bytes) / (double)total_original_bytes));
    printf0("\n");
}

void gpt2_print_ptq_summary(const GPT2 *model) {
    if (!model->ptq_enabled || model->ptq_precision == PTQ_PRECISION_NONE || !model->ptq.initialized) {
        return;
    }
    char tensor_list[128];
    tensor_list[0] = '\0';
    bool first = true;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i)) {
            continue;
        }
        if (!first) {
            strncat(tensor_list, ",", sizeof(tensor_list) - strlen(tensor_list) - 1);
        }
        strncat(tensor_list, parameter_tensor_name(i), sizeof(tensor_list) - strlen(tensor_list) - 1);
        first = false;
    }
    const size_t original_bytes = model->ptq.original_weight_bytes;
    const size_t quantized_bytes = model->ptq.quantized_weight_bytes;
    const double compression_ratio = quantized_bytes > 0 ? (double)original_bytes / (double)quantized_bytes : 0.0;
    const double savings_pct = original_bytes > 0 ? 100.0 * (1.0 - (double)quantized_bytes / (double)original_bytes) : 0.0;

    // Build a human readable group size summary. Group size is per-tensor (resolved
    // against each tensor's cols), so list "perRow" if every quantized tensor ended up
    // with group_size == cols, otherwise list the requested value.
    char group_size_desc[64];
    bool all_per_row = true;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
        if (!ptq_should_quantize_tensor(i)) continue;
        const QuantizedTensor* qt = &model->ptq.tensors[i];
        if (qt->group_size != qt->cols) { all_per_row = false; break; }
    }
    if (model->ptq_group_size <= 0 || all_per_row) {
        snprintf(group_size_desc, sizeof(group_size_desc), "perRow");
    } else {
        snprintf(group_size_desc, sizeof(group_size_desc), "%d", model->ptq_group_size);
    }

    printf0("| ptq tensors list      | %-50s |\n", tensor_list);
    printf0("| ptq tensors           | %-50d |\n", model->ptq.num_quantized_tensors);
    printf0("| ptq group_size        | %-50s |\n", group_size_desc);
    printf0("| ptq original bytes    | %-50zu |\n", original_bytes);
    printf0("| ptq quantized bytes   | %-50zu |\n", quantized_bytes);
    printf0("| ptq compression       | %-50.2fx |\n", compression_ratio);
    printf0("| ptq size saving       | %-49.2f%% |\n", savings_pct);
    printf0("| ptq runtime overhead  | %-50zu |\n", quantized_bytes);
    printf0("+-----------------------+----------------------------------------------------+\n");
}

void gpt2_allocate_weights(GPT2 *model) {
    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    // create memory for model parameters on the device
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void gpt2_allocate_state(GPT2 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;

    // allocate the space
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute, model->aq_enabled != 0);
    const size_t aq_extra_bytes = aq_estimate_activation_bytes(model);
    model->acts_memory = malloc_and_point_activations(model->acts_specs, aq_extra_bytes);
    gpt2_prepare_aq(model);
    // also create memory for caching inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);

    // cudaMallocConditionallyManaged can fall back to cudaMallocManaged if not enough memory on device
    // and returns a status code of 1 if it had to fall back, in that case we want to print warning.
    int memory_status = 0;

    // We allocate optimizer state after weights because optimizer memory is always FP32,
    // regardless of whether params/grads are stored as BF16 or FP32.
    // This is where most of the training-time "mixed precision" memory split is set up.
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters; // num parameters we are responsible for
    if (model->optim_quant == 0) {
        printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
        printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
        assert(model->m_memory == nullptr);
        assert(model->v_memory == nullptr);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(float));
        memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(float));
    } else {
        int    gs         = model->optim_group_size;
        size_t num_groups = CEIL_DIV(shard_num_parameters, (size_t)gs);
        // INT4: nibble-packed → 0.5 bytes/param; FP8/INT8: 1 byte/param.
        size_t qstate_bytes = (model->optim_quant == 3)
                              ? (shard_num_parameters + 1) / 2
                              : shard_num_parameters * sizeof(uint8_t);
        size_t meta_bytes   = num_groups * sizeof(float);
        // All quantized modes now use COAT-style k-factors for dynamic range expansion.
        const char* mode_name[] = {"", "FP8(COAT)", "INT8(COAT)", "INT4(COAT)"};
        printf0("allocating %zu MiB for %s optimizer state m (qstate + scales + kfactors)\n",
                (qstate_bytes + 2 * meta_bytes) >> 20, mode_name[model->optim_quant]);
        printf0("allocating %zu MiB for %s optimizer state v (qstate + scales + kfactors)\n",
                (qstate_bytes + 2 * meta_bytes) >> 20, mode_name[model->optim_quant]);
        assert(model->m_qstate == nullptr && model->v_qstate == nullptr);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->m_qstate,    qstate_bytes);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->v_qstate,    qstate_bytes);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->m_scales,    meta_bytes);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->v_scales,    meta_bytes);
        // All modes need k-factors (INT8 and INT4 use COAT expansion like FP8).
        memory_status |= cudaMallocConditionallyManaged((void**)&model->m_kfactors,  meta_bytes);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->v_kfactors,  meta_bytes);
    }

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }

    // report on mixed memory allocation status (re-using our float reduce function, bit awk ok)
    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged when initializing m,v,master_weights on %d GPUs\n", reduced_memory_status);
        printf0("         Prevents an OOM, but code may run much slower due to device <-> host memory movement\n");
    }
    // report on device memory usage
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void gpt2_write_to_checkpoint(GPT2 *model, const char* checkpoint_path) {
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    // Version encoding:
    //   3 = fp32, padded vocab (original)
    //   5 = bf16, padded vocab, layernorms in bf16 (original)
    //   7 = fp32 + beast-mode int8 PTQ
    //   8 = bf16 + beast-mode int8 PTQ
    //   9 = fp32 + beast-mode fp8 PTQ
    //  10 = bf16 + beast-mode fp8 PTQ
    //  11 = fp32 + beast-mode int4 PTQ
    //  12 = bf16 + beast-mode int4 PTQ
    bool beast = model->ptq_enabled && model->ptq.initialized;
    if (!beast) {
        model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5;
    } else {
        int base = PRECISION_MODE == PRECISION_FP32 ? 7 : 8;
        int precision_offset = 0;
        if (model->ptq_precision == PTQ_PRECISION_FP8) {
            precision_offset = 2;
        } else if (model->ptq_precision == PTQ_PRECISION_INT4) {
            precision_offset = 4;
        }
        model_header[1] = base + precision_offset;
        model_header[8] = (int)model->ptq_precision; // store precision enum in header
        // Group size for grouped quantization. 0 means per-row (legacy). Stored as the
        // requested value (so reload knows what to resolve against per-tensor cols).
        model_header[9] = model->ptq_group_size;
    }
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.channels;
    model_header[7] = model->config.padded_vocab_size;
    fwriteCheck(model_header, sizeof(int), 256, model_file);

    if (!beast) {
        // Original path: dump the full floatX params_memory blob.
        device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                       IO_BUF_SIZE, main_stream);
    } else {
        // Beast path: write each tensor in its canonical stored form.
        // Unquantized tensors are written as floatX from params_memory.
        // Quantized tensors are written as (qvalues: uint8) then (scales: float).
        floatX** ptrs[NUM_PARAMETER_TENSORS];
        get_parameter_tensor_ptrs(&model->params, ptrs);
        for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
            if (!ptq_should_quantize_tensor(i)) {
                size_t bytes = model->param_elements[i] * sizeof(floatX);
                device_to_file(model_file, *(ptrs[i]), bytes, IO_BUF_SIZE, main_stream);
            } else {
                const QuantizedTensor* qt = &model->ptq.tensors[i];
                size_t qbytes = qt->qvalue_bytes;
                size_t sbytes = qt->scale_count * sizeof(float);
                device_to_file(model_file, qt->qvalues, qbytes, IO_BUF_SIZE, main_stream);
                device_to_file(model_file, qt->scales,  sbytes, IO_BUF_SIZE, main_stream);
            }
        }
    }
    fcloseCheck(model_file);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init=true) {
    // If weight_init is true, load weights from this .bin checkpoint.
    // weight_init=false is used when weights will be restored from FP32 master weights in the state file.

    if (PRECISION_MODE == PRECISION_FP16) {
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    // Accepted versions:
    //  3 = fp32 (original)       5 = bf16 (original)
    //  7/8 = fp32/bf16 + beast int8    9/10 = fp32/bf16 + beast fp8
    //  11/12 = fp32/bf16 + beast int4
    bool is_beast_ckpt = (version >= 7 && version <= 12);
    if (!(version == 3 || version == 5 || is_beast_ckpt)) {
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    if (weight_init && !is_beast_ckpt) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: compile with `make train_gpt2cu PRECISION=FP32`\n");
            exit(EXIT_FAILURE);
        }
    }

    model->config.max_seq_len       = model_header[2];
    model->config.vocab_size        = model_header[3];
    model->config.num_layers        = model_header[4];
    model->config.num_heads         = model_header[5];
    model->config.channels          = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // Allocate full params_memory (needed as landing buffer; gpt2_prepare_ptq compacts it later
    // when PTQ is enabled from a fresh .bin, or we compact inline below for beast checkpoints).
    gpt2_allocate_weights(model);

    if (weight_init) {
        assert(model->params_memory != NULL);
        if (!is_beast_ckpt) {
            // Original format: one contiguous floatX blob.
            file_to_device(model->params_memory, model_file, model->num_parameters_bytes,
                           IO_BUF_SIZE, main_stream);
        } else {
            // Beast format: tensor-by-tensor.
            // Unquantized tensors → loaded as floatX into params.* slots.
            // Quantized tensors   → loaded as (qvalues uint8, scales float) into ptq.tensors[i].
            PTQPrecision ckpt_prec = (PTQPrecision)model_header[8];
            if (!ptq_precision_is_supported(ckpt_prec)) {
                fprintf(stderr, "Unsupported PTQ precision in checkpoint: %d\n", (int)ckpt_prec);
                exit(EXIT_FAILURE);
            }
            model->ptq_precision = ckpt_prec;
            // Group size: 0 in legacy headers (zeroed memset above) means per-row, which
            // resolves per-tensor to group_size = cols. Stored requested value otherwise.
            const int ckpt_group_size = model_header[9];
            model->ptq_group_size = ckpt_group_size;
            model->ptq.original_weight_bytes = 0;
            model->ptq.quantized_weight_bytes = 0;
            model->ptq.num_quantized_tensors = 0;
            floatX** ptrs[NUM_PARAMETER_TENSORS];
            get_parameter_tensor_ptrs(&model->params, ptrs);
            for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
                PTQTensorLayout layout = ptq_tensor_layout_for_index(model->config, i);
                size_t total_rows     = (size_t)layout.num_layers * layout.rows_per_layer;
                size_t total_elements = total_rows * layout.cols;
                size_t qbytes         = ptq_qvalue_bytes(total_elements, ckpt_prec);
                if (!ptq_should_quantize_tensor(i)) {
                    size_t bytes = model->param_elements[i] * sizeof(floatX);
                    file_to_device(*(ptrs[i]), model_file, bytes, IO_BUF_SIZE, main_stream);
                } else {
                    const int    group_size  = ptq_resolve_group_size(ckpt_group_size, layout.cols);
                    const size_t scale_count = total_rows * (size_t)ptq_num_groups(layout.cols, group_size);
                    QuantizedTensor* qt = &model->ptq.tensors[i];
                    if (!qt->initialized) {
                        cudaCheck(cudaMalloc((void**)&qt->qvalues, qbytes));
                        cudaCheck(cudaMalloc((void**)&qt->scales,  scale_count * sizeof(float)));
                        qt->num_layers     = layout.num_layers;
                        qt->rows_per_layer = layout.rows_per_layer;
                        qt->cols           = layout.cols;
                        qt->group_size     = group_size;
                        qt->qvalue_bytes   = qbytes;
                        qt->scale_count    = scale_count;
                        qt->initialized    = true;
                    }
                    model->ptq.original_weight_bytes += total_elements * sizeof(floatX);
                    model->ptq.quantized_weight_bytes += qbytes + scale_count * sizeof(float);
                    model->ptq.num_quantized_tensors += 1;
                    file_to_device(qt->qvalues, model_file, qbytes, IO_BUF_SIZE, main_stream);
                    file_to_device(qt->scales,  model_file, scale_count * sizeof(float),
                                   IO_BUF_SIZE, main_stream);
                }
            }
            model->ptq.initialized = true;
            cudaCheck(cudaStreamSynchronize(main_stream));

            // Compact params_memory: keep only unquantized tensors, null quantized ptrs.
            size_t compact_bytes = 0;
            for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i)
                if (!ptq_should_quantize_tensor(i))
                    compact_bytes += model->param_elements[i] * model->param_sizeof[i];
            void* compact_memory;
            cudaCheck(cudaMalloc(&compact_memory, compact_bytes));
            char* cit = (char*)compact_memory;
            for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
                size_t tb = model->param_elements[i] * model->param_sizeof[i];
                if (ptq_should_quantize_tensor(i)) {
                    *(ptrs[i]) = nullptr;
                } else {
                    cudaCheck(cudaMemcpyAsync(cit, *(ptrs[i]), tb,
                                             cudaMemcpyDeviceToDevice, main_stream));
                    *(ptrs[i]) = (floatX*)cit;
                    cit += tb;
                }
            }
            cudaCheck(cudaStreamSynchronize(main_stream));
            cudaCheck(cudaFree(model->params_memory));
            model->params_memory = compact_memory;

            // Allocate scratch_dequant for per-layer forward/backward dequant.
            size_t max_le = 0;
            for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
                if (!ptq_should_quantize_tensor(i)) continue;
                PTQTensorLayout lo = ptq_tensor_layout_for_index(model->config, i);
                size_t le = (size_t)lo.rows_per_layer * lo.cols;
                if (le > max_le) max_le = le;
            }
            if (model->scratch_dequant == nullptr || model->scratch_dequant_elems < max_le) {
                cudaFreeCheck(&model->scratch_dequant);
                cudaCheck(cudaMalloc((void**)&model->scratch_dequant, max_le * sizeof(floatX)));
                model->scratch_dequant_elems = max_le;
            }
        }
    }
    fcloseCheck(model_file);
    cudaCheck(cudaDeviceSynchronize());
}


void gpt2_set_hyperparameters(GPT2Config* config, const char* depth_str) {
    int depth = atoi(depth_str);
    assert(depth > 0); // atoi returns 0 if not a number
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; }   // (unofficial) gpt2-tiny (30M)
    else if (depth == 12) { channels = 768; num_heads = 12; }  // gpt2 (124M)
    else if (depth == 24) { channels = 1024; num_heads = 16; } // gpt2-medium (350M)
    else if (depth == 36) { channels = 1280; num_heads = 20; } // gpt2-large (774M)
    else if (depth == 48) { channels = 1600; num_heads = 25; } // gpt2-xl (1558M)
    else if (depth == 60) { channels = 1920; num_heads = 30; } // (unofficial) 2.7B
    else if (depth == 72) { channels = 2880; num_heads = 30; } // (unofficial) 7.3B
    else if (depth == 84) { channels = 3456; num_heads = 36; } // (unofficial) 12.2B
    else { fprintf(stderr, "Unsupported GPT-2 depth: %d\n", depth); exit(EXIT_FAILURE); }
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = num_heads;
    config->max_seq_len = 1024;
}

void gpt3_set_hyperparameters(GPT2Config* config, const char* channels_str) {
    // we use channels instead of depth for GPT-3 because GPT-3 model depths are not one-to-one
    // note that our models are not necessarily identical to GPT-3 because
    // we use dense attention, not the alternating dense/banded attention of GPT-3
    int channels = atoi(channels_str);
    assert(channels > 0); // atoi returns 0 if not a number
    int depth, head_size;
    if      (channels == 384)   { depth = 6;  head_size = 64; }  // (unofficial) gpt3-tiny (31M)
    else if (channels == 768)   { depth = 12; head_size = 64; }  // gpt3-small (125M)
    else if (channels == 1024)  { depth = 24; head_size = 64; }  // gpt3-medium (350M)
    else if (channels == 1536)  { depth = 24; head_size = 96; }  // gpt3-large (760M)
    else if (channels == 2048)  { depth = 24; head_size = 128; } // gpt3-xl (1.3B) [heads fixed]
    else if (channels == 2560)  { depth = 32; head_size = 80; }  // gpt3-2.7B
    else if (channels == 4096)  { depth = 32; head_size = 128; } // gpt3-6.7B
    else if (channels == 5140)  { depth = 40; head_size = 128; } // gpt3-13B
    else if (channels == 12288) { depth = 96; head_size = 128; } // gpt3 (175B)
    else { fprintf(stderr, "Unsupported GPT-3 channels: %d\n", channels); exit(EXIT_FAILURE); }
    assert(channels % head_size == 0);
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = channels / head_size;
    config->max_seq_len = 2048; // NOTE: GPT-3 uses context length of 2048 tokens, up from 1024 in GPT-2
}

void gpt_build_from_descriptor(GPT2 *model, const char* descriptor) {
    // The model descriptor can be:
    // - legacy format "dX", where X is number, e.g. "d12". This creates GPT-2 model with 12 layers.
    // - new explicit format "gpt2:dX", same as above, e.g. "gpt2:d48" for GPT-2 with 48 layers.
    // - "gpt3:cX", where X is now the channel count, e.g. "gpt3:c768" is the smallest GPT-3 model.

    // check the valid prexies and dispatch to the right setup function
    assert(descriptor != NULL);
    size_t len = strlen(descriptor);
    if (len > 1 && descriptor[0] == 'd') {
        gpt2_set_hyperparameters(&model->config, descriptor + 1); // pass along the depth str without the 'd'
    } else if (len > 6 && strncmp(descriptor, "gpt2:d", 6) == 0) {
        gpt2_set_hyperparameters(&model->config, descriptor + 6); // pass along the depth str without the 'gpt2:d'
    } else if (len > 6 && strncmp(descriptor, "gpt3:c", 6) == 0) {
        gpt3_set_hyperparameters(&model->config, descriptor + 6); // pass along the channels str without the 'gpt3:c'
    } else {
        fprintf(stderr, "Unsupported model descriptor: %s\n", descriptor); exit(EXIT_FAILURE);
    }

    // both GPT-2 and GPT-3 use the same tokenizer with 50257 tokens
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128 for CUDA kernel efficiency

    gpt2_allocate_weights(model);

    // allocate and random init the memory for all the parameters with GPT-2 schema
    // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
    // NOTE: assuming all parameters are of the type floatX, could be relaxed later
    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_memory_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_memory_cpu, 0, model->num_parameters_bytes);
    // fill in all the weights with random values
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    // we have to init all these tensors exactly in the order that PyTorch initializes them
    // so that we can match them up and get correctness and exactly the same initial conditions
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (int l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            // the layernorm parameters are all initialized to 1
            if (l == 0 && (i == 2 || i == 8 || i == 14)) { // only at l = 0 to init these just once
                for (size_t j = 0; j < model->param_elements[i]; j++) {
                    params_memory_cpu[offset + j] = 1.0f;
                }
            }
            // weights tensors are handled here
            if ((l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
              || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = model->param_elements[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    // for wte tensor (padded vocab) override to init V instead of Vp rows
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    // weight tensors, we are only initializing layer l
                    assert(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                // in GPT-2, the projections back into the residual stream are additionally
                // scaled by 1/sqrt(2*L) for training stability
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                // okay let's draw the random numbers and write them
                float *fp32_buffer = (float*)mallocCheck(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng);
                for (size_t j = 0; j < n; j++) {
                    params_memory_cpu[offset + layer_offset + j] = (floatX)fp32_buffer[j];
                }
                free(fp32_buffer);
            }
            offset += model->param_elements[i];
        }
    }

    // copy them to GPU
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes, cudaMemcpyHostToDevice));
    free(params_memory_cpu);
}

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    // validate B,T are not larger than the values used at initialisation
    // (smaller B,T are okay for inference only)
    if (B > model->batch_size || T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    // validate inputs, all indices must be in the range [0, V)
    // we can do this while the copies are already underway
    tokenCheck(inputs, B*T, V);

    // Forward pass overview:
    // 1) embeddings / residual stream are floatX
    // 2) GEMMs run through cuBLASLt using floatX inputs/outputs but FP32 compute accumulation
    //    (see cublas_compute in common_start and llmc/matmul.cuh)
    // 3) numerically sensitive reductions/statistics (layernorm mean/rstd, losses) stay float
    // There is no AMP/autocast framework here; precision is selected at compile time.
    ParameterTensors params = model->params; // PTQ constrains params in-place when enabled
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]

    // first layernorm isn't fused
    bool aq = model->aq_enabled && model->aq.initialized;
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);
    if (aq && model->recompute < 2) {
        aq_quantize_tensor_slice(model, AQ_TENSOR_LN1, 0, acts.ln1, main_stream);
    }

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // In beast mode, quantized weight ptrs (qkvw, attprojw, fcw, fcprojw) are NULL.
        // We dequantize each one into scratch_dequant immediately before use.
        // In non-PTQ mode, params.* are valid and we use them directly.
        bool beast = model->ptq_enabled && model->ptq.initialized;
        floatX* sd = model->scratch_dequant; // alias for readability

        // -- non-quantized weight pointers (always valid) --
        floatX* l_qkvb     = params.qkvb     + l * 3*C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w     = params.ln2w     + l * C;
        floatX* l_ln2b     = params.ln2b     + l * C;
        floatX* l_fcb      = params.fcb      + l * 4*C;
        floatX* l_fcprojb  = params.fcprojb  + l * C;

        // -- quantized weight pointers (valid only when !beast) --
        floatX* l_qkvw     = beast ? nullptr : params.qkvw     + l * 3*C * C;
        floatX* l_attprojw = beast ? nullptr : params.attprojw + l * C * C;
        floatX* l_fcw      = beast ? nullptr : params.fcw      + l * 4*C * C;
        floatX* l_fcprojw  = beast ? nullptr : params.fcprojw  + l * C * 4*C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2_store = (model->recompute < 2)
                            ? (aq ? model->aq_scratch : acts.ln2 + l * B * T * C)
                            : acts.lnf;
        floatX* l_ln2 = l_ln2_store;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu_store = (model->recompute < 1)
                                 ? (aq ? acts.scratch_bt4c : acts.fch_gelu + l * B * T * 4*C)
                                 : acts.fch_gelu;
        floatX* l_fch_gelu = l_fch_gelu_store;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.

        if (aq && model->recompute < 2) {
            aq_dequantize_tensor_slice(model, AQ_TENSOR_LN1, l, model->aq_scratch, main_stream);
            l_ln1 = model->aq_scratch;
        }

        // ---------- QKV matmul ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[4], l, model->ptq_precision, main_stream);
            l_qkvw = sd;
        }
        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T;
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        if (T != model->seq_len) {
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        #endif
        if (aq) {
            aq_quantize_tensor_slice(model, AQ_TENSOR_QKVR, l, l_qkvr, main_stream);
        }

        // ---------- attention projection ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[6], l, model->ptq_precision, main_stream);
            l_attprojw = sd;
        }
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        if (aq) {
            aq_quantize_tensor_slice(model, AQ_TENSOR_ATTY, l, l_atty, main_stream);
        }
        fused_residual_forward5(l_residual2, l_ln2_store, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);
        if (aq) {
            if (model->recompute < 2) {
                aq_quantize_tensor_slice(model, AQ_TENSOR_LN2, l, l_ln2_store, main_stream);
            }
            aq_quantize_tensor_slice(model, AQ_TENSOR_RESIDUAL2, l, l_residual2, main_stream);
        }

        // ---------- MLP up-projection (fc) ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[10], l, model->ptq_precision, main_stream);
            l_fcw = sd;
        }
        matmul_forward_cublaslt(l_fch_gelu_store, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        if (aq) {
            aq_quantize_tensor_slice(model, AQ_TENSOR_FCH, l, l_fch, main_stream);
            if (model->recompute < 1) {
                aq_quantize_tensor_slice(model, AQ_TENSOR_FCH_GELU, l, l_fch_gelu_store, main_stream);
                aq_dequantize_tensor_slice(model, AQ_TENSOR_FCH_GELU, l, model->aq_scratch, main_stream);
                l_fch_gelu = model->aq_scratch;
            }
        }

        // ---------- MLP down-projection (fcproj) ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[12], l, model->ptq_precision, main_stream);
            l_fcprojw = sd;
        }
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);

        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
            if (aq && model->recompute < 2) {
                aq_quantize_tensor_slice(model, AQ_TENSOR_LN1, l + 1, l_ln1, main_stream);
            }
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, main_stream);
        }
    }

    // Final logits. wte is always floatX (never nulled by beast mode).
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}



// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float gpt2_validate(GPT2 *model, const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    // forward the model itself
    gpt2_forward(model, inputs, B, T);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    // fused classifier: does the forward pass and first part of the backward pass
    const float dloss = 1.0f / (B * T); // results in the uniform average loss over all elements
    // note: we don't need to generate dlogits here
    cudaCheck(cudaMemset(acts.losses, 0, B*T*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V); // while the memcpy is underway, validate the targets
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B*T; i++) {
        mean_loss += model->cpu_losses[i];
    }
    mean_loss /= B*T;
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

void gpt2_backward_and_reduce(GPT2 *model, int* inputs, const int* targets, int grad_accum_steps, int micro_step) {
    if(model->grads_memory == nullptr) {
        fprintf(stderr, "Need to allocate gradients before backward");
        exit(EXIT_FAILURE);
    }
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    if (micro_step == 0) {
        // there are currently two state vars during the gradient accumulation inner loop:
        // 1) the losses accumulate += into acts.losses, reset here
        // 2) the gradients accumulate += into grads_memory, reset here
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters * sizeof(floatX), main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    ParameterTensors params = model->params; // PTQ constrains params in-place when enabled
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;

    // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    const float dloss = 1.0f / (float)(B * T * grad_accum_steps); // results in the uniform average loss over all elements
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, True, main_stream);

    // Backward pass mirrors forward. Gradients for learnable tensors are stored in floatX,
    // not FP32, so this code relies on BF16 being stable enough in practice plus FP32 Adam
    // moments/master weights on the update side. There is no gradient scaler here.

    // reset residual stream gradients (put here to work with gradient accumulation)
    floatX* dresidual = (floatX*)model->acts.scratch_btc; // the main buffer holding the gradient in the backward pass
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    // re-use the output buffer of the forward pass as a scratchpad during backward pass
    float*  scratchF = (float*)acts.output;
    floatX* scratchX = (floatX*)acts.output;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(model->acts.scratch_bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    // backward the final layernorm
    floatX* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, model->acts.scratch_bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, main_stream);

    // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
    // scratch for backward computations
    floatX* dl_btc = residual;

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        bool beast = model->ptq_enabled && model->ptq.initialized;
        bool aq = model->aq_enabled && model->aq.initialized;
        floatX* sd = model->scratch_dequant;

        // -- non-quantized weight pointers (always valid) --
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_ln1b = params.ln1b + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;

        // -- quantized weight pointers (valid only when !beast; set to scratch on demand below) --
        floatX* l_qkvw     = beast ? nullptr : params.qkvw     + l * 3*C * C;
        floatX* l_attprojw = beast ? nullptr : params.attprojw + l * C * C;
        floatX* l_fcw      = beast ? nullptr : params.fcw      + l * 4*C * C;
        floatX* l_fcprojw  = beast ? nullptr : params.fcprojw  + l * C * 4*C;

        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w    = grads.ln1w    + l * C;
        floatX* dl_ln1b    = grads.ln1b    + l * C;
        floatX* dl_qkvw    = grads.qkvw    + l * 3*C * C;
        floatX* dl_qkvb    = grads.qkvb    + l * 3*C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = grads.attprojb + l * C;
        floatX* dl_ln2w    = grads.ln2w    + l * C;
        floatX* dl_ln2b    = grads.ln2b    + l * C;
        floatX* dl_fcw     = grads.fcw     + l * 4*C * C;
        floatX* dl_fcb     = grads.fcb     + l * 4*C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        floatX* dl_fcprojb = grads.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? (aq ? nullptr : acts.ln2 + l * B * T * C) : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = (model->recompute < 1) ? (aq ? nullptr : acts.fch_gelu + l * B * T * 4*C) : acts.fch_gelu;

        if (aq) {
            aq_dequantize_tensor_slice(model, AQ_TENSOR_FCH, l, l_fch_pre_gelu, main_stream);
            aq_dequantize_tensor_slice(model, AQ_TENSOR_RESIDUAL2, l, l_residual2, main_stream);
            aq_dequantize_tensor_slice(model, AQ_TENSOR_ATTY, l, l_atty, main_stream);
            aq_dequantize_tensor_slice(model, AQ_TENSOR_QKVR, l, l_qkvr, main_stream);
        }

        floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;

        // start the backward pass for this layer
        if(model->recompute >= 1) {
            gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);
        }

        // ---------- fcprojw backward ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[12], l, model->ptq_precision, main_stream);
            l_fcprojw = sd;
        }
        if (aq && model->recompute < 1) {
            aq_dequantize_tensor_slice(model, AQ_TENSOR_FCH_GELU, l, model->aq_scratch, main_stream);
            l_fch_gelu = model->aq_scratch;
        }
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);

        if(model->recompute >= 2) {
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
        }

        // ---------- fcw backward ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[10], l, model->ptq_precision, main_stream);
            l_fcw = sd;
        }
        if (aq && model->recompute < 2) {
            aq_dequantize_tensor_slice(model, AQ_TENSOR_LN2, l, model->aq_scratch, main_stream);
            l_ln2 = model->aq_scratch;
        }
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);

        // ---------- attprojw backward ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[6], l, model->ptq_precision, main_stream);
            l_attprojw = sd;
        }
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T;
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
        #endif

        if(model->recompute >= 2) {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        }

        // ---------- qkvw backward ----------
        if (beast) {
            ptq_dequantize_layer_slice(sd, &model->ptq.tensors[4], l, model->ptq_precision, main_stream);
            l_qkvw = sd;
        }
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, scratchF, B, T, C, 3 * C, main_stream);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);

        // Accumulate gradients from this layer in a background stream.
        if(last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_ln1b,
                dl_qkvw, dl_qkvb,
                dl_attprojw, dl_attprojb,
                dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb,
                dl_fcprojw, dl_fcprojb
            };
            const size_t nelem[] = {
                C, C,
                3 * C * C, 3 * C,
                C * C, C,
                C, C,
                4 * C * C, 4 * C,
                C * 4 * C, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }
    encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);

    // Aggregate all gradients that are not part of the transformer blocks
    if(last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        // reduce the gradients for non-transformer block parameters
        floatX* const pointers[] = {grads.wte, grads.wpe, grads.lnfw, grads.lnfb};
        const size_t nelem[] = {Vp * C, T * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        model->mean_loss /= B*T*grad_accum_steps;
    } else {
        model->mean_loss = -1.f; // no loss available yet
    }
}

// Gets the offset of a specific tensor for a specific layer in the GPT2 model
// layer_id is ignored for weights that are not part of a transformer block
ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
    // first offset our way to the parameter tensor start
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id] ;
    // if we are in the transformer block, we need to additionally offset by the layer id
    if(2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

float gpt2_calculate_grad_norm(GPT2 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;

    // repurposing this buffer (which isn't needed now) to write grad norm into it
    float* grad_norm_squared = (float*)model->acts.output;
    float grad_norm_squared_cpu = 0.0f;

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        // because of the ncclReduceScatter() in backward,
        // grads_memory only contains the averaged gradients at the local shards,
        // so we only calculate the grad norm at the grads_memory belonging to the local shards
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool is_first_pass = (i == 0);
            if((i < 2 || i > 13)) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, 0, 1,
                                    max_num_block_sums, is_first_pass, main_stream);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, tensor.size, model->config.num_layers,
                                    max_num_block_sums, is_first_pass, main_stream);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
#if MULTI_GPU
        // further sum the (partial) squared norm across all GPUs
        ncclCheck(ncclAllReduce(grad_norm_squared, grad_norm_squared, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, main_stream));
#endif
    } else {
        // in regular DDP, backward has averaged the gradients across all GPUs
        // so each GPU can compute the squared norm over the whole grad vector, with no added comms needed
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters, 0, 1, max_num_block_sums, true, main_stream);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float grad_scale, int t,
                 MultiGpuConfig* multi_gpu_config, bool init_from_master_only=false) {
    // update the model parameters using the AdamW optimizer
    // keep in mind that optimizer sharding (ZeRO-1) assigns different parameters to different GPUs
    // so we may not be responsible for the entire parameter tensor
    // also, this function was very simple a while back but become very complex, only because we want to
    // selectively weight decay some, but not all tensors :(
    // TODO: revisit and probably refactor this entire function
    NVTX_RANGE_FN();
    bool quant_optim = model->optim_quant > 0;
    bool coat_exp    = (bool)model->coat_expansion;
    if(model->grads_memory == nullptr ||
       (!quant_optim && (model->m_memory == nullptr || model->v_memory == nullptr)) ||
       ( quant_optim && (model->m_qstate == nullptr || model->v_qstate == nullptr))) {
        fprintf(stderr, "Need to allocate optimizer state before update");
        exit(EXIT_FAILURE);
    }

    bool init_state = model->init_state;
    if(init_state) {
        model->init_state = false;
        NvtxRange rng("InitOpt");
        size_t np = multi_gpu_config->shard_num_parameters;
        if (!quant_optim) {
            cudaCheck(cudaMemset(model->m_memory, 0, np * sizeof(float)));
            cudaCheck(cudaMemset(model->v_memory, 0, np * sizeof(float)));
        } else {
            int    gs         = model->optim_group_size;
            size_t num_groups = CEIL_DIV(np, (size_t)gs);
            size_t qstate_bytes = (model->optim_quant == 3) ? (np + 1) / 2 : np * sizeof(uint8_t);
            cudaCheck(cudaMemset(model->m_qstate, 0, qstate_bytes));
            cudaCheck(cudaMemset(model->v_qstate, 0, qstate_bytes));
            cudaCheck(cudaMemset(model->m_scales,   0, num_groups * sizeof(float)));
            cudaCheck(cudaMemset(model->v_scales,   0, num_groups * sizeof(float)));
            // All quantized modes use COAT k-factors now.
            cudaCheck(cudaMemset(model->m_kfactors, 0, num_groups * sizeof(float)));
            cudaCheck(cudaMemset(model->v_kfactors, 0, num_groups * sizeof(float)));
        }
    }

    // save RNG state at this point so we can round from master weights identically when restoring from a checkpoint
    model->rng_state_last_update = model->rng_state;

    // AdamW update precision story:
    // - read gradient from floatX and promote to float
    // - update m and v in float
    // - update parameter in float, preferably from master_weights
    // - stochastic-round the new value back into floatX for next forward pass
    //   (for beast mode: the floatX output is thrown away; we requantize from master_weights instead)
    bool beast = model->ptq_enabled && model->ptq.initialized;
    // Build a pointer array into model->params.* once, before the loop.
    // After gpt2_prepare_ptq, params_memory is the compact block and the
    // params.* pointers have been updated in-place to reflect the compact layout.
    // Using these directly avoids recomputing offsets that are wrong for compact memory.
    floatX* param_ptrs[NUM_PARAMETER_TENSORS];
    {
        floatX** pp[NUM_PARAMETER_TENSORS];
        get_parameter_tensor_ptrs(&model->params, pp);
        for (int k = 0; k < NUM_PARAMETER_TENSORS; ++k) param_ptrs[k] = *(pp[k]);
    }
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        unsigned int seed = random_u32(&model->rng_state);

        int num_layers = model->config.num_layers;
        if((i < 2 || i > 13)) { num_layers = 1; }

        ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
        ShardInfo shard  = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
        ptrdiff_t local_offset_full    = tensor.offset + shard.offset;
        ptrdiff_t local_offset_partial = tensor.offset / multi_gpu_config->num_processes;

        float wd = (i == 0 || i == 1 || i == 4 || i == 6 || i == 10 || i == 12) ? weight_decay : 0.0f;
        floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;

        ptrdiff_t opt_state_offset = multi_gpu_config->zero_stage < 1 ? local_offset_full : local_offset_partial;
        float* m_ptr = quant_optim ? nullptr : model->m_memory + opt_state_offset;
        float* v_ptr = quant_optim ? nullptr : model->v_memory + opt_state_offset;
        int    oq    = model->optim_quant;
        int    ogs   = model->optim_group_size;
        float* master_ptr = nullptr;
        if (model->master_weights != nullptr) { master_ptr = model->master_weights + opt_state_offset; }

        if (!beast || !ptq_should_quantize_tensor(i)) {
            // ----------------------------------------------------------------
            // Non-quantized path: param_ptr comes from model->params.* which
            // is always correct for both full (non-beast) and compact (beast)
            // params_memory layouts. Do NOT recompute from params_memory+offset.
            // ----------------------------------------------------------------
            floatX* param_ptr = param_ptrs[i] + shard.offset; // shard.offset==0 for single GPU
            if (init_state && model->master_weights != nullptr) {
                size_t grid_size = CEIL_DIV(shard.size, 512);
                copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, main_stream>>>(
                    master_ptr, param_ptr, shard.size, shard.size, tensor.size);
                cudaCheck(cudaGetLastError());
            }
            if (init_from_master_only) {
                init_from_master(param_ptr, master_ptr, shard.size, tensor.size, shard.size, num_layers, seed, main_stream);
            } else if (oq == 0) {
                adamw_update(param_ptr, master_ptr, grad_ptr,
                             m_ptr, v_ptr,
                             shard.size, tensor.size, tensor.size, shard.size, num_layers,
                             learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, main_stream);
            } else if (oq == 1) {
                adamw_update_coat(param_ptr, master_ptr, grad_ptr,
                                  model->m_qstate   + opt_state_offset,
                                  model->v_qstate   + opt_state_offset,
                                  model->m_scales   + opt_state_offset / ogs,
                                  model->v_scales   + opt_state_offset / ogs,
                                  model->m_kfactors + opt_state_offset / ogs,
                                  model->v_kfactors + opt_state_offset / ogs,
                                  shard.size, tensor.size, tensor.size, shard.size, num_layers, ogs,
                                  learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, coat_exp, main_stream);
            } else if (oq == 2) {
                adamw_update_int8(param_ptr, master_ptr, grad_ptr,
                                  model->m_qstate   + opt_state_offset,
                                  model->v_qstate   + opt_state_offset,
                                  model->m_scales   + opt_state_offset / ogs,
                                  model->v_scales   + opt_state_offset / ogs,
                                  model->m_kfactors + opt_state_offset / ogs,
                                  model->v_kfactors + opt_state_offset / ogs,
                                  shard.size, tensor.size, tensor.size, shard.size, num_layers, ogs,
                                  learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, coat_exp, main_stream);
            } else { // INT4
                adamw_update_int4(param_ptr, master_ptr, grad_ptr,
                                  model->m_qstate   + opt_state_offset / 2,
                                  model->v_qstate   + opt_state_offset / 2,
                                  model->m_scales   + opt_state_offset / ogs,
                                  model->v_scales   + opt_state_offset / ogs,
                                  model->m_kfactors + opt_state_offset / ogs,
                                  model->v_kfactors + opt_state_offset / ogs,
                                  shard.size, tensor.size, tensor.size, shard.size, num_layers, ogs,
                                  learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, coat_exp, main_stream);
            }
        } else {
            // ----------------------------------------------------------------
            // Beast-mode quantized path: tensor lives in ptq.tensors[i].
            // Process one layer at a time, using scratch_dequant as floatX staging.
            // ----------------------------------------------------------------
            QuantizedTensor* qt = &model->ptq.tensors[i];
            const size_t layer_elems = (size_t)qt->rows_per_layer * qt->cols; // elements per layer
            assert(model->ptq_precision != PTQ_PRECISION_INT4 || (layer_elems % 2 == 0));
            // grad tensor offset for layer 0 of this tensor (grads_memory layout is same as original params_memory)
            floatX* grad_base = (floatX*)model->grads_memory + tensor.offset;

            for (int l = 0; l < num_layers; ++l) {
                floatX* sd = model->scratch_dequant;
                // Dequantize layer l's current quantized weights into sd (floatX)
                ptq_dequantize_layer_slice(sd, qt, l, model->ptq_precision, main_stream);

                float* layer_master_ptr = master_ptr ? master_ptr + l * (ptrdiff_t)layer_elems : nullptr;
                floatX* layer_grad_ptr  = grad_base  + l * layer_elems;

                if (init_state && layer_master_ptr != nullptr) {
                    // First-touch: initialize FP32 master from the dequantized floatX
                    size_t grid_size = CEIL_DIV(layer_elems, 512);
                    copy_and_cast_kernel<<<grid_size, 512, 0, main_stream>>>(
                        layer_master_ptr, sd, layer_elems, layer_elems, layer_elems);
                    cudaCheck(cudaGetLastError());
                }

                if (!init_from_master_only) {
                    // AdamW: sd is the param (for weight decay reads); master_ptr gets the FP32 update;
                    // sd also receives the stochastic-rounded floatX output (we discard it below).
                    ptrdiff_t layer_param_offset = opt_state_offset + l * (ptrdiff_t)layer_elems;
                    ptrdiff_t layer_meta_offset  = layer_param_offset / ogs;
                    if (oq == 0) {
                        adamw_update(sd, layer_master_ptr, layer_grad_ptr,
                                     m_ptr + l * (ptrdiff_t)layer_elems,
                                     v_ptr + l * (ptrdiff_t)layer_elems,
                                     layer_elems, layer_elems, layer_elems, layer_elems, 1,
                                     learning_rate, beta1, beta2, t, eps, wd, grad_scale,
                                     seed + (unsigned int)l, main_stream);
                    } else if (oq == 1) {
                        adamw_update_coat(sd, layer_master_ptr, layer_grad_ptr,
                                          model->m_qstate   + layer_param_offset,
                                          model->v_qstate   + layer_param_offset,
                                          model->m_scales   + layer_meta_offset,
                                          model->v_scales   + layer_meta_offset,
                                          model->m_kfactors + layer_meta_offset,
                                          model->v_kfactors + layer_meta_offset,
                                          layer_elems, layer_elems, layer_elems, layer_elems, 1, ogs,
                                          learning_rate, beta1, beta2, t, eps, wd, grad_scale,
                                          seed + (unsigned int)l, coat_exp, main_stream);
                    } else if (oq == 2) {
                        adamw_update_int8(sd, layer_master_ptr, layer_grad_ptr,
                                          model->m_qstate   + layer_param_offset,
                                          model->v_qstate   + layer_param_offset,
                                          model->m_scales   + layer_meta_offset,
                                          model->v_scales   + layer_meta_offset,
                                          model->m_kfactors + layer_meta_offset,
                                          model->v_kfactors + layer_meta_offset,
                                          layer_elems, layer_elems, layer_elems, layer_elems, 1, ogs,
                                          learning_rate, beta1, beta2, t, eps, wd, grad_scale,
                                          seed + (unsigned int)l, coat_exp, main_stream);
                    } else { // INT4
                        adamw_update_int4(sd, layer_master_ptr, layer_grad_ptr,
                                          model->m_qstate   + layer_param_offset / 2,
                                          model->v_qstate   + layer_param_offset / 2,
                                          model->m_scales   + layer_meta_offset,
                                          model->v_scales   + layer_meta_offset,
                                          model->m_kfactors + layer_meta_offset,
                                          model->v_kfactors + layer_meta_offset,
                                          layer_elems, layer_elems, layer_elems, layer_elems, 1, ogs,
                                          learning_rate, beta1, beta2, t, eps, wd, grad_scale,
                                          seed + (unsigned int)l, coat_exp, main_stream);
                    }
                    // Re-quantize from master_weights (FP32) → qvalues/scales.
                    // This is more accurate than quantizing from the stochastic-rounded floatX in sd.
                    const float* src = layer_master_ptr ? layer_master_ptr : nullptr;
                    if (src != nullptr) {
                        const int num_groups_t = ptq_num_groups(qt->cols, qt->group_size);
                        size_t scale_offset = (size_t)l * qt->rows_per_layer * num_groups_t;
                        size_t elem_offset  = (size_t)l * qt->rows_per_layer * qt->cols;
                        size_t byte_offset  = ptq_qvalue_bytes(elem_offset, model->ptq_precision);
                        ptq_quantize_rows_gpu_fp32(
                            qt->qvalues + byte_offset,
                            qt->scales  + scale_offset,
                            model->row_maxes_scratch,
                            src, qt->rows_per_layer, qt->cols, qt->group_size,
                            model->ptq_precision, main_stream);
                    } else {
                        // No master weights: adamw wrote the updated params into sd (floatX).
                        // Re-quantize directly from sd using the floatX source path.
                        const int num_groups_t = ptq_num_groups(qt->cols, qt->group_size);
                        size_t scale_offset = (size_t)l * qt->rows_per_layer * num_groups_t;
                        size_t elem_offset  = (size_t)l * qt->rows_per_layer * qt->cols;
                        size_t byte_offset  = ptq_qvalue_bytes(elem_offset, model->ptq_precision);
                        ptq_quantize_rows_gpu(
                            qt->qvalues + byte_offset,
                            qt->scales  + scale_offset,
                            model->row_maxes_scratch,
                            sd, qt->rows_per_layer, qt->cols, qt->group_size,
                            model->ptq_precision, main_stream);
                    }
                } else {
                    // Checkpoint-resume: restore floatX param from master (for non-beast we'd use init_from_master)
                    // For beast: dequantize the master back into qvalues/scales
                    if (master_ptr != nullptr) {
                        const int num_groups_t = ptq_num_groups(qt->cols, qt->group_size);
                        size_t scale_offset = (size_t)l * qt->rows_per_layer * num_groups_t;
                        size_t elem_offset  = (size_t)l * qt->rows_per_layer * qt->cols;
                        size_t byte_offset  = ptq_qvalue_bytes(elem_offset, model->ptq_precision);
                        ptq_quantize_rows_gpu_fp32(
                            qt->qvalues + byte_offset,
                            qt->scales  + scale_offset,
                            model->row_maxes_scratch,
                            layer_master_ptr, qt->rows_per_layer, qt->cols, qt->group_size,
                            model->ptq_precision, main_stream);
                    }
                }
            }
        }
        // ZeRO-1 all-gather (single GPU: no-op)
        if (multi_gpu_config->zero_stage == 1 && !beast) {
#if MULTI_GPU
            ncclCheck(ncclGroupStart());
            floatX* param_ptr = (floatX*)model->params_memory + local_offset_full;
            for(int l = 0; l < num_layers; ++l) {
                ncclCheck(ncclAllGather(param_ptr + l * tensor.size,
                                        (floatX*) model->params_memory + tensor.offset + l * tensor.size,
                                        shard.size, ncclFloatX,
                                        multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream));
            }
            ncclCheck(ncclGroupEnd());
#endif
        }
    }
    // Beast mode: qvalues/scales are already up-to-date (updated per-layer above).
    // Non-beast mode: nothing extra needed.

    cudaCheck(cudaDeviceSynchronize());
}


float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt) {
    /*
    Estimate model flops utilization (MFU)
    ref: Section 2.1 of https://arxiv.org/pdf/2001.08361
    Note: Ideally, the N here would be only the parameters that actually
    participate in matrix multiplications. In this N, we are over-estimating by
    including LayerNorm params, biases, and the position embedding weights,
    but these are very small terms. Also keep in mind that we would want to exclude
    the token embedding weights, but in GPT-2 these are weight shared, so they
    participate in the classifier matmul, so they are correct to be included in N.
    Note 2: The first term (6 * N) in flops_per_token is all weight matmuls, the
    second is the attention matmul, which is also usually a small contribution.
    */
    size_t N = model->num_parameters;
    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    float flops_achieved = (float)flops_per_step * (1.0f / dt); // per second
    float flops_promised = get_flops_promised(deviceProp.name, PRECISION_MODE) * 1e12f;
    if(flops_promised < 0) {
        return -1.f;   // don't know
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

void gpt2_free(GPT2 *model) {
    gpt2_clear_ptq(model);
    gpt2_clear_aq(model);
    cudaFreeCheck(&model->scratch_dequant);
    cudaFreeCheck(&model->row_maxes_scratch);
    cudaFreeCheck(&model->aq_scratch);
    cudaFreeCheck(&model->aq_group_maxes_scratch);
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->m_qstate);
    cudaFreeCheck(&model->v_qstate);
    cudaFreeCheck(&model->m_scales);
    cudaFreeCheck(&model->v_scales);
    cudaFreeCheck(&model->m_kfactors);
    cudaFreeCheck(&model->v_kfactors);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
    cudaFreeCheck(&model->targets);
    cudaFreeCheck(&model->accumulated_mean_loss);
    cudaCheck(cudaFreeHost(model->cpu_losses));
    free(model->workload_indices);
    free(model->bucket_info);
}

// ----------------------------------------------------------------------------
// common init & free code for all of train/test/profile

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {

    // get CUDA device infos
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf("[System]\n");
        printf("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // In FP32 builds we may still run GEMMs in TF32 on Ampere/Hopper for throughput.
    // That is separate from BF16 storage: TF32 only changes how FP32 matmuls are executed.
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    #ifdef ENABLE_CUDNN
    create_cudnn();
    #endif
}

void common_free(GPT2 &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}


void save_state(const char* filename, int step, GPT2* model, DataLoader* loader) {
    printf("Writing state to %s\n", filename);
    FILE *state_file = fopenCheck(filename, "wb");
    int state_header[256];
    memset(state_header, 0, sizeof(state_header));
    // basic identifying information
    state_header[0] = 20240527; // magic number
    state_header[1] = 1; // version number
    state_header[2] = multi_gpu_config.num_processes; // number of processes
    state_header[3] = multi_gpu_config.process_rank; // rank of this process
    state_header[4] = model->use_master_weights;  // whether we're using fp32 master weights
    state_header[5] = loader->should_shuffle; // shuffle state of the dataloader
    // int main state, start at 10 to leave some padding
    state_header[10] = step; // step of the optimization
    // model rng state, start at 20 to leave some padding
    *((unsigned long long*)&state_header[20]) = model->rng_state; // random number generator state
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update; // last gpt2_update
    // dataloader state, start at 30 to leave some padding
    *((size_t*)&state_header[30]) = loader->current_shard_idx; // shard of the dataset
    *((size_t*)&state_header[32]) = loader->current_sample_idx; // position in shard
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    // write AdamW optimizer state and master_weights
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if (model->optim_quant == 0) {
        device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    } else {
        int    gs         = model->optim_group_size;
        size_t num_groups = CEIL_DIV(shard_num_parameters, (size_t)gs);
        size_t qstate_bytes = (model->optim_quant == 3)
                              ? (shard_num_parameters + 1) / 2
                              : shard_num_parameters * sizeof(uint8_t);
        device_to_file(state_file, model->m_qstate, qstate_bytes, IO_BUF_SIZE, main_stream);
        device_to_file(state_file, model->v_qstate, qstate_bytes, IO_BUF_SIZE, main_stream);
        device_to_file(state_file, model->m_scales,   num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        device_to_file(state_file, model->v_scales,   num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        // All quantized modes save k-factors (all use COAT-style expansion now).
        device_to_file(state_file, model->m_kfactors, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        device_to_file(state_file, model->v_kfactors, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
    }
    if(model->use_master_weights) {
        device_to_file(state_file, model->master_weights, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    }

    // write dataloader state if we are using the Permuted version of it
    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);  // number of shards
        fwriteCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void load_state(int* step, GPT2* model, DataLoader* loader, const char* filename) {
    FILE *state_file = fopenCheck(filename, "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    assert(state_header[0] == 20240527); // magic number
    assert(state_header[1] == 1); // version number
    assert(state_header[2] == multi_gpu_config.num_processes); // number of processes
    assert(state_header[3] == multi_gpu_config.process_rank); // rank of this process
    int use_master_weights = state_header[4];  // whether we're using fp32 master weights
    int should_shuffle = state_header[5]; // shuffle state of the dataloader
    *step = state_header[10]; // step of the optimization
    model->rng_state = *((unsigned long long*)&state_header[20]); // random number generator state
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]); // last gpt2_update
    size_t current_shard_idx = *((size_t*)&state_header[30]); // shard index
    size_t current_sample_idx = *((size_t*)&state_header[32]); // position in shard

    // read AdamW optimizer state and master_weights
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if(use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Master weights are present in state, but not enabled for current run.");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        printf0("Error: Master weights requested, but not present in state file.");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;      // we just got the state from file, no need to do first-touch init
    if (model->optim_quant == 0) {
        assert(model->m_memory != nullptr && model->v_memory != nullptr);
        file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    } else {
        int    gs         = model->optim_group_size;
        size_t num_groups = CEIL_DIV(shard_num_parameters, (size_t)gs);
        size_t qstate_bytes = (model->optim_quant == 3)
                              ? (shard_num_parameters + 1) / 2
                              : shard_num_parameters * sizeof(uint8_t);
        assert(model->m_qstate != nullptr && model->v_qstate != nullptr);
        file_to_device(model->m_qstate, state_file, qstate_bytes, IO_BUF_SIZE, main_stream);
        file_to_device(model->v_qstate, state_file, qstate_bytes, IO_BUF_SIZE, main_stream);
        file_to_device(model->m_scales,   state_file, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        file_to_device(model->v_scales,   state_file, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        // All quantized modes load k-factors.
        file_to_device(model->m_kfactors, state_file, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
        file_to_device(model->v_kfactors, state_file, num_groups * sizeof(float), IO_BUF_SIZE, main_stream);
    }
    if(model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        // restore weights from the FP32 master weights using the RNG state before last update
        // so the low-precision params are re-rounded identically to the original run.
        model->rng_state = model->rng_state_last_update;
        gpt2_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, &multi_gpu_config, /* init_from_master_only*/ true);
        model->rng_state = *((unsigned long long*)&state_header[20]); // use final RNG state from checkpoint after this
    }

    // revive the DataLoader object and its state
    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        // ensure the number of shards matches
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        // read the shard indices
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        // ensure the number of samples matches
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        // read the intra-shard indices
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        // read the shuffle rng state
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);

    // all done, close state file
    fcloseCheck(state_file);
}

void write_checkpoint(const char* output_log_dir, int step, GPT2* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    // only rank 0 writes the model file because it is the same across all ranks
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        gpt2_write_to_checkpoint(model, filename_buffer);
    }
    // all ranks write their state file
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    save_state(filename_buffer, step, model, train_loader);
    // DONE file is a signal that this checkpoint as a whole is complete
    multi_gpu_barrier(multi_gpu_config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* multi_gpu_config) {
    // mirrors write_checkpoint function, cleans up checkpoint from disk
    printf0("Deleting checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    remove(filename_buffer);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip everything below this point

// ----------------------------------------------------------------------------
// training resumption logic, very useful when jobs crash once in a while
// the goal is that we can resume optimization from any checkpoint, bit-perfect
// note that "state" refers to things not already saved in the model checkpoint file

// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)\n");
    fprintf(stderr, "  -o <string> output log dir (default = NULL, no logging; writes main.log and train_losses.csv)\n");
    fprintf(stderr, "  -lg <int>   log gpu info every x steps (default = -1; disabled)\n");
    fprintf(stderr, "  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)\n");
    fprintf(stderr, "  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)\n");
    fprintf(stderr, "  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.\n");
    fprintf(stderr, "  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)\n");
    // token layout for each step of the optimization
    fprintf(stderr, "  -b <int>    (per-GPU, micro) batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation\n");
    // workload (number of steps)
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    // optimization
    fprintf(stderr, "  -k <string> learning rate scheduler (default = cosine)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -u <int>    learning rate warmup iterations (default = 0, no warmup)\n");
    fprintf(stderr, "  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))\n");
    fprintf(stderr, "  -c <float>  weight decay (default = 0.0f)\n");
    fprintf(stderr, "  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)\n");
    fprintf(stderr, "  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)\n");
    // evaluation
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -h <int>    hellaswag eval run? (default = 0)\n");
    // debugging
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    // numerics
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)\n");
    // memory management
    fprintf(stderr, "  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)\n");
    fprintf(stderr, "  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln\n");
    fprintf(stderr, "  --ptq <0|1>           enable row-wise PTQ for large weight tensors (default = 0)\n");
    fprintf(stderr, "  --ptq_precision <str> PTQ precision for quantized weights: int8|fp8|int4 (default = int8)\n");
    fprintf(stderr, "  --ptq_group_size <int> group size along cols for group-wise int4 quantization\n");
    fprintf(stderr, "                         (default = 128 when precision=int4, 0 = per-row otherwise)\n");
    fprintf(stderr, "  --aq <0|1>            enable activation quantization (default = 0)\n");
    fprintf(stderr, "  --aq_type <str>       activation quantization type: fp8|int8|int4 (default = fp8)\n");
    fprintf(stderr, "  --aq_group_size <int> cols per group for activation quantization — COAT-style 1xN row groups (default = 32)\n");
    fprintf(stderr, "  --optim_quant <str>   optimizer state format: fp32|fp8|int8|int4 (default = fp32)\n");
    fprintf(stderr, "  --optim_group_size <int> group size for quantized moments (default = %d)\n", COAT_GROUP_SIZE);
    fprintf(stderr, "  --coat_expansion <0|1> COAT dynamic range expansion: 1=on (default), 0=plain absmax\n");
    // multi-node settings
    fprintf(stderr, "  -pn <int>    num_processes (default = 1)\n");
    fprintf(stderr, "  -pr <int>    process_rank (default = 0)\n");
    fprintf(stderr, "  -pg <int>    gpus_per_node (default = 8)\n");
    fprintf(stderr, "  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)\n");
    fprintf(stderr, "  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)\n");
    fprintf(stderr, "  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {
    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* load_filename = "gpt2_124M_bf16.bin"; // bf16 weights of the model
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = "logs";
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int B = 4; // batch size
    int T = 1024; // sequence length max
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 3e-4f;
    int log_gpu_every = -1;
    int warmup_iterations = 0;
    float final_learning_rate_frac = 1.0f; // final fraction of learning rate, at end of training
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f; // skip update if loss goes above this in zscore
    float skip_update_gradz = 0.0f; // skip update if grad_norm goes above this in zscore
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = -1;
    int override_enable_tf32 = 1;
    int use_master_weights = 0;
    int gelu_fusion = -1; // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
    int recompute = 1; // recompute during backward setting, 0 = none, 1 = recompute gelu
    int zero_stage = 0; // Zero Optimization Stage for Multi-GPU training
    int hellaswag_eval = 0;
    int ptq_enabled = 0;
    const char* ptq_precision_name = "int8";
    int ptq_group_size = -1; // -1 = unset; resolved after we know the precision
    const char* optim_quant_name = "fp32";
    int optim_group_size = COAT_GROUP_SIZE;
    int coat_expansion = 1;
    int aq_enabled = 0;
    const char* aq_type_name = "fp8";
    int aq_group_size = 32;
    // multi-node settings
    int num_processes = 1;  // this should be set by the slurm environment
    int process_rank = 0;  // this should be set by the slurm environment
    int gpus_per_node = 8;  // this should be set by the slurm environment
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // used if init_method set to "tcp" -> set to your server ip address
    char fs_path[256] = "";  // used if init_method set to "fs" -> set to a shared filesystem path
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (strcmp(argv[i], "--ptq") == 0) { ptq_enabled = atoi(argv[i+1]); continue; }
        if (strcmp(argv[i], "--ptq_precision") == 0) { ptq_precision_name = argv[i+1]; continue; }
        if (strcmp(argv[i], "--ptq_group_size") == 0) { ptq_group_size = atoi(argv[i+1]); continue; }
        if (strcmp(argv[i], "--aq") == 0) { aq_enabled = atoi(argv[i+1]); continue; }
        if (strcmp(argv[i], "--aq_type") == 0) { aq_type_name = argv[i+1]; continue; }
        if (strcmp(argv[i], "--aq_group_size") == 0) { aq_group_size = atoi(argv[i+1]); continue; }
        if (strcmp(argv[i], "--optim_quant") == 0) { optim_quant_name = argv[i+1]; continue; }
        if (strcmp(argv[i], "--optim_group_size") == 0) { optim_group_size = atoi(argv[i+1]); continue; }
        if (strcmp(argv[i], "--coat_expansion") == 0) { coat_expansion = atoi(argv[i+1]); continue; }
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { error_usage(); } // must be -x[y] (one dash, one or two letters)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i+1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y') { resume = atoi(argv[i+1]); }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); } // Per-GPU (micro) batch size
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == 'g') { log_gpu_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i+1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i+1]; }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i+1]); }
        else { error_usage(); }
    }

    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(override_enable_tf32, false); // common init code for train/test/profile

    // should do a bit more error checking here
    assert(warmup_iterations >= 0);
    if (output_log_dir != NULL) {
        assert(strlen(output_log_dir) < 400); // careful bunch of hardcoded snprintf around this
    }
    int tokens_per_fwdbwd = B * T * multi_gpu_config.num_processes; // one micro-batch processes this many tokens
    // calculate sensible default for total batch size as assuming no gradient accumulation
    if (total_batch_size == -1) { total_batch_size = tokens_per_fwdbwd; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    if (gelu_fusion == -1) { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_gpt2.py does the same)
    if (overfit_single_batch == 1) { train_data_pattern = val_data_pattern; }
    // Resolve default ptq_group_size before printing so the banner shows the effective value.
    // 128 for int4 (group-wise is the practical default), 0 (per-row) otherwise.
    if (ptq_group_size < 0) {
        const PTQPrecision pp = ptq_enabled ? ptq_precision_from_string(ptq_precision_name)
                                            : PTQ_PRECISION_NONE;
        ptq_group_size = (pp == PTQ_PRECISION_INT4) ? 128 : 0;
    }
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| Parameter             | Value                                              |\n");
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| train data pattern    | %-50s |\n", train_data_pattern);
    printf0("| val data pattern      | %-50s |\n", val_data_pattern);
    printf0("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf0("| checkpoint_every      | %-50d |\n", checkpoint_every);
    printf0("| resume                | %-50d |\n", resume);
    printf0("| micro batch size B    | %-50d |\n", B);
    printf0("| sequence length T     | %-50d |\n", T);
    printf0("| total batch size      | %-50d |\n", total_batch_size);
    printf0("| LR scheduler          | %-50s |\n", lr_scheduler_type);
    printf0("| learning rate (LR)    | %-50e |\n", learning_rate);
    printf0("| warmup iterations     | %-50d |\n", warmup_iterations);
    printf0("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
    printf0("| weight decay          | %-50e |\n", weight_decay);
    printf0("| skip update lossz     | %-50f |\n", skip_update_lossz);
    printf0("| skip update gradz     | %-50f |\n", skip_update_gradz);
    printf0("| max_steps             | %-50d |\n", max_steps);
    printf0("| val_loss_every        | %-50d |\n", val_loss_every);
    printf0("| val_max_steps         | %-50d |\n", val_max_steps);
    printf0("| sample_every          | %-50d |\n", sample_every);
    printf0("| genT                  | %-50d |\n", genT);
    printf0("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf0("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf0("| gelu_fusion           | %-50d |\n", gelu_fusion);
    printf0("| recompute             | %-50d |\n", recompute);
    printf0("| ptq enabled           | %-50s |\n", ptq_enabled ? "yes" : "no");
    printf0("| ptq precision         | %-50s |\n", ptq_enabled ? ptq_precision_name : "n/a");
    printf0("| ptq group_size (req)  | %-50d |\n", ptq_enabled ? ptq_group_size : 0);
    printf0("| aq enabled            | %-50s |\n", aq_enabled ? "yes" : "no");
    printf0("| aq type               | %-50s |\n", aq_enabled ? aq_type_name : "n/a");
    printf0("| aq group_size         | %-50d |\n", aq_enabled ? aq_group_size : 0);
    printf0("| optim_quant           | %-50s |\n", optim_quant_name);
    printf0("| optim_group_size      | %-50d |\n", optim_group_size);
    printf0("| coat_expansion        | %-50d |\n", coat_expansion);
    printf0("+-----------------------+----------------------------------------------------+\n");
    const char* precision_str = (PRECISION_MODE == PRECISION_FP32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16");
    printf0("| device                | %-50s |\n", deviceProp.name);
    printf0("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, PRECISION_MODE));
    printf0("| precision             | %-50s |\n", precision_str);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // figure out if we are going to be resuming the optimization
    int resuming = 0;
    // find the DONE file with the highest step count
    int resume_max_step = find_max_step(output_log_dir);
    if (resume == 1) { // is -y 1 resume flag set?
        assert(output_log_dir != NULL);
        if (resume_max_step != -1) {
            resuming = 1; // -y 1 is set, and we found a checkpoint we can resume from
            snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, resume_max_step);
        }
    }

    // build the GPT-2 model
    GPT2 model;
    gpt2_init_common(&model);
    if (resuming == 1) {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        bool weight_init = !use_master_weights;
        gpt2_build_from_checkpoint(&model, filename_buffer, weight_init);
    } else if (ends_with_bin(load_filename)) {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        gpt2_build_from_checkpoint(&model, load_filename);
    } else {
        // if it's not .bin, it could be a "special descriptor". This descriptor is used to
        // construct GPT-2 / GPT-3 models in a convenient format. See the function for docs.
        gpt_build_from_descriptor(&model, load_filename);
    }

    model.use_master_weights = use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = recompute;
    model.ptq_enabled = ptq_enabled;
    // Parse optim_quant name → integer
    {
        int oq = 0;
        if      (strcmp(optim_quant_name, "fp32") == 0) oq = 0;
        else if (strcmp(optim_quant_name, "fp8")  == 0) oq = 1;
        else if (strcmp(optim_quant_name, "int8") == 0) oq = 2;
        else if (strcmp(optim_quant_name, "int4") == 0) oq = 3;
        else {
            fprintf(stderr, "Unknown --optim_quant '%s'. Expected fp32|fp8|int8|int4.\n", optim_quant_name);
            exit(EXIT_FAILURE);
        }
        if (oq > 0) {
            bool is_pow2 = (optim_group_size > 0) && ((optim_group_size & (optim_group_size - 1)) == 0);
            if (!is_pow2 || optim_group_size < 4 || optim_group_size > 1024) {
                fprintf(stderr, "--optim_group_size must be a power of 2 in [4, 1024] (got %d)\n", optim_group_size);
                exit(EXIT_FAILURE);
            }
        }
        if (oq == 3 && optim_group_size % 2 != 0) {
            fprintf(stderr, "--optim_group_size must be even for int4 (got %d)\n", optim_group_size);
            exit(EXIT_FAILURE);
        }
        model.optim_quant      = oq;
        model.optim_group_size = optim_group_size;
        model.coat_expansion   = coat_expansion;
    }
    // Parse aq_type name → AQType
    {
        AQType at = AQ_TYPE_NONE;
        if (aq_enabled) {
            if      (strcmp(aq_type_name, "fp8")  == 0) at = AQ_TYPE_FP8;
            else if (strcmp(aq_type_name, "int8") == 0) at = AQ_TYPE_INT8;
            else if (strcmp(aq_type_name, "int4") == 0) at = AQ_TYPE_INT4;
            else {
                fprintf(stderr, "Unknown --aq_type '%s'. Expected fp8|int8|int4.\n", aq_type_name);
                exit(EXIT_FAILURE);
            }
            if (aq_group_size <= 0) {
                fprintf(stderr, "--aq_group_size must be positive (got %d)\n", aq_group_size);
                exit(EXIT_FAILURE);
            }
        }
        model.aq_enabled   = aq_enabled;
        model.aq_type      = at;
        model.aq_group_size = aq_group_size;
    }
    // Parse aq_type name → AQType
    {
        AQType at = AQ_TYPE_NONE;
        if (aq_enabled) {
            if      (strcmp(aq_type_name, "fp8")  == 0) at = AQ_TYPE_FP8;
            else if (strcmp(aq_type_name, "int8") == 0) at = AQ_TYPE_INT8;
            else if (strcmp(aq_type_name, "int4") == 0) at = AQ_TYPE_INT4;
            else {
                fprintf(stderr, "Unknown --aq_type '%s'. Expected fp8|int8|int4.\n", aq_type_name);
                exit(EXIT_FAILURE);
            }
            if (aq_group_size <= 0) {
                fprintf(stderr, "--aq_group_size must be positive (got %d)\n", aq_group_size);
                exit(EXIT_FAILURE);
            }
        }
        model.aq_enabled   = aq_enabled;
        model.aq_type      = at;
        model.aq_group_size = aq_group_size;
    }
    model.ptq_precision = ptq_enabled ? ptq_precision_from_string(ptq_precision_name) : PTQ_PRECISION_NONE;
    // ptq_group_size has already been resolved above (default 128 for int4, 0 per-row otherwise).
    model.ptq_group_size = ptq_group_size;
    if (!(resuming == 1 && use_master_weights == 1)) {
        gpt2_prepare_ptq(&model);
    }
    printf0("| weight init method    | %-50s |\n", resuming == 1 ? "intermediate checkpoint" : load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");
    gpt2_print_ptq_summary(&model);

    // build DataLoaders for both train and val
    int permute_train_loader = (overfit_single_batch == 1) ? 0 : 1;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, permute_train_loader);
    dataloader_init(&val_loader, val_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);
    // figure out the number of training steps we will run for
    int train_num_batches = max_steps; // passed in from command line
    if (train_num_batches == -1) {
        // sensible default is to train for exactly one epoch
        size_t ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = ntok / total_batch_size;
    }
    // figure out the number of validation steps to run for
    int val_num_batches = val_max_steps; // passed in from command line
    if (val_num_batches == -1) {
        // sensible default is to evaluate the full validation split
        size_t ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = ntok / tokens_per_fwdbwd;
    }
    printf0("| train_num_batches     | %-50d |\n", train_num_batches);
    printf0("| val_num_batches       | %-50d |\n", val_num_batches);
    printf0("+-----------------------+----------------------------------------------------+\n");
    // build an EvalLoader for HellaSwag
    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        evalloader_init(&eval_loader, hellaswag_path, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    }
    printf0("| run hellaswag         | %-50s |\n", run_hellaswag ? "yes" : "no");
    printf0("+-----------------------+----------------------------------------------------+\n");

    // pretty print in a table the multi-gpu configuration as well
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    printf0("| num_processes         | %-50d |\n", multi_gpu_config.num_processes);
    printf0("| zero_stage            | %-50d |\n", multi_gpu_config.zero_stage);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // prints outside of pretty table to here and below
    if (!hellaswag_available) {
        printf0("HellaSwag eval not found at %s, skipping its evaluation\n", hellaswag_path);
        printf0("You can run `python dev/data/hellaswag.py` to export and use it with `-h 1`.\n");
    }
    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf0("num_parameters: %zu => bytes: %zu\n", model.num_parameters, model.num_parameters_bytes);
    printf0("allocated %d MiB for model parameters\n", (int)round(model.num_parameters_bytes / (1024 * 1024)));
    // few more prints for gradient accumulation math up above
    printf0("batch_size B=%d * seq_len T=%d * num_processes=%d and total_batch_size=%d\n",
            B, T, multi_gpu_config.num_processes, total_batch_size);
    printf0("=> setting grad_accum_steps=%d\n", grad_accum_steps);

    // set up logging
    if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    Logger logger;
    logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);

    // set up the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // set up learning rate scheduler
    LearningRateScheduler lr_scheduler;
    lr_scheduler_init(&lr_scheduler, lr_scheduler_type, learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);

    // some memory for generating samples from the model
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // if we found a checkpoint to resume from, load the optimization state
    int step = 0;
    gpt2_allocate_state(&model, B, T);
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        load_state(&step, &model, &train_loader, filename_buffer);
        if (model.ptq_enabled && model.use_master_weights) {
            gpt2_print_ptq_summary(&model);
        }
    }

    // init an OutlierDetector the training loss
    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if (T < model.config.max_seq_len) {
        printf0("!!!!!!!!\n");
        printf0("WARNING:\n");
        printf0("- The training sequence length is: T=%d (set with -t)\n", T);
        printf0("- The model's max sequence length is: max_seq_len=%d\n", model.config.max_seq_len);
        printf0("You are attempting to train with a sequence length shorter than the model's max.\n");
        printf0("This will lead to unused parameters in the wpe position embedding weights.\n");
        printf0("If you know what you're doing you can ignore this warning.\n");
        printf0("If you're like ???, you are most likely misconfiguring your training run.\n");
        printf0("---> HINT: If you're training GPT-2 use -t 1024. If GPT-3, use -t 2048.\n");
        printf0("!!!!!!!!\n");
    }
    // in any case, this must be true or we'd index beyond the model's wpe (position embedding table)
    assert(T <= model.config.max_seq_len);

    // train
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaProfilerStart());
    double total_sum_iteration_time_s = 0.0;
    float ema_tokens_per_second = 0.0f;
    for (; step <= train_num_batches; step++) {
        NvtxRange step_range("Train step", step);

        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss (all processes collaborate)
        if (step % val_loss_every == 0 || last_step) {
            NvtxRange validation_range("validation");
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                val_loss += gpt2_validate(&model, val_loader.inputs, val_loader.targets, B, T);
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if (run_hellaswag &&
           ((step > 0 && step % val_loss_every == 0) || last_step)) {
            NvtxRange evaluation_range("evaluation");
            float eval_acc_norm = 0.0f;
            evalloader_reset(&eval_loader);
            for (int i = 0; i < eval_loader.num_batches; i++) {
                if (i % 10 == 0) { printf("evaluating HellaSwag: %d/%d\r", i, eval_loader.num_batches); }
                evalloader_next_batch(&eval_loader);
                gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
                int correct = evalloader_stat_losses(&eval_loader, model.cpu_losses);
                eval_acc_norm += (float)correct;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            printf0("HellaSwag: %d/%d = %f\n", (int)eval_acc_norm, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples);
            logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);
        }

        // once in a while do model inference to print generated text (only rank 0)
        if (multi_gpu_config.process_rank == 0 && sample_every > 0 &&
           (step > 0 && (step % sample_every) == 0 || last_step)) {
            NvtxRange generation_range("generation");
            unsigned long long sample_rng_state = 1337;
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            int eot_token = tokenizer.eot_token;
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                NvtxRange generation_range("Generation step", t);
                // we try not to be too wasteful for inference by not calculating all of B,T
                // Using a smaller B is always bit-for-bit identical, but T is more tricky
                // for non-CUDNN, we need to make sure the attention buffer is memset to 0
                // for cuDNN, it might suddenly decide to use a slightly different algorithm...
                // on cuDNN 9.2.1 with cuDNN FrontEnd 1.5.2, T >= 256 seems bit-for-bit identical
                // (but even if it wasn't fully identical that's probably not the end of the world)
                // note this is still somewhat wasteful because we don't have a KV cache!
                gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = (float)cpu_logits_raw[i];
                }
                // sample the next token
                float coin = random_f32(&sample_rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if ((checkpoint_every > 0 && output_log_dir != NULL && resuming == 0) &&
            ((step > 0 && step % checkpoint_every == 0) || last_step)) {
            // writes model .bin file, state .bin files, and DONE file for step
            write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            int step_delete = step - checkpoints_keep * checkpoint_every;
            if (checkpoints_keep > 0 && step_delete > 0 &&
               (major_checkpoint_every == 0 || step_delete % major_checkpoint_every != 0)
                ) {
                delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }
        resuming = 0;

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // --------------- TRAINING SECTION BEGIN -----------------
        if (overfit_single_batch == 1) {
            // if we are trying to overfit a single batch, we reset the loader here
            dataloader_reset(&train_loader);
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        cudaCheck(cudaEventRecord(start));
        // gradient and loss accumulation loop over micro-batches
        for (int micro_step = 0; micro_step < grad_accum_steps; micro_step++) {
            // fetch the next data batch
            dataloader_next_batch(&train_loader);
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            gpt2_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            gpt2_backward_and_reduce(&model, train_loader.inputs, train_loader.targets, grad_accum_steps, micro_step);
        }
        float zloss = (float)(update_detector(&loss_outlier_detector, (double)model.mean_loss)); // loss z-score
        // fetch the next learning rate
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        // calculate the gradient norm and how much we wish to scale the gradient
        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        float zgrad = (float)(update_detector(&grad_norm_outlier_detector, (double)grad_norm)); // grad z-score
        // update the model parameters
        if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
            printf0("skipping update due to loss z-score of %f\n", zloss);
        } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
            printf0("skipping update due to grad z-score of %f\n", zgrad);
        } else {
            // clip the gradient norm to a maximum value
            float grad_clip = 1.0f;
            float grad_scale = (grad_norm > grad_clip) ? grad_clip / grad_norm : 1.0f;
            gpt2_update(&model, step_learning_rate, 0.9f, 0.95f, 1e-8f, weight_decay, grad_scale, step+1, &multi_gpu_config);
        }
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end)); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        float bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0f;
            // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
            ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        }
        float mfu = gpt2_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| lr %.2e | %.2f ms | %.1f%% bf16 MFU | %.0f tok/s\n",
                step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad, step_learning_rate,
                time_elapsed_ms, 100*mfu, bias_corrected_ema_tokens_per_second);
        if(log_gpu_every > 0 && (step + 1) % log_gpu_every == 0) {
            GPUUtilInfo gpu_info = get_gpu_utilization_info();
            printf0("                  compute %2.1f%% | memory: %2.1f%% | fan: %2d%% | %4d MHz / %4d MHz | %3d W / %3d W | %d°C / %d°C | %s\n",
                    gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
                    gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
        }
        logger_log_train(&logger, step, model.mean_loss, step_learning_rate, grad_norm);
        logger_log_train_loss(&logger, step + 1, model.mean_loss, step_learning_rate, grad_norm,
                              zloss, zgrad, time_elapsed_ms, bias_corrected_ema_tokens_per_second);

        // disable the profiler after 3 steps of optimization
        if (step == 3) { cudaProfilerStop(); }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    printf0("total average iteration time: %f ms\n", total_sum_iteration_time_s / (train_num_batches-1) * 1000);

    // free and destroy everything
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaEventDestroy(start));
    if (run_hellaswag) { evalloader_free(&eval_loader); }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(cpu_logits_raw);
    free(cpu_logits);
    free(gen_tokens);
    multi_gpu_config_free(&multi_gpu_config);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
#endif
