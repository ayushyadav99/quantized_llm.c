/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
// FP8 E4M3 (±448): recommended for weights and forward activations.
// cuBLASLt FP8 GEMM requires sm_89+ (Ada Lovelace) or sm_90+ (Hopper).
// The output (C/D) matrix is forced to BF16 by matmul.cuh when sizeof(floatX)==1.
#elif defined(ENABLE_FP8_E4M3)
#define CUBLAS_LOWP CUDA_R_8F_E4M3
// FP8 E5M2 (±57344): recommended for gradients due to wider exponent range.
// Same hardware and output-matrix constraints as E4M3 apply.
#elif defined(ENABLE_FP8_E5M2)
#define CUBLAS_LOWP CUDA_R_8F_E5M2
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H