/*
 * COAT FP8 optimizer state compression helpers — Phase 1: math primitives.
 *
 * Reference: Xi et al., "COAT: Compressing Optimizer States and Activations
 * for Memory-Efficient FP8 Training", 2025.
 *
 * This file is intentionally self-contained (only needs <cuda_fp8.h>).
 * Phase 1 scope: expand/unexpand math + per-group quantize/dequantize.
 * GPU kernel integration (adamw_update) happens in Phase 3.
 */
#pragma once

#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

// Group size for per-group quantization of optimizer states.
// Must be a power of 2. COAT paper uses 128; change at compile time with
// -DCOAT_GROUP_SIZE=64 etc. if you want to experiment.
#ifndef COAT_GROUP_SIZE
#define COAT_GROUP_SIZE 128
#endif

// FP8 E4M3 numerical constants.
// max representable = 448, min positive = 2^{-9} = 1/512
// Dynamic range ratio R_fp8 = 448 * 512 = 229376
// ln(229376) = 12.344  (used in k computation)
static constexpr float COAT_FP8_MAX       = 448.0f;
static constexpr float COAT_FP8_LOG_RANGE = 12.344f;  // ln(R_fp8)

// Bounds on k from the COAT paper (Appendix A).
// m typically lands in [1, 3], v in [5, 15].
static constexpr float COAT_K_MIN = 1.0f;
static constexpr float COAT_K_MAX = 15.0f;

// ---------------------------------------------------------------------------
// Core math: dynamic range expansion and its inverse
// ---------------------------------------------------------------------------

// f(x) = sign(x) * |x|^k
// Stretches a peaked distribution to fill FP8's dynamic range.
// k=1 short-circuit avoids powf(x,1) rounding noise (powf uses exp(log(x)) internally).
__host__ __device__ inline float coat_expand(float x, float k) {
    return (k == 1.0f) ? x : copysignf(powf(fabsf(x), k), x);
}

// f^{-1}(x) = sign(x) * |x|^{1/k}
// Recovers the original value after dequantization.
__host__ __device__ inline float coat_unexpand(float x, float k) {
    return (k == 1.0f) ? x : copysignf(powf(fabsf(x), 1.0f / k), x);
}

// Compute the expansion exponent k for a group whose actual dynamic range
// ratio is R_actual = max(|x|) / min(|x|, x≠0).
// Derivation: we want R_actual^k = R_fp8, so k = ln(R_fp8) / ln(R_actual).
__host__ __device__ inline float coat_compute_k(float R_actual) {
    // When R_actual <= 1, all values have the same magnitude — no range variation to
    // exploit. Expansion (k>1) only adds powf rounding noise. Use k=1 (identity).
    if (R_actual <= 1.0f) return COAT_K_MIN;
    float k = COAT_FP8_LOG_RANGE / logf(R_actual);
    return fminf(fmaxf(k, COAT_K_MIN), COAT_K_MAX);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 encode / decode (self-contained, no dependency on train_gpt2.cu)
// ---------------------------------------------------------------------------

__host__ __device__ inline uint8_t coat_fp8_encode(float val) {
    return __nv_fp8_e4m3(val).__x;
}

__host__ __device__ inline float coat_fp8_decode(uint8_t raw) {
    __nv_fp8_e4m3 v;
    v.__x = raw;
    return (float)v;
}

// ---------------------------------------------------------------------------
// Per-group quantize: float[n] -> uint8[n] + scale + k
//
// This is a sequential reference implementation intended for:
//   - CPU-side testing (Phase 1)
//   - Single-threaded device code if called from within a kernel (Phase 3
//     will replace the inner loops with warp-level reductions)
//
// 'n' is typically COAT_GROUP_SIZE but may be smaller for the last group
// when total parameters are not a multiple of COAT_GROUP_SIZE.
// ---------------------------------------------------------------------------

__host__ __device__ inline void coat_quantize_group(
    const float* __restrict__ src,
    uint8_t*     __restrict__ dst_fp8,
    float*                    out_scale,
    float*                    out_k,
    int                       n
) {
    // Pass 1: find max and min absolute values.
    // min_abs excludes exact zeros (they carry no dynamic-range information).
    float max_abs = 0.0f;
    float min_abs = 3.4e38f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
        if (a > 0.0f && a < min_abs) min_abs = a;
    }

    // Degenerate case: all zeros — nothing to represent.
    if (max_abs == 0.0f) {
        for (int i = 0; i < n; i++) dst_fp8[i] = 0;
        *out_scale = 1.0f;
        *out_k     = 1.0f;
        return;
    }

    // Degenerate case: single unique magnitude — no range to expand.
    if (min_abs >= max_abs) min_abs = max_abs;

    float R_actual = max_abs / min_abs;
    float k        = coat_compute_k(R_actual);

    // Guard: when max_abs < 1, large k causes max_abs^k to underflow in float32,
    // making scale = FP8_MAX / 0 = inf. Cap k so max_abs^k >= sqrt(FLT_MIN) ~1e-19,
    // which keeps scale <= 4e21 — well within float32 range.
    // sqrt(FLT_MIN) = 1.08e-19, log(1.08e-19) = -43.668
    if (max_abs > 0.0f && max_abs < 1.0f) {
        float k_limit = -43.668f / logf(max_abs);   // logf(max_abs) < 0, so k_limit > 0
        k = fminf(k, k_limit);
    }

    float max_expanded = coat_expand(max_abs, k);  // reuse k==1 short-circuit to avoid powf rounding
    float scale        = COAT_FP8_MAX / max_expanded;  // maps max_expanded -> 448

    // Pass 2: expand, scale, clamp, encode.
    for (int i = 0; i < n; i++) {
        float expanded = coat_expand(src[i], k) * scale;
        expanded = fmaxf(fminf(expanded, COAT_FP8_MAX), -COAT_FP8_MAX);
        dst_fp8[i] = coat_fp8_encode(expanded);
    }

    *out_scale = scale;
    *out_k     = k;
}

// ---------------------------------------------------------------------------
// Per-group dequantize: uint8[n] + scale + k -> float[n]
// ---------------------------------------------------------------------------

__host__ __device__ inline void coat_dequantize_group(
    const uint8_t* __restrict__ src_fp8,
    float                       scale,
    float                       k,
    float*         __restrict__ dst,
    int                         n
) {
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        // Undo the scale to recover the expanded value, then undo the expansion.
        float expanded = coat_fp8_decode(src_fp8[i]) * inv_scale;
        dst[i] = coat_unexpand(expanded, k);
    }
}
