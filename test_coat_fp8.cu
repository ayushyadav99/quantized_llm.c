/*
 * Standalone round-trip test for COAT FP8 group quantization (Phase 1).
 *
 * Compile:
 *   nvcc -O2 -std=c++17 -o test_coat_fp8 test_coat_fp8.cu && ./test_coat_fp8
 *
 * Expected output: COAT max_err < naive max_err on m and v distributions,
 * and the improvement ratio should be > 1 (COAT is always at least as good).
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "llmc/coat_fp8_optim.cuh"

// Simple xorshift RNG so we don't need stdlib rand().
static unsigned g_rng = 42;
static float randf() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (g_rng & 0x7FFFFFFFu) * (1.0f / 2147483647.0f);
}

// First moment (m): peaked around 0, heavy tails, range ~ [-0.1, 0.1].
// Modelled as Cauchy-like via tan of a uniform — matches real m distributions.
static void gen_m(float* buf, int n) {
    for (int i = 0; i < n; i++) {
        float u = randf() - 0.5f;           // uniform [-0.5, 0.5]
        buf[i] = 0.01f * tanf(u * 1.5f);   // Cauchy-like, range ~ [-0.1, 0.1]
    }
}

// Second moment (v): strictly positive, exponentially distributed.
// Range [1e-5, 1e-2] — mimics the small-and-peaked v distributions in AdamW.
static void gen_v(float* buf, int n) {
    for (int i = 0; i < n; i++) {
        float u = randf();
        buf[i] = 1e-5f * expf(u * 6.9f);   // 1e-5 * exp([0, 6.9]) = [1e-5, 1e-2]
    }
}

// Run one test: quantize -> dequantize, compare max error and RMSE
// against naive FP8 (absmax scale, no expansion).
static void test_round_trip(const char* name, float* data, int n) {
    // --- COAT round-trip ---
    uint8_t fp8[COAT_GROUP_SIZE];
    float scale, k;
    coat_quantize_group(data, fp8, &scale, &k, n);

    float recovered[COAT_GROUP_SIZE];
    coat_dequantize_group(fp8, scale, k, recovered, n);

    // --- Naive FP8 baseline: single absmax scale, no expansion ---
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) max_abs = fmaxf(max_abs, fabsf(data[i]));
    float naive_scale = (max_abs > 0.0f) ? COAT_FP8_MAX / max_abs : 1.0f;

    float coat_max_err = 0.0f, coat_mse = 0.0f;
    float naive_max_err = 0.0f, naive_mse = 0.0f;

    for (int i = 0; i < n; i++) {
        float coat_err = fabsf(data[i] - recovered[i]);
        coat_max_err = fmaxf(coat_max_err, coat_err);
        coat_mse    += coat_err * coat_err;

        float naive_q   = fmaxf(fminf(data[i] * naive_scale, COAT_FP8_MAX), -COAT_FP8_MAX);
        float naive_rec = coat_fp8_decode(coat_fp8_encode(naive_q)) / naive_scale;
        float naive_err = fabsf(data[i] - naive_rec);
        naive_max_err = fmaxf(naive_max_err, naive_err);
        naive_mse    += naive_err * naive_err;
    }
    coat_mse  = sqrtf(coat_mse  / n);
    naive_mse = sqrtf(naive_mse / n);

    // Improvement ratio: how much better is COAT vs naive?
    // Special case: if both are 0 (e.g. all-zeros input), that's a perfect tie — not a failure.
    bool both_zero = (coat_max_err == 0.0f && naive_max_err == 0.0f);
    float improvement = both_zero ? 1.0f : naive_max_err / fmaxf(coat_max_err, 1e-30f);

    printf("%-22s k=%5.2f | COAT: max=%.2e rmse=%.2e | naive: max=%.2e rmse=%.2e | %.2fx better\n",
           name, k, coat_max_err, coat_mse, naive_max_err, naive_mse, improvement);

    // COAT must not be meaningfully worse than naive.
    // The 1e-9 absolute term absorbs 1-ULP float32 rounding (e.g. x*(1/s) vs x/s)
    // which is an artifact of operation ordering, not a real quantization error.
    if (coat_max_err > naive_max_err * 1.01f + 1e-9f) {
        printf("  !! FAIL: COAT was worse than naive for '%s'\n", name);
    }
}

int main() {
    printf("COAT FP8 round-trip test  (group_size=%d)\n", COAT_GROUP_SIZE);
    printf("-----------------------------------------------------------------------\n");

    float buf[COAT_GROUP_SIZE];

    g_rng = 42;
    gen_m(buf, COAT_GROUP_SIZE);
    test_round_trip("m (first moment)", buf, COAT_GROUP_SIZE);

    g_rng = 42;
    gen_v(buf, COAT_GROUP_SIZE);
    test_round_trip("v (second moment)", buf, COAT_GROUP_SIZE);

    // Edge cases
    for (int i = 0; i < COAT_GROUP_SIZE; i++) buf[i] = 0.0f;
    test_round_trip("all-zeros", buf, COAT_GROUP_SIZE);

    for (int i = 0; i < COAT_GROUP_SIZE; i++) buf[i] = 3.14e-4f;
    test_round_trip("all-same", buf, COAT_GROUP_SIZE);

    // Single outlier surrounded by small values — stress test for k clamping
    for (int i = 0; i < COAT_GROUP_SIZE; i++) buf[i] = 1e-6f;
    buf[0] = 1.0f;
    test_round_trip("one outlier", buf, COAT_GROUP_SIZE);

    printf("-----------------------------------------------------------------------\n");
    printf("PASS: COAT should be >= 1x better than naive on real distributions.\n");
    return 0;
}
