/*
Convert a GPT-2 checkpoint (.bin) into an FP8 weights-only checkpoint.
Usage: ./convert_fp8_checkpoint input.bin output_fp8.bin
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "llmc/utils.h"

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

enum CheckpointVersion {
    MODEL_VERSION_FP32 = 3,
    MODEL_VERSION_BF16 = 5,
    MODEL_VERSION_FP8_E4M3 = 6,
};

constexpr const int NUM_PARAMETER_TENSORS = 16;
constexpr const float FP8_E4M3_MAX = 448.0f;

bool tensor_is_fp8_quantized(int tensor_idx) {
    return tensor_idx == 4 || tensor_idx == 6 || tensor_idx == 10 || tensor_idx == 12;
}

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;
    param_sizes[1] = maxT * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3 * C) * C;
    param_sizes[5] = L * (3 * C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;
}

float fp8_quant_scale(const float* values, size_t count) {
    float amax = 0.0f;
    for (size_t i = 0; i < count; i++) {
        amax = fmaxf(amax, fabsf(values[i]));
    }
    return amax == 0.0f ? 1.0f : amax / FP8_E4M3_MAX;
}

uint8_t float_to_fp8_e4m3(float value) {
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(value);
    uint8_t raw;
    memcpy(&raw, &fp8, sizeof(raw));
    return raw;
}

void write_tensor_as_bf16(FILE* out, const float* values, size_t count) {
    __nv_bfloat16* buffer = (__nv_bfloat16*)mallocCheck(count * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < count; i++) {
        buffer[i] = (__nv_bfloat16)values[i];
    }
    fwriteCheck(buffer, sizeof(__nv_bfloat16), count, out);
    free(buffer);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output_fp8.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE* in = fopenCheck(argv[1], "rb");
    int header[256];
    freadCheck(header, sizeof(int), 256, in);
    if (header[0] != 20240326) {
        fprintf(stderr, "Bad checkpoint magic.\n");
        return EXIT_FAILURE;
    }
    int version = header[1];
    if (!(version == MODEL_VERSION_FP32 || version == MODEL_VERSION_BF16)) {
        fprintf(stderr, "Input checkpoint must be FP32 or BF16.\n");
        return EXIT_FAILURE;
    }

    GPT2Config config;
    config.max_seq_len = header[2];
    config.vocab_size = header[3];
    config.num_layers = header[4];
    config.num_heads = header[5];
    config.channels = header[6];
    config.padded_vocab_size = header[7];

    size_t param_sizes[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(param_sizes, config);

    FILE* out = fopenCheck(argv[2], "wb");
    int out_header[256];
    memset(out_header, 0, sizeof(out_header));
    out_header[0] = 20240326;
    out_header[1] = MODEL_VERSION_FP8_E4M3;
    out_header[2] = config.max_seq_len;
    out_header[3] = config.vocab_size;
    out_header[4] = config.num_layers;
    out_header[5] = config.num_heads;
    out_header[6] = config.channels;
    out_header[7] = config.padded_vocab_size;
    fwriteCheck(out_header, sizeof(int), 256, out);

    for (int tensor_idx = 0; tensor_idx < NUM_PARAMETER_TENSORS; tensor_idx++) {
        size_t count = param_sizes[tensor_idx];
        float* values = (float*)mallocCheck(count * sizeof(float));
        if (version == MODEL_VERSION_FP32) {
            freadCheck(values, sizeof(float), count, in);
        } else {
            __nv_bfloat16* bf16_values = (__nv_bfloat16*)mallocCheck(count * sizeof(__nv_bfloat16));
            freadCheck(bf16_values, sizeof(__nv_bfloat16), count, in);
            for (size_t i = 0; i < count; i++) {
                values[i] = (float)bf16_values[i];
            }
            free(bf16_values);
        }

        if (!tensor_is_fp8_quantized(tensor_idx)) {
            write_tensor_as_bf16(out, values, count);
        } else {
            int num_layers = config.num_layers;
            size_t layer_size = count / num_layers;
            float* scales = (float*)mallocCheck(num_layers * sizeof(float));
            uint8_t* qvalues = (uint8_t*)mallocCheck(count * sizeof(uint8_t));
            for (int layer = 0; layer < num_layers; layer++) {
                size_t offset = layer * layer_size;
                float scale = fp8_quant_scale(values + offset, layer_size);
                scales[layer] = scale;
                for (size_t i = 0; i < layer_size; i++) {
                    float normalized = values[offset + i] / scale;
                    qvalues[offset + i] = float_to_fp8_e4m3(normalized);
                }
            }
            fwriteCheck(scales, sizeof(float), num_layers, out);
            fwriteCheck(qvalues, sizeof(uint8_t), count, out);
            free(scales);
            free(qvalues);
        }
        free(values);
    }

    fcloseCheck(in);
    fcloseCheck(out);
    printf("Wrote FP8 checkpoint to %s\n", argv[2]);
    return 0;
}
