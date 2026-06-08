#include "mininsf/mininsf.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef MININSF_PI_F
#define MININSF_PI_F 3.14159265358979323846f
#endif

static float mininsf_sin_2pi_approx(float phase) {
    float x;
    float x2;
    float p;
    float sign = 1.0f;

    phase = phase - floorf(phase + 0.5f);
    x = (2.0f * MININSF_PI_F) * phase;
    if (x < 0.0f) {
        sign = -1.0f;
        x = -x;
    }
    if (x > (0.5f * MININSF_PI_F)) {
        x = MININSF_PI_F - x;
    }

    x2 = x * x;
    p = -2.3889859e-08f;
    p = p * x2 + 2.7525562e-06f;
    p = p * x2 - 1.9840897e-04f;
    p = p * x2 + 8.3333315e-03f;
    p = p * x2 - 1.6666666e-01f;
    p = p * x2 + 1.0f;
    return sign * x * p;
}

void mininsf_default_config(MiniNsfConfig *config) {
    if (config == NULL) {
        return;
    }
    config->source_sample_rate = 5512.5f;
    config->upsample = 64;
}

static int mininsf_fastsinegen_impl_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output,
    int fast_sin) {
    MiniNsfConfig local_config;
    int64_t b;

    if (f0 == NULL || output == NULL || batch <= 0 || n_frames <= 0) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }
    if (config == NULL) {
        mininsf_default_config(&local_config);
        config = &local_config;
    }
    if (config->source_sample_rate <= 0.0f || config->upsample <= 0) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }

    for (b = 0; b < batch; ++b) {
        const float *f0_b = f0 + b * n_frames;
        float *out_b = output + b * n_frames * (int64_t)config->upsample;
        float phase_sum = 0.0f;
        int64_t t;

        for (t = 0; t < n_frames; ++t) {
            const float s0 = f0_b[t] / config->source_sample_rate;
            const float next_s0 = (t + 1 < n_frames)
                ? f0_b[t + 1] / config->source_sample_rate
                : s0;
            const float ds0 = (t + 1 < n_frames) ? (next_s0 - s0) : 0.0f;
            const float offset = fmodf(phase_sum, 1.0f);
            int32_t k;

            for (k = 0; k < config->upsample; ++k) {
                const float n = (float)(k + 1);
                const float rad =
                    s0 * n +
                    0.5f * ds0 * n * (n - 1.0f) / (float)config->upsample +
                    offset;
                out_b[t * (int64_t)config->upsample + k] = fast_sin
                    ? mininsf_sin_2pi_approx(rad)
                    : sinf((2.0f * MININSF_PI_F) * rad);
            }

            {
                const float n = (float)config->upsample;
                const float rad_last =
                    s0 * n +
                    0.5f * ds0 * n * (n - 1.0f) / (float)config->upsample;
                const float rad2 = fmodf(rad_last + 0.5f, 1.0f) - 0.5f;
                phase_sum += rad2;
            }
        }
    }

    return MININSF_OK;
}

#if defined(_OPENMP)
static int mininsf_fastsinegen_fast_two_pass_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output) {
    MiniNsfConfig local_config;
    float *phase_offsets;
    int64_t b;
    int64_t total_frames;

    if (f0 == NULL || output == NULL || batch <= 0 || n_frames <= 0) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }
    if (config == NULL) {
        mininsf_default_config(&local_config);
        config = &local_config;
    }
    if (config->source_sample_rate <= 0.0f || config->upsample <= 0) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }

    total_frames = batch * n_frames;
    phase_offsets = (float *)malloc((size_t)total_frames * sizeof(float));
    if (phase_offsets == NULL) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }

    for (b = 0; b < batch; ++b) {
        const float *f0_b = f0 + b * n_frames;
        float phase_sum = 0.0f;
        int64_t t;
        for (t = 0; t < n_frames; ++t) {
            const float s0 = f0_b[t] / config->source_sample_rate;
            const float next_s0 = (t + 1 < n_frames)
                ? f0_b[t + 1] / config->source_sample_rate
                : s0;
            const float ds0 = (t + 1 < n_frames) ? (next_s0 - s0) : 0.0f;
            const float n = (float)config->upsample;
            const float rad_last =
                s0 * n +
                0.5f * ds0 * n * (n - 1.0f) / (float)config->upsample;
            const float rad2 = fmodf(rad_last + 0.5f, 1.0f) - 0.5f;

            phase_offsets[b * n_frames + t] = fmodf(phase_sum, 1.0f);
            phase_sum += rad2;
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(total_frames >= 256)
#endif
    for (int64_t bt = 0; bt < total_frames; ++bt) {
        const int64_t b_idx = bt / n_frames;
        const int64_t t = bt - b_idx * n_frames;
        const float *f0_b = f0 + b_idx * n_frames;
        float *out_b = output + b_idx * n_frames * (int64_t)config->upsample;
        const float s0 = f0_b[t] / config->source_sample_rate;
        const float next_s0 = (t + 1 < n_frames)
            ? f0_b[t + 1] / config->source_sample_rate
            : s0;
        const float ds0 = (t + 1 < n_frames) ? (next_s0 - s0) : 0.0f;
        const float offset = phase_offsets[bt];
        const float inv_upsample = 1.0f / (float)config->upsample;
        int32_t k;

#if defined(_OPENMP)
#pragma omp simd
#endif
        for (k = 0; k < config->upsample; ++k) {
            const float n = (float)(k + 1);
            const float rad =
                s0 * n +
                0.5f * ds0 * n * (n - 1.0f) * inv_upsample +
                offset;
            out_b[t * (int64_t)config->upsample + k] = mininsf_sin_2pi_approx(rad);
        }
    }

    free(phase_offsets);
    return MININSF_OK;
}
#endif

int mininsf_fastsinegen_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output) {
    return mininsf_fastsinegen_impl_f32(f0, batch, n_frames, config, output, 0);
}

int mininsf_fastsinegen_fast_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output) {
#if defined(_OPENMP)
    if (batch * n_frames >= 256) {
        return mininsf_fastsinegen_fast_two_pass_f32(f0, batch, n_frames, config, output);
    }
#endif
    return mininsf_fastsinegen_impl_f32(f0, batch, n_frames, config, output, 1);
}

int mininsf_source_conv1x1_f32(
    const float *sine,
    int64_t batch,
    int64_t n_samples,
    const float *weight,
    const float *bias,
    int64_t channels,
    float *output) {
    int64_t b;

    if (sine == NULL || weight == NULL || bias == NULL || output == NULL ||
        batch <= 0 || n_samples <= 0 || channels <= 0) {
        return MININSF_ERROR_INVALID_ARGUMENT;
    }

    for (b = 0; b < batch; ++b) {
        const float *sine_b = sine + b * n_samples;
        float *out_b = output + b * channels * n_samples;
        int64_t c;
        for (c = 0; c < channels; ++c) {
            const float w = weight[c];
            const float v_bias = bias[c];
            int64_t i;
            float *out_c = out_b + c * n_samples;
            for (i = 0; i < n_samples; ++i) {
                out_c[i] = sine_b[i] * w + v_bias;
            }
        }
    }

    return MININSF_OK;
}
