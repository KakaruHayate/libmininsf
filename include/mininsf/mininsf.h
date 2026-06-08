#ifndef MININSF_MININSF_H_
#define MININSF_MININSF_H_

#include <stdint.h>

#if defined(_WIN32)
#if defined(MININSF_BUILD_SHARED)
#define MININSF_API __declspec(dllexport)
#elif defined(MININSF_USE_SHARED)
#define MININSF_API __declspec(dllimport)
#else
#define MININSF_API
#endif
#else
#define MININSF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiniNsfConfig {
    float source_sample_rate;
    int32_t upsample;
} MiniNsfConfig;

enum {
    MININSF_OK = 0,
    MININSF_ERROR_INVALID_ARGUMENT = -1
};

MININSF_API void mininsf_default_config(MiniNsfConfig *config);

MININSF_API int mininsf_fastsinegen_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output);

MININSF_API int mininsf_fastsinegen_fast_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output);

MININSF_API int mininsf_source_conv1x1_f32(
    const float *sine,
    int64_t batch,
    int64_t n_samples,
    const float *weight,
    const float *bias,
    int64_t channels,
    float *output);

#ifdef __cplusplus
}
#endif

#endif
