#include "mininsf/mininsf.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float actual, float expected, float eps, const char *label) {
    const float diff = fabsf(actual - expected);
    if (diff > eps) {
        fprintf(stderr, "%s: actual=%g expected=%g diff=%g eps=%g\n",
                label, actual, expected, diff, eps);
        return 0;
    }
    return 1;
}

int main(void) {
    MiniNsfConfig config;
    float f0[2] = {100.0f, 600.0f};
    float exact[128];
    float fast[128];
    const int indexes[] = {0, 1, 2, 3, 4, 5, 6, 7, 63, 64, 65, 66, 127};
    const float refs[] = {
        0.113734052f, 0.234657422f, 0.360362321f, 0.487610906f,
        0.612287700f, 0.729407370f, 0.833197117f, 0.917272985f,
        0.113733403f, 0.715865612f, 0.996037781f, 0.828240991f,
        -0.0995673537f
    };
    int i;

    mininsf_default_config(&config);
    if (config.source_sample_rate != 5512.5f || config.upsample != 64) {
        fprintf(stderr, "unexpected default config\n");
        return 1;
    }

    if (mininsf_fastsinegen_f32(f0, 1, 2, &config, exact) != MININSF_OK) {
        fprintf(stderr, "exact fastsinegen failed\n");
        return 1;
    }
    if (mininsf_fastsinegen_fast_f32(f0, 1, 2, &config, fast) != MININSF_OK) {
        fprintf(stderr, "fast fastsinegen failed\n");
        return 1;
    }

    for (i = 0; i < (int)(sizeof(indexes) / sizeof(indexes[0])); ++i) {
        char label[64];
        snprintf(label, sizeof(label), "exact[%d]", indexes[i]);
        if (!near_f32(exact[indexes[i]], refs[i], 1e-5f, label)) {
            return 1;
        }
        snprintf(label, sizeof(label), "fast[%d]", indexes[i]);
        if (!near_f32(fast[indexes[i]], refs[i], 1e-4f, label)) {
            return 1;
        }
    }

    {
        float weight[2] = {2.0f, -1.0f};
        float bias[2] = {0.5f, 0.25f};
        float conv[4];
        if (mininsf_source_conv1x1_f32(exact, 1, 2, weight, bias, 2, conv) != MININSF_OK) {
            fprintf(stderr, "source conv failed\n");
            return 1;
        }
        if (!near_f32(conv[0], exact[0] * 2.0f + 0.5f, 1e-6f, "conv[0]")) {
            return 1;
        }
        if (!near_f32(conv[2], -exact[0] + 0.25f, 1e-6f, "conv[2]")) {
            return 1;
        }
    }

    return 0;
}
