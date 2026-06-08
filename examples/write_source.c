#include "mininsf/mininsf.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    MiniNsfConfig config;
    const int n_frames = 4;
    float f0[4] = {220.0f, 240.0f, 260.0f, 280.0f};
    float *source;
    int i;

    mininsf_default_config(&config);
    source = (float *)malloc((size_t)n_frames * (size_t)config.upsample * sizeof(float));
    if (source == NULL) {
        return 1;
    }

    if (mininsf_fastsinegen_fast_f32(f0, 1, n_frames, &config, source) != MININSF_OK) {
        free(source);
        return 1;
    }

    for (i = 0; i < 16; ++i) {
        printf("%d %.9f\n", i, source[i]);
    }

    free(source);
    return 0;
}
