#include "mininsf/mininsf.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    MiniNsfConfig config;
    mininsf_default_config(&config);

    const std::vector<float> f0 = {220.0f, 240.0f, 260.0f, 280.0f};
    std::vector<float> source(f0.size() * static_cast<std::size_t>(config.upsample));

    const int ret = mininsf_fastsinegen_fast_f32(
        f0.data(),
        1,
        static_cast<int64_t>(f0.size()),
        &config,
        source.data()
    );
    if (ret != MININSF_OK) {
        std::cerr << "mininsf_fastsinegen_fast_f32 failed: " << ret << "\n";
        return EXIT_FAILURE;
    }

    for (std::size_t i = 0; i < 16 && i < source.size(); ++i) {
        std::cout << i << " " << source[i] << "\n";
    }
    return EXIT_SUCCESS;
}
