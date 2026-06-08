# libmininsf

`libmininsf` is a small C implementation of the MiniNSF source generator used by
OpenVPI DiffSinger's NSF-HiFiGAN vocoder export path.

The original NSF idea comes from the Neural Source-Filter work by the
Yamagishi laboratory:

- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

MiniNSF here refers to the simplified `fastsinegen(f0)` source path used by
OpenVPI DiffSinger:

- https://github.com/openvpi/DiffSinger/blob/main/modules/nsf_hifigan/models.py#L254

For the 44.1 kHz / hop 512 / 128-bin DiffSinger vocoder, the default MiniNSF
configuration is:

- `source_sample_rate = 5512.5`
- `upsample = 64`
- output layout: `[batch, 1, n_frames * upsample]`

## Features

- C99 API with no required runtime dependencies.
- Exact path using `sinf`, intended for reference-quality matching.
- Fast path using phase reduction and a polynomial sine approximation.
- Optional OpenMP acceleration for larger inputs.
- Optional 1x1 source convolution helper.
- CMake builds on Windows, Linux, and macOS.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Build a static library:

```sh
cmake -S . -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-static --config Release
```

Disable OpenMP:

```sh
cmake -S . -B build -DMININSF_USE_OPENMP=OFF
```

Enable native CPU code generation for local benchmarking:

```sh
cmake -S . -B build -DMININSF_NATIVE=ON
```

`MININSF_NATIVE=ON` is not recommended for redistributable binaries because it
may emit instructions unavailable on older CPUs.

## C API

```c
#include "mininsf/mininsf.h"

MiniNsfConfig config;
mininsf_default_config(&config);

float f0[2] = {100.0f, 600.0f};
float source[2 * 64];

mininsf_fastsinegen_fast_f32(f0, 1, 2, &config, source);
```

Available functions:

```c
void mininsf_default_config(MiniNsfConfig *config);

int mininsf_fastsinegen_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output);

int mininsf_fastsinegen_fast_f32(
    const float *f0,
    int64_t batch,
    int64_t n_frames,
    const MiniNsfConfig *config,
    float *output);

int mininsf_source_conv1x1_f32(
    const float *sine,
    int64_t batch,
    int64_t n_samples,
    const float *weight,
    const float *bias,
    int64_t channels,
    float *output);
```

`f0` is contiguous `[batch, n_frames]` float32 Hz. `output` is contiguous
`[batch, 1, n_frames * upsample]` float32.

## Examples

C:

```sh
cmake --build build --config Release --target write_source_c
```

C++:

```sh
cmake --build build --config Release --target write_source_cpp
```

C# / .NET P/Invoke:

```sh
dotnet run --project examples/csharp/MininsfExample.csproj
```

For the C# example, place the native library where the .NET runtime can find it,
or set the platform library search path:

- Windows: `mininsf.dll` next to the executable, or on `PATH`
- Linux: `libmininsf.so` next to the executable, or on `LD_LIBRARY_PATH`
- macOS: `libmininsf.dylib` next to the executable, or on `DYLD_LIBRARY_PATH`

## Accuracy And Performance

The exact path is intended to closely match an ONNX export of DiffSinger's
`fastsinegen`. The fast path trades a small sine approximation error for lower
latency. In local tests against the DiffSinger ONNX CPU output, the fast path
stayed around `3e-6` max absolute error for common vocal f0 ranges.

OpenMP acceleration is only used by the fast path for larger inputs. The phase
offsets are computed sequentially, then frames are generated in parallel with
SIMD hints. This preserves the frame-to-frame phase dependency while exposing
most of the sample generation work to the compiler and runtime.

## Notes

This project implements only the MiniNSF sine source used by DiffSinger's
exported vocoder path. It is not a full implementation of the original NSF
model family.
