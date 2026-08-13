# libmininsf

> **Languages:** [English](README.md) | [中文](README_CN.md)

A small C implementation of the **MiniNSF source generator** used by OpenVPI DiffSinger's NSF-HiFiGAN vocoder export path.

The original NSF idea comes from the Neural Source-Filter work by the Yamagishi laboratory:

- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

MiniNSF here refers to the simplified `fastsinegen(f0)` source path used by OpenVPI DiffSinger:

- https://github.com/openvpi/DiffSinger/blob/main/modules/nsf_hifigan/models.py#L254

For the 44.1 kHz / hop 512 / 128-bin DiffSinger vocoder, the default MiniNSF configuration is:

- `source_sample_rate = 5512.5`
- `upsample = 64`
- output layout: `[batch, 1, n_frames * upsample]`

## Features

- C99 API, no required runtime dependencies
- **Exact path**: `sinf`-based implementation for reference-level numeric alignment
- **Fast path**: phase reduction + polynomial sine approximation
- Optional OpenMP acceleration (larger inputs)
- Optional 1x1 source convolution helper
- CMake support on Windows / Linux / macOS

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

> `MININSF_NATIVE=ON` is not recommended for distributable binaries (it may
> emit instructions older CPUs do not support).

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
    const float *f0, int64_t batch, int64_t n_frames,
    const MiniNsfConfig *config, float *output);

int mininsf_fastsinegen_fast_f32(
    const float *f0, int64_t batch, int64_t n_frames,
    const MiniNsfConfig *config, float *output);

int mininsf_source_conv1x1_f32(
    const float *sine, int64_t batch, int64_t n_samples,
    const float *weight, const float *bias, int64_t channels,
    float *output);
```

`f0` is a contiguous `[batch, n_frames]` float32 array in Hz; `output` is a
contiguous `[batch, 1, n_frames * upsample]` float32 array.

## Examples

```sh
cmake --build build --config Release --target write_source_c      # C
cmake --build build --config Release --target write_source_cpp    # C++
dotnet run --project examples/csharp/MininsfExample.csproj        # C# / .NET P/Invoke
```

For the C# example, the native library must be findable by the .NET runtime:

- Windows: place `mininsf.dll` next to the executable or add it to `PATH`
- Linux: place `libmininsf.so` next to the executable or add it to `LD_LIBRARY_PATH`
- macOS: place `libmininsf.dylib` next to the executable or add it to `DYLD_LIBRARY_PATH`

## Accuracy and performance

The exact path is intended to closely match the CPU/NumPy semantics of DiffSinger's `fastsinegen`: float32 phase accumulation, `fmod`, and sine source generation without implementation-specific CUDA prefix-scan adjustment. The fast path keeps the same phase semantics and trades a small sine approximation error for lower latency. In local tests against the DiffSinger ONNX CPU output, the fast path stayed around `3e-6` max absolute error for common vocal f0 ranges.

OpenMP acceleration is only used by the fast path for larger inputs. Phase offsets are computed sequentially, then frames are generated in parallel with SIMD hints — this preserves frame-to-frame phase dependency while exposing most of the sample generation work to the compiler and runtime.

## CUDA alignment notes

DiffSinger vocoders are usually trained with the Torch CUDA implementation of `Generator.fastsinegen(f0)`. For the official `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` checkpoint, Torch CUDA is therefore the training-side baseline.

We tested a CPU-side phase adjustment that nudges the accumulated phase to better match Torch CUDA's `rad2.cumsum(dim=1).fmod(1.0)` output. This helped one non-transposed validation segment, but it did not generalize to pitch-shifted f0. On a 20 s vocal segment:

| path | original f0 wav SNR vs Torch CUDA | +4 semitones wav SNR vs Torch CUDA |
| --- | ---: | ---: |
| ONNX DML full vocoder | 68.33 dB | 77.53 dB |
| libmininsf initial CPU/NumPy phase path + external DML generator | 62.90 dB | 56.84 dB |
| experimental CUDA-adjusted libmininsf + external DML generator | 68.44 dB | 62.12 dB |

The result indicates that the CUDA adjustment is fitting an implementation detail of Torch/CUB prefix scan for one f0 distribution, not a stable MiniNSF semantic. The adjustment is therefore **not used on `main`**; the experimental implementation is preserved on the `cuda-align-experimental` branch for reference and future investigation.

## Recommended path

- Use the `main` implementation as the stable MiniNSF source generator — it is the original CPU/NumPy-aligned path and the recommended semantic target for new exports or future vocoder training.
- For existing Torch-CUDA-trained checkpoints, keep the original full ONNX vocoder path when pitch control changes f0 at inference time. If a deployment splits MiniNSF into `libmininsf + external generator`, validate that exact checkpoint and pitch-control range before enabling it in production.
- For future training, the cleanest long-term option is to train or fine-tune the generator with this stable MiniNSF source path from the start — this removes the dependency on CUDA prefix-scan rounding and lets CPU / DML / other platforms share the same source semantics.

## Notes

This project implements only the MiniNSF sine source used by DiffSinger's exported vocoder path. It is not a full implementation of the original NSF model family.

## License

| Component | License |
|---|---|
| This repository (code) | MIT (see `LICENSE`) |
| OpenVPI DiffSinger (reference implementation) | MIT |
| Yamagishi lab NSF scripts (original concept, reference only — not vendored) | see upstream repo |

No binaries or model weights are distributed in this repository.
