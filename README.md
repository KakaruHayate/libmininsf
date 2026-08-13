# libmininsf
> **Languages:** [English](README.md) | [中文](README_CN.md)

A small C implementation of the **MiniNSF source generator** used by OpenVPI DiffSinger's NSF-HiFiGAN vocoder export path.

The original NSF idea comes from the Neural Source-Filter work by the Yamagishi laboratory

- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

MiniNSF here refers to the simplified `fastsinegen(f0)` source path used by OpenVPI DiffSinger

- https://github.com/openvpi/DiffSinger/blob/main/modules/nsf_hifigan/models.py#L254

For the 44.1 kHz / hop 512 / 128-bin DiffSinger vocoder, the default MiniNSF configuration is

- `source_sample_rate = 5512.5`
- `upsample = 64`
- output layout

## Features

- C99 API，无运行时依赖 / C99 API with no required runtime dependencies
- **Exact path**
- **Fast path**
- 可选 OpenMP 加速（较大输入）/ Optional OpenMP acceleration for larger inputs
- 可选 1×1 source 卷积辅助 / Optional 1x1 source convolution helper
- CMake 支持 Windows / Linux / macOS

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Build a static library

```sh
cmake -S . -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-static --config Release
```

Disable OpenMP

```sh
cmake -S . -B build -DMININSF_USE_OPENMP=OFF
```

Enable native CPU code generation for local benchmarking

```sh
cmake -S . -B build -DMININSF_NATIVE=ON
```

> `MININSF_NATIVE=ON` 不推荐用于可分发二进制（可能生成旧 CPU 不支持的指令）。

## C API

```c
#include "mininsf/mininsf.h"

MiniNsfConfig config;
mininsf_default_config(&config);

float f0[2] = {100.0f, 600.0f};
float source[2 * 64];

mininsf_fastsinegen_fast_f32(f0, 1, 2, &config, source);
```

Available functions

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

`f0` 为连续 `[batch, n_frames]` float32 Hz；`output` 为连续 `[batch, 1, n_frames * upsample]` float32。

## Examples

```sh
cmake --build build --config Release --target write_source_c      # C
cmake --build build --config Release --target write_source_cpp    # C++
dotnet run --project examples/csharp/MininsfExample.csproj        # C# / .NET P/Invoke
```

C# 示例需让 .NET 运行时找到原生库，或设置平台库搜索路径：
- Windows：`mininsf.dll` 放在可执行文件旁，或加入 `PATH`
- Linux：`libmininsf.so` 放在可执行文件旁，或加入 `LD_LIBRARY_PATH`
- macOS：`libmininsf.dylib` 放在可执行文件旁，或加入 `DYLD_LIBRARY_PATH`

## Accuracy and performance

The exact path is intended to closely match the CPU/NumPy semantics of DiffSinger's `fastsinegen`: float32 phase accumulation, `fmod`, and sine source generation without implementation-specific CUDA prefix-scan adjustment. The fast path keeps the same phase semantics and trades a small sine approximation error for lower latency. In local tests against the DiffSinger ONNX CPU output, the fast path stayed around `3e-6` max absolute error for common vocal f0 ranges.

精确路径面向 DiffSinger `fastsinegen` 的 CPU/NumPy 语义（float32 相位累积、`fmod`、正弦源生成，不含 CUDA 专属前缀扫描修正）。快速路径保持相同相位语义，以少量正弦近似误差换取更低延迟；本地对 DiffSinger ONNX CPU 输出的测试中，常见人声 f0 区间最大绝对误差约 `3e-6`。

OpenMP acceleration is only used by the fast path for larger inputs. Phase offsets are computed sequentially, then frames are generated in parallel with SIMD hints — this preserves frame-to-frame phase dependency while exposing most of the sample generation work to the compiler and runtime.

OpenMP 加速仅用于快速路径的大输入：相位偏移顺序计算，帧级生成并行化（带 SIMD 提示），既保留帧间相位依赖，又把大部分采样生成工作交给编译器与运行时。

## CUDA alignment notes

DiffSinger vocoders are usually trained with the Torch CUDA implementation of `Generator.fastsinegen(f0)`. For the official `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` checkpoint, Torch CUDA is therefore the training-side baseline.

DiffSinger vocoder 通常以 `Generator.fastsinegen(f0)` 的 Torch CUDA 实现训练；对官方 `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` checkpoint，Torch CUDA 即训练侧基线。

We tested a CPU-side phase adjustment that nudges the accumulated phase to better match Torch CUDA's `rad2.cumsum(dim=1).fmod(1.0)` output. This helped one non-transposed validation segment, but it did not generalize to pitch-shifted f0. On a 20 s vocal segment

| path
| --- | ---: | ---: |
| ONNX DML full vocoder | 68.33 dB | 77.53 dB |
| libmininsf initial CPU/NumPy phase path + external DML generator | 62.90 dB | 56.84 dB |
| experimental CUDA-adjusted libmininsf + external DML generator | 68.44 dB | 62.12 dB |

The result indicates that the CUDA adjustment is fitting an implementation detail of Torch/CUB prefix scan for one f0 distribution, not a stable MiniNSF semantic. The adjustment is therefore **not used on `main`**; the experimental implementation is preserved on the `cuda-align-experimental` branch for reference and future investigation.

结果表明该 CUDA 修正只是在拟合 Torch/CUB 前缀扫描对某一 f0 分布的实现细节，并非稳定的 MiniNSF 语义，因此 **`main` 分支不使用该修正**；实验实现保留在 `cuda-align-experimental` 分支供参考与后续研究。

## Recommended path

- Use the `main` implementation as the stable MiniNSF source generator — it is the original CPU/NumPy-aligned path and the recommended semantic target for new exports or future vocoder training.
- For existing Torch-CUDA-trained checkpoints, keep the original full ONNX vocoder path when pitch control changes f0 at inference time. If a deployment splits MiniNSF into `libmininsf + external generator`, validate that exact checkpoint and pitch-control range before enabling it in production.
- For future training, the cleanest long-term option is to train or fine-tune the generator with this stable MiniNSF source path from the start — this removes the dependency on CUDA prefix-scan rounding and lets CPU / DML / other platforms share the same source semantics.

## Notes

This project implements only the MiniNSF sine source used by DiffSinger's exported vocoder path. It is not a full implementation of the original NSF model family.
