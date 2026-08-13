# libmininsf

> **语言：** [English](README.md) | [中文](README_CN.md)

**MiniNSF 正弦源生成器的小型 C 实现** —— 用于 OpenVPI DiffSinger 的 NSF-HiFiGAN vocoder 导出路径。

原始 NSF 概念源自山岸实验室（Yamagishi laboratory）的 Neural Source-Filter 工作：

- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

此处 MiniNSF 指 OpenVPI DiffSinger 使用的简化 `fastsinegen(f0)` 源路径：

- https://github.com/openvpi/DiffSinger/blob/main/modules/nsf_hifigan/models.py#L254

对 44.1 kHz / hop 512 / 128-bin DiffSinger vocoder，默认 MiniNSF 配置为：

- `source_sample_rate = 5512.5`
- `upsample = 64`
- 输出布局：`[batch, 1, n_frames * upsample]`

## 特性

- C99 API，无运行时依赖
- **精确路径**：`sinf` 实现，面向参考级数值对齐
- **快速路径**：相位归约 + 多项式正弦近似
- 可选 OpenMP 加速（较大输入）
- 可选 1×1 source 卷积辅助
- CMake 支持 Windows / Linux / macOS

## 构建

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

构建静态库：

```sh
cmake -S . -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-static --config Release
```

关闭 OpenMP：

```sh
cmake -S . -B build -DMININSF_USE_OPENMP=OFF
```

本地基准测试启用原生 CPU 指令：

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

可用函数：

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

## 示例

```sh
cmake --build build --config Release --target write_source_c      # C
cmake --build build --config Release --target write_source_cpp    # C++
dotnet run --project examples/csharp/MininsfExample.csproj        # C# / .NET P/Invoke
```

C# 示例需让 .NET 运行时找到原生库，或设置平台库搜索路径：
- Windows：`mininsf.dll` 放在可执行文件旁，或加入 `PATH`
- Linux：`libmininsf.so` 放在可执行文件旁，或加入 `LD_LIBRARY_PATH`
- macOS：`libmininsf.dylib` 放在可执行文件旁，或加入 `DYLD_LIBRARY_PATH`

## 精度与性能

精确路径面向 DiffSinger `fastsinegen` 的 CPU/NumPy 语义（float32 相位累积、`fmod`、正弦源生成，不含 CUDA 专属前缀扫描修正）。快速路径保持相同相位语义，以少量正弦近似误差换取更低延迟；本地对 DiffSinger ONNX CPU 输出的测试中，常见人声 f0 区间最大绝对误差约 `3e-6`。

OpenMP 加速仅用于快速路径的大输入：相位偏移顺序计算，帧级生成并行化（带 SIMD 提示），既保留帧间相位依赖，又把大部分采样生成工作交给编译器与运行时。

## CUDA 对齐说明

DiffSinger vocoder 通常以 `Generator.fastsinegen(f0)` 的 Torch CUDA 实现训练；对官方 `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` checkpoint，Torch CUDA 即训练侧基线。

我们测试过一种 CPU 侧相位修正（使累积相位更贴近 Torch CUDA 的 `rad2.cumsum(dim=1).fmod(1.0)`），对单个非转置验证片段有效，但无法泛化到变调 f0。20 s 人声片段实测：

| 路径 | 原始 f0 wav SNR vs Torch CUDA | +4 半音 wav SNR vs Torch CUDA |
| --- | ---: | ---: |
| ONNX DML 完整 vocoder | 68.33 dB | 77.53 dB |
| libmininsf 初始 CPU/NumPy 相位路径 + 外部 DML generator | 62.90 dB | 56.84 dB |
| 实验性 CUDA 修正 libmininsf + 外部 DML generator | 68.44 dB | 62.12 dB |

结果表明该 CUDA 修正只是在拟合 Torch/CUB 前缀扫描对某一 f0 分布的实现细节，并非稳定的 MiniNSF 语义，因此 **`main` 分支不使用该修正**；实验实现保留在 `cuda-align-experimental` 分支供参考与后续研究。

## 推荐路径

- `main` 为稳定实现（原始 CPU/NumPy 对齐路径），是新增导出或未来 vocoder 训练的推荐语义目标。
- 对已有 Torch-CUDA 训练 checkpoint，推理期变调时仍走完整 ONNX vocoder 路径；若部署拆分 `libmininsf + external generator`，上线前须在目标 checkpoint 与变调范围内验证。
- 未来训练的最优长期方案是从一开始就用本稳定 MiniNSF 源路径训练/微调，消除对 CUDA 前缀扫描舍入的依赖，使 CPU / DML 等平台共享同一源语义。

## 说明

本项目仅实现 DiffSinger 导出 vocoder 路径使用的 MiniNSF 正弦源，不是原始 NSF 模型族的完整实现。

## 许可

| 组件 | 许可 |
|---|---|
| 本仓库（代码） | MIT（见 `LICENSE`） |
| OpenVPI DiffSinger（参考实现） | MIT |
| 山岸实验室 NSF 脚本（原始概念，仅作参考，未 vendor） | 见上游仓库 |

本仓库不分发任何二进制或模型权重。
