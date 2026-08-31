# RTX 3090 单卡部署 Qwen3.8-27B 实战指南（llama.cpp + 推测解码 + 视觉）

> 硬件：RTX 3090 24GB / Ryzen 9 9900X（12C24T）/ 60GB RAM / Ubuntu 22.04 / CUDA 12.x
> 软件版本：llama.cpp `cc83d7b48`（b10684）

---

## 0. 成果一览

| 指标 | 数值 |
|---|---|
| 20K 上下文 prefill | ~1,200 tok/s |
| 100K 上下文 prefill | ~845 tok/s |
| 176K 长文本 prefill | ~530 tok/s |
| decode（20K 上下文） | 62~76 tok/s |
| decode（176K 上下文） | ~42 tok/s |
| 最大可用上下文 | 256K（KV q4/q4）或 200K（KV 混合精度） |
| 视觉理解 | ✅ mmproj + `--image-min-tokens 1024` |
| 空闲显存余量 | 420~830 MiB |

> 期望管理：vLLM 官方基准（DFlash2 greedy ~131 tok/s）是不同框架与精度栈的成绩；
> llama.cpp 单卡简配置跑出 55~65 tok/s 的 decode 属正常水平，不必对标。

---

## 1. 模型文件

| 文件 | 大小 | 用途 |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 15.4 GB | 主模型（推荐） |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.1 GB | DFlash2 草稿模型（block diffusion drafter） |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.37 GB | MTP 草稿（llama.cpp 内置 MTP 支持，与 DFlash2 二选一） |
| `mmproj-Q8_0.gguf` | 629 MB | 视觉投影（**别用 BF16 931MB，满配会 OOM**） |

**主模型选 Q4_K_S 而不是 Q4_K_M**（实测同配置对比）：

| 场景 | Q4_K_M (16.5GB) | Q4_K_S (15.4GB) | 差异 |
|---|---|---|---|
| 20K/30K/100K prefill | 1,200 / 1,074 / 815 | 1,209 / 1,080 / 819 | ≈持平 |
| 20K/30K/100K decode | 60.9 / 72.8 / 52.3 | 62.7 / 75.8 / 65.4 | +3~5% |
| 空闲显存 | 24,009 MiB | 23,115 MiB | **省 894 MiB** |

文件更小 → 显存带宽压力更小 → 反而更快，且省出近 1GB 余量。
（注：短输出 decode 单次波动 ±10%，看趋势别抠单点。）

---

## 2. 编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build-allquant -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DGGML_CCACHE=OFF
cmake --build build-allquant --config Release -j$(nproc)
```

要点：
- `CMAKE_CUDA_ARCHITECTURES=86`：3090 是 Ampere（CC 8.6），只编这一个架构，编译时间省一半以上
- `GGML_CUDA_FA_ALL_QUANTS=ON`：**解锁混合 K/V 量化类型的 FlashAttention 内核**（49 种类型配对 × head_dim 64/128/256）。编译多花几分钟，运行时零开销。不开这个宏就只能用同类型 KV 配对（见 §3）
- 非交互 shell 里 `nvcc` 往往不在 PATH，显式给 `-DCMAKE_CUDA_COMPILER`
- 24 核机器全程约 10 分钟

---

## 3. KV cache 量化 —— 本文最重要的坑

### 3.1 规则

llama.cpp **默认构建只编译 4 种同类型 KV 配对**的 FA 内核：
`f16/f16`、`q4_0/q4_0`、`q8_0/q8_0`、`bf16/bf16`

源码依据（`ggml/src/ggml-cuda/fattn.cu`）：

```c
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;   // K/V 类型不一致 → CUDA 判定不支持
    }
#endif
```

### 3.2 故障复盘：混合配对 + 默认构建 = 静默 CPU 回退

配置 `K=q8_0 + V=q4_0` 跑默认构建时：

```
症状：GPU 利用率 0~6%，CPU 896%，prefill 从 820 tok/s 掉到 53 tok/s
      服务不报错、不崩溃、日志无异常 —— 只是静默变慢 20 倍
根因：CUDA 没有该组合的 FA 内核 → ggml 调度器把 FLASH_ATTN_EXT 整个算子
      派给 CPU 后端（ggml-cpu 有实现），KV 每层 GPU↔CPU 拷贝
```

**排查方法**（请求进行中采样，不是请求结束后）：

```bash
( for i in $(seq 1 100); do nvidia-smi --query-gpu=utilization.gpu \
  --format=csv,noheader,nounits; sleep 0.5; done ) &   # 另开窗口
curl ... -d @long_prompt.json                            # 发长 prompt
# GPU avg < 20% + CPU 接近 (核数×100)% → 算子回退 CPU 了
```

### 3.3 三种合法组合实测（速度无差异，显存不同）

| KV 配置 | 需要的构建 | 20K prefill / decode | 空闲显存(150K ctx) |
|---|---|---|---|
| q8_0 / q8_0 | 默认 | 1,209 / 62.7 | 23,115 MiB |
| q4_0 / q4_0 | 默认 | 1,207 / 62.9 | 20,771 MiB（**省 2.3GB**） |
| K=q8_0 + V=q4_0 | ALL_QUANTS | 1,206 / 63.1 | 21,943 MiB（省 1.2GB） |
| ~~K=q8_0 + V=q4_0~~ | ~~默认~~ | ~~53（CPU 回退）~~ | — |

**显存经验值**（每千 token）：q8/q8 ≈ 31.3 MiB、q4/q4 ≈ 15.6 MiB、K8/V4 混合 ≈ 23.4 MiB，另计计算缓冲 ~7 MiB/千 token。
K 精度比 V 敏感（影响 attention logits），追求质量选混合 K=q8_0 + V=q4_0。

---

## 4. 推测解码：MTP vs DFlash2

两种方案在 llama.cpp 里都开箱即用（温度 0、greedy 接受率 65~88%）：

- **MTP**：内置草稿，`--spec-type draft-mtp`，无需 `--model-draft`，用独立 MTP 头 GGUF
- **DFlash2**：block diffusion drafter（block_size=8, n_extract=5），`--spec-type draft-dflash` + `--model-draft` 指向草稿 GGUF

数据（Q4_K_M 主模型 + 150K ctx，短输出 20 token 专测 prefill）：

| 场景 | MTP n=4 | DF2 n=3 | DF2 n=4 |
|---|---|---|---|
| prefill 20K / 30K / 100K | 1,249 / 1,086 / 813 | 1,199 / 1,068 / 813 | 1,200 / 1,074 / 815 |
| decode 20K / 30K / 100K | 72.2 / 69.2 / 48.6 | 56.0 / 66.1 / 55.9 | 60.9 / 72.8 / 52.3 |
| 接受率 20K / 30K / 100K | 70/70/68% | 72/72/87% | 70/70/68% |
| 长输出(1000 tok) decode | 49.8 | — | 55.0 |
| agent 短 prompt decode | 83.9 | — | 78.8 |

**n-max 调参结论**：
- MTP n=4 最优；降到 n=3 短输出反而变慢 5~8%
- DF2 n=7 + 150K ctx 会 OOM（多留的草稿 KV 放不下），n=4 是显存约束下的甜点
- 选型：两者综合性能相差 <3%。MTP 短输出/agent 场景略快，DFlash2 长输出略快且省一份显存峰值

---

## 5. 最终启动脚本

```bash
#!/bin/bash
# start_llama_dflash2.sh —— Q4_K_S + DFlash2 + 视觉 + 200K ctx + 混合 KV
~/llama.cpp/build-allquant/bin/llama-server \
  -m  ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --model-draft ~/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --mmproj ~/models/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf \
  --port 8000 --host 0.0.0.0 \
  --ctx-size 204800 \
  --n-gpu-layers 99 --parallel 1 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --spec-type draft-dflash --spec-draft-n-max 4 \
  --flash-attn on \
  --image-min-tokens 1024 \
  --threads 12 --reasoning-effort low
```

**显存预算**（200K ctx，实测空闲 23,601 MiB）：

| 项 | 占用 |
|---|---|
| 主模型 Q4_K_S | ~14,650 MiB |
| DFlash2 草稿 | ~1,050 MiB |
| mmproj-Q8_0 | ~600 MiB |
| KV cache（200K，K8/V4） | ~4,680 MiB |
| CUDA 上下文 + 计算缓冲/图 | ~1,600 MiB |
| **合计** | **~23.6 GB（余 ~0.5 GB）** |

> 不要用 `-hfd` 指向本地 GGUF 文件（它期望 HF repo id）；
> `--batch-size` 保持 2048 级别，配推测解码时过小的 batch 会触发
> `GGML_ASSERT(n_ubatch > n_keep_tail)` 崩溃。

> 安全提示：`--host 0.0.0.0` 会监听所有网卡。若机器暴露在公网/共享网络，
> 建议加 `--api-key <自定义密钥>` 开启鉴权，或用防火墙限制来源地址。

---

## 6. 上下文长度扩展（24GB 的边界）

| ctx-size | KV 配置 | 结果 |
|---|---|---|
| 150,000 | 任意 | 从容，余 1~3.4GB |
| 262,144 (256K) | q4/q4 | ✅ 余 734 MiB |
| 262,144 / 256,000 | K8/V4 | ❌ OOM（差 ~500 MiB） |
| 245,760 | K8/V4 | ❌ OOM |
| 229,376 (224K) | K8/V4 | ⚠️ 能启动但请求运行时失败（余仅 30 MiB） |
| **204,800 (200K)** | **K8/V4** | ✅ **稳定，余 ~420 MiB（本文推荐配置）** |

上下文越长 decode 越慢（每步都要对全部 KV 做 attention）：

```
20K ctx → 62.6 tok/s    100K → 53.7    176K → 42.1
```

**前缀缓存**：同一服务连续请求若共享 prompt 前缀，llama.cpp 只实际计算增量部分
（如 30K 请求只 eval 9,168 tok）。做基准测试时注意这会"美化"后续请求的 prefill 数字，
测纯 prefill 需重启服务或更换前缀。

---

## 7. 视觉能力

- `--mmproj` 加载 Q8_0 投影权重，chat 请求 `messages` 里直接放 image（base64 / URL）
- `--image-min-tokens 1024`：保证高分辨率图不被压成几十个 token（代价是图像占用更多上下文）
- BF16 mmproj（931MB）在满配下必 OOM，用 Q8_0（629MB）

---

## 8. 踩坑清单

1. **KV K/V 类型不一致 + 默认构建 → FA 算子静默回退 CPU**（§3，最大坑，无报错只有 20 倍减速）
2. `pkill -f llama-server` 会误杀命令行里含该字符串的父 shell（比如内联执行的远程脚本）——用脚本文件方式执行，或精确匹配二进制路径
3. llama.cpp 的 `--fit`/自动 fit 与显式参数混用会打架；显式指定了参数时以显式为准
4. 短 prompt 的 prefill tok/s 只有 100~300 是正常现象（流水线没填满），别拿它评估长文本性能
5. 20-token 短输出的 decode 数字波动 ±10%，跨配置对比要看多场景趋势
6. mmproj-BF16 + 满配主模型 = OOM；mmproj 一律 Q8_0
7. vLLM（0.28.x）不支持 GGUF 主模型加载，HF 侧草稿权重下载也常受限——llama.cpp 双 GGUF 是单卡现实路径
8. KV 量化必须开 `--flash-attn on`；量化 V cache 精度别乱降（q4_0 V 必须搭配正确的 FA 内核）
9. 服务静默退出是常见现象，重启后务必 `ps` + `curl /v1/models` 双确认再开测
10. 电源：单 3090 推理实测 ~150W，原电源够用；加第二张 3090 建议 1200W+

---

## 9. 快速验收清单

```bash
# 1. 启动成功
tail -f ~/llama.log          # 看到 "listening on http://0.0.0.0:8000"
# 2. 显存余量 > 400 MiB
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# 3. 请求进行中 GPU util > 80%（低于 20% → KV 量化回退 CPU，见 §3）
# 4. 服务进程 CPU < 100%（单请求）
# 5. 长文本首字延迟与 tok/s 抽查
```

满足以上 5 条，即为部署成功的基线状态。

---

*基准方法：温度 0，20K/30K/100K/176K 为同源重复文本 prompt，`max_tokens=20` 专测 prefill；
所有对比均在同 payload、同参数下完成；数据为单次采样，趋势可复现。*
