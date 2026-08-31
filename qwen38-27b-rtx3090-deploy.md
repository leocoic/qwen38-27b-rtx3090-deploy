# RTX 3090 单卡部署 Qwen3.8-27B 实战指南（llama.cpp + 推测解码 + 视觉）

> 硬件：RTX 3090 24GB / Ryzen 9 9900X（12C24T）/ 60GB RAM / Ubuntu 22.04 / CUDA 12.x
> 软件版本：llama.cpp `cc83d7b48`（b10684）+ 本地多模态补丁（§7.1，`git pull` 重编译后需重打）
> 本文所有路径以 `~` 代替用户主目录，端口/参数请按需修改。**不含任何 IP、密钥、密码等敏感信息。**

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
| 视觉理解 | ✅ mmproj + image min/max-tokens（推荐 MTP 版免补丁；DFlash2 版需打 §7.1 补丁） |
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

**接受率与任务强相关**（补丁后复测，20K 上下文 + 相同 prompt 前缀，仅换生成任务）：

| 生成任务 | 接受率 | decode |
|---|---|---|
| 数数（高可预测） | 93% | 97.8 tok/s |
| 复述上下文结尾 | 83% | 90.8 tok/s |
| 单字问答 | 70% | 78.7 tok/s |
| 创作短文（最难预测） | 45% | 58.7 tok/s |

本文的接受率数字（65~88%）来自同源重复文本类 prompt，属偏有利任务；
拿创意写作类请求会测到低得多的接受率，属正常现象而非部署问题。
decode 速度与接受率同步波动（93% → 98 tok/s，45% → 59 tok/s），
评估 speculative 收益时务必用与真实业务同类的任务。

**MTP vs DFlash2 同环境对决**（补丁后复测：Q4_K_S + 200K ctx + K8/V4，n_max=4，温度 0）：

| 指标 | DFlash2（+§7.1 补丁） | MTP | 优者 |
|---|---|---|---|
| decode short @20K/30K/100K/176K | 65.3 / 54.9 / 44.3 / 49.0 | 64.5 / 61.5 / 47.4 / 43.2 | 互有胜负 |
| decode long @同上（300 tok） | 57.3 / 51.5 / 43.4 / 25.8 | 64.9 / 60.4 / 45.0 / 30.9 | **MTP 全胜（+8~20%）** |
| 接受率 @20K/30K/100K/176K（long） | 44 / 39 / 42 / 26% | 56 / 54 / 52 / 44% | **MTP（+10~18pp）** |
| 含图对话（10K 正文 + 图） | 22.6 tok/s，接受率 2% | 55.2 tok/s，接受率 45% | **MTP（2.4×）** |
| 多模态需要 §7.1 补丁 | 需要 | **不需要** | MTP |
| 空闲显存 | 23,601 MiB | 23,643 MiB | 持平 |
| prefill @30K/100K/176K | 1,072 / 816 / 528 | 1,091 / 840 / 517 | 持平 |

结论：**多模态部署选 MTP**（图文通吃且免补丁）；纯文本部署两者皆可，
MTP 长输出全面略优。DFlash2 仅 176K short 一点领先。

**生产链路复测**（经 Trae 同款代理链路：网关 → sidecar → GPU :8000，
同一组 payload、同一会话内切引擎 A/B，纯文本、温度 0）：

| 场景 | DFlash2（+§7.1 补丁） | MTP | 差距 |
|---|---|---|---|
| 短 prompt decode（300 tok 输出） | 39.8 tok/s（接受率 18%） | 60.3 tok/s（43%） | **MTP +51%** |
| 10K 上下文 decode（300 tok 输出） | 39.9 tok/s（20%） | 49.5 tok/s（33%） | **MTP +24%** |
| 10K 流式 150 tok 总耗时（455 chunks） | 3.2s | 2.5s | MTP 快 22% |
| 客户端完整请求 wall（10K + 300 tok） | 8.0s | 6.5s | MTP 快 19% |

结论：走生产链路结论不变——**纯文本也是 MTP 全面占优**，且短 prompt 场景
（agent 高频路径）优势最大。代理网关会原样透传 llama-server 的 `timings`
与 `draft_n/draft_n_accepted` 字段，链路级 A/B 可直接复用本文方法。

**Q4_K_M 主模型复测**（生产链路、同 payload、180K ctx；UD-Q4_K_M 实测
仅 15.3 GB，与 Q4_K_S 几乎同体积，装 200K 也够）：

| 配置 | 空闲显存 | 短 prompt decode | 10K decode | 流式 150 tok |
|---|---|---|---|---|
| MTP + Q4_K_S @200K（线上） | 482 MiB | **60.3 tok/s（43%）** | **49.5 tok/s（33%）** | **2.5s** |
| MTP + Q4_K_M @180K | 480 MiB | 51.1 tok/s（35%） | 47.3 tok/s（33%） | 2.85s |
| DFlash2 + Q4_K_S @200K | 524 MiB | 39.8 tok/s（18%） | 39.9 tok/s（20%） | 3.2s |
| DFlash2 + Q4_K_M @180K | 374 MiB | 37.3 tok/s（18%） | 35.5 tok/s（17%） | 3.9s |

结论：**Q4_K_M 未复现历史 +3~5% 优势，反而略慢**（MTP 短 prompt -15%，
伴随 MTP 头量化差异导致的接受率下降 43%→35%；DFlash2 同趋势）。两种引擎
下 Q4_K_S 都是更优选择，维持 Q4_K_S + MTP + 200K 为最终配置。
注：历史"+3~5%"是 DFlash2 + 150K ctx + 不同 payload 下的结果，不代表本配置。

---

## 5. 最终启动脚本

本仓库收录两份生产启动脚本（统一结构：启动前清理 + 就绪健康检查 + sanity 请求）：

| 文件 | 配置 | 说明 |
|---|---|---|
| `start_llama_mtp.sh` | MTP（**推荐，当前线上**） | 免补丁，图文通吃 |
| `start_llama_dflash2.sh` | DFlash2（备选） | 需先打 §7.1 补丁（脚本头部有警告） |

脚本结构（两版相同，约 110 行）：

1. 文件检查（模型 / 草稿 / mmproj / 二进制缺失即退出）
2. `pkill -9 -x llama-server` 精确匹配 + **VRAM 与端口双条件等待释放**
   （只等显存会在竞态下让新实例撞端口 `couldn't bind HTTP server socket`，
   健康检查误报"进程已死"）
3. `nohup` 启动 + PID 入 pid 文件
4. 加载等待（每 5s，最多 120s；检测死进程并 grep 日志找失败原因）
5. 就绪检查（/v1/models）+ 自动 sanity 请求（max_tokens=10）
6. `"$@"` 支持追加参数：`bash start_llama_mtp.sh --log-verbosity 2`

MTP 版核心命令（DFlash2 版仅多 `--model-draft` 与 `--spec-type draft-dflash` 两处）：

```bash
llama-server \
  --model   ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --mmproj  ~/models/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf \
  --host 0.0.0.0 --port 8000 --ctx-size 204800 \
  --n-gpu-layers 99 --parallel 1 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --flash-attn on \
  --image-min-tokens 1024 --image-max-tokens 2048 \
  --mtmd-batch-max-tokens 8192 \
  --threads 12 --reasoning-effort low
```

**显存预算**（200K ctx，实测空闲 MTP 23,643 / DFlash2 23,601 MiB）：

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
> `--image-max-tokens 2048`：图像 token 硬上限，超限图自动降采样
> （DFlash2 草稿缓存兜不住更大的图，见 §7.1）；
> `--mtmd-batch-max-tokens 8192`：mtmd 编码批次上限，默认 1024，不改则多模态请求 500。

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
- `--image-max-tokens 2048`：图像 token 硬上限，超限图自动降采样（DFlash2 草稿兜不住更大的图，见 §7.1）
- `--mtmd-batch-max-tokens 8192`：单次多模态编码批次上限，默认 1024，不改则大图/多图请求 500
- BF16 mmproj（931MB）在满配下必 OOM，用 Q8_0（629MB）
- 请求格式：仅支持 OpenAI 风格 `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`，
  且每个 content 段必须带非空 `text`；anthropic 风格 400、空 text 500

图像 token 数实测（min/max = 1024/2048，mmproj 元数据 image_size=768, patch=16, merge=2）：

| 源图 | 实际 prompt token | 结果 |
|---|---|---|
| 512×512 / 1024×1024 | ~1,069 | ✅ |
| 1920×1080 | 2,085 | ✅ |
| 2560×1440 / 2048×2048 | 2,025 / 2,070（自动降采样） | ✅ |
| 不设 max 上限 | ~21K（2048² 源图） | ❌ 失败 |

### 7.1 DFlash2 × 多模态必挂 —— 上游 bug 与本地补丁（重要）

**症状**：只要"正文在前、图片在后"，请求 100% 失败返回 `failed to process mtmd chunk`。
服务端日志三连：

```
W find_slot: non-consecutive token position 9122 after 9121 ... with 512 new tokens
W decode: failed to find a memory slot for batch of size 512
E process: llama_decode(ctx_dft) failed rc=1 (n_tokens=512, offset=512)
```

而图片放在最前（位置 0）反而能过——**纯属巧合**：图 token 数恰好塞满草稿缓存环。
"正文在前必挂、图在前偶尔过"是排查时最大的迷惑项。

**根因**（两层叠加，加调试日志实测确认）：

1. Qwen3.8-VL 对图像 token 做**位置合并**：整张图共享同一个 position。
   注入草稿的 batch 位置是 `[9122, 9122, 9122, ...]`，不是递增序列
2. DFlash2 草稿自身 KV 是 **SWA 滚动环**（启动日志 `creating SWA KV cache, size = 2560 cells`
   = n_swa 2048 + ubatch 512）。同 position 的 cell 永远滑不出 2048 窗口 → 无法淘汰；
   第一块 512-token 注入侥幸挤出空间后，第二块注入时环已满 → `find_slot` 失败 → 请求 500

**上游状态**：`common/speculative.cpp` dflash `process()` 里留着
`TODO: revisit after ggml-org/llama.cpp#24669 is merged`，截至最新 master 未修。

**补丁**：让草稿跳过多模态 embedding 注入（文本 token 注入不受影响）。
补丁完整内容见本仓库的 **`dflash-mm.patch`**（对 llama.cpp `cc83d7b48` 生成，
`git apply --check` 校验通过）：在 `common/speculative.cpp` 的
`common_speculative_impl_draft_dflash::process()` 内、`if (has_tokens == has_embeddings)`
判断之后新增 `if (has_embeddings) { return true; }`（含注释共 10 行）。

**方式一（推荐）：git apply 补丁文件**

```bash
cd ~/llama.cpp
cp /path/to/repo/dflash-mm.patch ./
cp common/speculative.cpp /tmp/speculative.cpp.bak-dflash-pos   # 留底
git apply --check dflash-mm.patch && git apply dflash-mm.patch
cmake --build build-allquant --target llama-server -j$(nproc)
```

`--check` 失败说明 llama.cpp 版本差异过大（hunk 上下文对不上），改用方式二。

**方式二：手工/脚本插入**

编辑 `common/speculative.cpp`，在 `common_speculative_impl_draft_dflash::process()`
内找到：

```cpp
        if (has_tokens == has_embeddings) {
            return true;
        }
```

紧随其后插入：

```cpp
        // [local patch] skip multimodal embedding batches: the DFlash draft's SWA ring
        // cannot accommodate position-consolidated image tokens (hundreds of cells at
        // one position never age out of the SWA window), which made llama_decode(ctx_dft)
        // fail for any image preceded by text. Skipping leaves a context hole in the
        // draft cache: speculative acceptance degrades for image conversations while
        // decoding stays correct. Revisit upstream (ggml-org/llama.cpp#24669).
        if (has_embeddings) {
            return true;
        }
```

python 脚本版（anchor 断言，版本变了会报错而不是打歪）：

```bash
cd ~/llama.cpp
python3 - << 'EOF'
path = 'common/speculative.cpp'
src = open(path).read()
anchor = """        if (has_tokens == has_embeddings) {
            return true;
        }"""
inject = anchor + """
        if (has_embeddings) {
            return true;
        }"""
assert src.count(anchor) == 1, "anchor not found, llama.cpp version may have changed"
open(path, 'w').write(src.replace(anchor, inject))
print("patched")
EOF
```

**补丁代价**（实测）：

| 场景 | 补丁前 | 补丁后 |
|---|---|---|
| 纯文本对话 | DFlash2 全速 | 不变（复测见下表） |
| 含图对话 decode | 直接报错 | 22.6 tok/s 实测（接受率 2%，死草稿空转税，比无草稿基线 ~40 还低） |
| 回答正确性 | — | ✅ 无损（主模型上下文完整） |

**补丁后纯文本 decode 复测**（Q4_K_S + DFlash2 n=4 + 200K ctx + K8/V4 KV；
同 payload 先发 `max_tokens=5` 预热前缀缓存再测，温度 0）：

| 上下文 | short 输出（20 tok） | long 输出（300 tok） | 草稿接受率（long） |
|---|---|---|---|
| ~15 tok（agent 类短 prompt） | — | 39.7 tok/s | — |
| 20K | 65.3 tok/s | 57.3 tok/s | 44% |
| 30K | 54.9 tok/s | 51.5 tok/s | 39% |
| 100K | 44.3 tok/s | 43.4 tok/s | 42% |
| 176K | 49.0 tok/s | 25.8 tok/s | 26% |

同轮 prefill 复测：30K 1,072 / 100K 816 / 176K 528 tok/s（与 §0 数字同量级）。
结论：**补丁对纯文本零回归**——short 输出落在本文既有 62~76（20K）与 ~42~49（176K）
区间内；176K 长输出 decode 偏低与草稿在超长上下文接受率下降（26%）相关，
属 DFlash2 本身特性，与补丁无关（补丁只跳过多模态 batch 的注入）。

> 口径提醒：上表 20K~176K 行的 long 输出任务为"创作短文"（对草稿最不利的任务），
> 接受率仅 26~44%；不同任务的接受率差异巨大（§4 有 45%~93% 的对照）。
> 另外预热后 short/long 请求只重算 ~30 token 后缀，其"prefill tok/s"
> （31~47）是缓存命中小批量的口径假象，真实 prefill 见预热行（528~1,072）。

原理：跳过注入后草稿缓存留"洞"，含图轮次的推测接受率≈0（实测 2/108）；
但草稿并未停摆——每个 decode 步仍要做 4-token 提议 + 5-token 目标验证，
全部被拒 = 纯开销。实测含图轮次 decode 22.6 tok/s，**比无草稿基线（~40）还低**
（等于单解速度再扣一笔"死草稿税"）；纯文本轮次完全不受影响（补丁在文本
路径只加一个分支判断，补丁前后实测一致）。

**多模态场景的更优解：直接换 MTP**。`--spec-type draft-mtp` 并删去
`--model-draft`（MTP 头内嵌于主模型 GGUF，`qwen35.nextn_predict_layers = 1`，
草稿上下文复用主模型权重）。其图像注入走独立实现、普通 KV 无 SWA 环限制，
**免补丁、图文通吃**：同样的"10K 正文 + 图"请求 decode 55.2 tok/s、接受率 45%
（DFlash2+补丁：22.6 / 2%）。文本性能对决见 §4——MTP 长输出全面更快，
本文最终推荐配置即 MTP 版（§5）。

**注意**：`git pull` / 重编译会还原源文件，补丁需重打。重打前先
`git apply --reverse --check dflash-mm.patch` 判断是否已打过（报错=未打）。

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
11. 重启脚本杀进程后必须等 **VRAM 与端口双条件**释放再启动：只等显存会留竞态窗口，新实例 `couldn't bind HTTP server socket, port: 8000` 直接退出，健康检查误报"进程已死"（§5）
11. DFlash2 + 图片（正文在前）必挂：图像位置合并 × 草稿 SWA 环无法淘汰同位置 cell
    → 打 §7.1 补丁；"图放最前能过"是撞上缓存环容量的巧合，别依赖
12. 客户端/代理层报错不可信（错误码可能被替换成请求 ID），真实原因以服务端日志为准
    （如 `failed to process mtmd chunk, res = -1`），配合 `--verbose` 看 `find_slot` 警告定位

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
