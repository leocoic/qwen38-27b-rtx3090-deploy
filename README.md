# RTX 3090 单卡部署 Qwen3.8-27B（llama.cpp + 推测解码 + 视觉）

> 硬件：RTX 3090 24GB / Ryzen 9 9900X（12C24T）/ 60GB RAM / Ubuntu 22.04 / CUDA 12.x
> 软件版本：llama.cpp `cc83d7b48`（b10684，含 `GGML_CUDA_FA_ALL_QUANTS` 重编译）

## 最终配置

**MTP 推测解码 + Q4_K_S 主模型 + 200K ctx + 混合 KV（K=q8_0 / V=q4_0）+ 视觉**

| 指标 | 数值 |
|---|---|
| prefill 20K / 100K / 176K | ~1,200 / ~845 / ~530 tok/s |
| decode 20K（生产链路实测） | 50~60 tok/s（任务相关，接受率 33%~92%） |
| 视觉理解 | ✅ MTP 版免补丁；DFlash2 版需 §7.1 补丁 |
| 空闲显存余量 | ~470 MiB |

完整实战指南（编译、KV 量化坑、MTP vs DFlash2 对决数据、多模态补丁原理）见
[qwen38-27b-rtx3090-deploy.md](qwen38-27b-rtx3090-deploy.md)。

## 仓库文件

| 文件 | 说明 |
|---|---|
| [qwen38-27b-rtx3090-deploy.md](qwen38-27b-rtx3090-deploy.md) | 完整部署指南（§0 成果 / §3 KV 量化 / §4 推测解码 / §7.1 多模态补丁） |
| `start_llama_mtp.sh` | 生产启动脚本（**推荐，线上配置**）：MTP + 200K + K8/V4 + 视觉 |
| `start_llama_dflash2.sh` | 备选启动脚本：DFlash2（需先应用 `dflash-mm.patch`） |
| `dflash-mm.patch` | llama.cpp `common/speculative.cpp` 多模态兼容补丁（§7.1） |

## 快速开始

```bash
# 1. 按指南 §2 编译 llama.cpp（build-allquant 构建）
# 2. 按 §7.1 对 DFlash2 路径打补丁（选 MTP 可跳过）
# 3. 启动（二选一，脚本自带清理/健康检查/sanity 请求）
bash start_llama_mtp.sh            # 推荐
bash start_llama_dflash2.sh
```

## 选型结论（摘要）

- **推测解码选 MTP**：生产链路同 payload 对比，短 prompt decode +51%、长输出 +24%，
  且多模态免补丁（DFlash2 × 图像有 SWA 环位置合并 bug，见 §7.1）
- **主模型选 Q4_K_S**：UD-Q4_K_M 与 Q4_K_S 几乎同体积且 MTP 短输出 -15%，无收益
- **KV 选 K=q8_0 + V=q4_0**：混合精度省 1.2GB 显存，速度无损（必须 ALL_QUANTS 构建）
