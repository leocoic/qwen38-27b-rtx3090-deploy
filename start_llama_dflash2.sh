#!/bin/bash
# ============================================================
# Qwen3.8-27B llama-server 启动脚本 (DFlash2 看图版)
# 10.1.127.117  RTX 3090 24GB 专用
# ============================================================
# 功能:
#   - 纯文本 + 看图 (vision, mmproj Q8_0)
#   - 200K 上下文 (KV 混合精度: K=q8_0 V=q4_0)
#   - DFlash2 投机解码 n_max=4 (需 --model-draft 草稿模型)
#   - 图像 token 1024~2048 (超限自动降采样)
#
# 用法:
#   bash ~/下载/start_llama_mtp.sh                # 启动
#   bash ~/下载/start_llama_mtp.sh --threads 8    # 追加参数
#
# 日志:   /home/yang-ubuntu/llama_mtp.log  (实时看: tail -f 该文件)
# 停止:   pkill -x llama-server
#
# 说明:
#   - 与 MTP 版 (start_llama_mtp.sh) 仅差 spec-type / model-draft 两处
#   - 依赖 speculative.cpp 本地补丁 (部署文档 §7.1 / dflash-mm.patch):
#     未打补丁时"正文在前、图片在后"的请求必挂; 打了补丁含图轮次无草稿加速
#   - 多模态场景优先用 MTP 版 (start_llama_mtp.sh)
#   - 显存不足时追加: --ubatch-size 256 (省 ~0.5GB)
# ============================================================

set -u

# ---- 路径 ----
MODEL=/home/yang-ubuntu/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf
DRAFT=/home/yang-ubuntu/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
MMPROJ=/home/yang-ubuntu/models/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf
BIN=/home/yang-ubuntu/llama.cpp/build-allquant/bin/llama-server
LOG=/home/yang-ubuntu/llama_dflash2.log
PIDF=/home/yang-ubuntu/llama_dflash2.pid

# ---- 环境 ----
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/yang-ubuntu/llama.cpp/build-allquant/bin:${LD_LIBRARY_PATH:-}
export LLAMA_ARG_FIT=0

# ---- 检查文件 ----
for f in "$MODEL" "$DRAFT" "$MMPROJ" "$BIN"; do
  [ -f "$f" ] || { echo "缺少文件: $f" >&2; exit 1; }
done

# ---- 清理旧进程, 等显存释放 ----
echo ">>> [1/4] 清理现有 llama-server, 等显存释放..."
pkill -9 -x llama-server 2>/dev/null || true
U=9999
for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 9999)
  PORT=$(ss -tln 2>/dev/null | grep -c ':8000 ' || true)
  [ "$U" -lt 500 ] && [ "$PORT" -eq 0 ] && break
  sleep 2
done
echo "    当前显存占用: ${U} MiB"

# ---- 启动 ----
echo ">>> [2/4] 启动 llama-server (DFlash2 看图版)"
echo "    附加参数: $*"
nohup "$BIN" \
  --model "$MODEL" \
  --model-draft "$DRAFT" \
  --mmproj "$MMPROJ" \
  --host 0.0.0.0 \
  --port 8000 \
  --ctx-size 204800 \
  --n-gpu-layers 99 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --spec-draft-type-k q4_0 \
  --spec-draft-type-v q4_0 \
  --spec-type draft-dflash \
  --spec-draft-n-max 4 \
  --flash-attn on \
  --image-min-tokens 1024 \
  --image-max-tokens 2048 \
  --mtmd-batch-max-tokens 8192 \
  --threads 12 \
  --reasoning-effort low \
  "$@" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
echo "    PID=$PID  日志: $LOG"

# ---- 等待加载 ----
echo ">>> [3/4] 等待加载(每5s检查, 最多120s)..."
DEAD=0
for i in $(seq 1 24); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then DEAD=1; break; fi
  if curl -s --max-time 3 http://127.0.0.1:8000/v1/models | grep -q '"name"'; then break; fi
done

if [ "$DEAD" -eq 1 ]; then
  echo "!!! [4/4] 进程已死. 失败原因:"
  grep -E "error|failed|out of memory|exiting" "$LOG" | tail -6
  echo "--- 完整日志: less $LOG"
  exit 1
fi

# ---- 就绪检查 ----
echo ">>> [4/4] 服务就绪"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
grep -iE "MTP draft context|listening" "$LOG" | head -3
echo "--- 文本 sanity:"
curl -s --max-time 60 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"回答 1+1=? 只要数字"}],"max_tokens":10}' | head -c 300
echo
echo "--- 实时日志: tail -f $LOG"
