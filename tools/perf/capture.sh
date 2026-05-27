#!/usr/bin/env bash
# tools/perf/capture.sh — one-shot GB10 decode perf capture → gamut report.
#
# Runs the whole dance (rebuild → plain nsys trace → gb20b metrics → ptxas regs
# → accept telemetry → optional ncu stalls → gamut MD/JSON/HTML) so each kernel
# experiment is one command. Captures are serialized (ds4 holds the GPU lock).
#
#   tools/perf/capture.sh --label moe_down_lb2 [--rebuild] [--ncu] [-n 48]
#       [-m ds4flash.gguf] [--mtp PATH] [-p knight]
#
# Output: tools/perf/runs/<label>.{md,json,html}  (+ /tmp/<label>_*.{nsys-rep,sqlite})
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NVCC=/usr/local/cuda/bin/nvcc
ARCH="-gencode=arch=compute_121a,code=sm_121a"

LABEL=""; REBUILD=0; DO_NCU=0
MODEL="ds4flash.gguf"
MTP="/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
PROMPT="knight"; NTOK=48
# Prod-flag matrix knobs (defaults preserve the original synthetic capture).
CTX=""; TEMP="0"; PROMPTFILE=""; WARM=0; THINKFLAG="--nothink"
KERNELS="moe_down_expert_tile8_row32|matmul_q8_0_preq_batch_share_warp|moe_gate_up_mid_expert_tile8_row32"

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --rebuild) REBUILD=1; shift;;
    --ncu) DO_NCU=1; shift;;
    -m) MODEL="$2"; shift 2;;
    --mtp) MTP="$2"; shift 2;;
    -p) PROMPT="$2"; shift 2;;
    -n) NTOK="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --temp) TEMP="$2"; shift 2;;
    --prompt-file) PROMPTFILE="$2"; shift 2;;
    --warm) WARM=1; shift;;
    --think) THINKFLAG="--think"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$LABEL" ] || { echo "need --label" >&2; exit 2; }

cd "$ROOT"
RUNS="$HERE/runs"; mkdir -p "$RUNS"
T="/tmp/${LABEL}"
ARGS=(-m "$MODEL" --mtp "$MTP" -n "$NTOK" --temp "$TEMP" "$THINKFLAG" -sys "")
if [ -n "$PROMPTFILE" ]; then ARGS+=(--prompt-file "$PROMPTFILE"); else ARGS+=(-p "$PROMPT"); fi
[ -n "$CTX" ] && ARGS+=(--ctx "$CTX")
[ "$WARM" = 1 ] && ARGS+=(--warm-weights)

[ "$REBUILD" = 1 ] && { echo "## rebuild"; make cuda-spark >/dev/null; }

echo "## 1/5 plain CUDA trace"
nsys profile -o "${T}_p" -t cuda --sample none --force-overwrite true ./ds4 "${ARGS[@]}" >/dev/null 2>&1
nsys export --type sqlite --force-overwrite true -o "${T}_p.sqlite" "${T}_p.nsys-rep" >/dev/null 2>&1

echo "## 2/5 gb20b GPU metrics"
nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=gb20b --gpu-metrics-frequency=20000 \
  -o "${T}_gm" --force-overwrite true ./ds4 "${ARGS[@]}" >/dev/null 2>&1
nsys export --type sqlite --force-overwrite true -o "${T}_gm.sqlite" "${T}_gm.nsys-rep" >/dev/null 2>&1

echo "## 3/5 ptxas registers"
$NVCC -O3 --use_fast_math $ARCH -Xptxas=-v -c -o /tmp/_ptxas.o ds4_cuda.cu 2>"${T}_ptxas.txt"

echo "## 4/5 accept + throughput"
DS4_MTP_TIMING=1 ./ds4 "${ARGS[@]}" >"${T}_accept.txt" 2>&1 || true
PREFILL=$(grep -oE 'prefill: [0-9.]+' "${T}_accept.txt" | tail -1 | grep -oE '[0-9.]+' || echo "")
DECODE=$(grep -oE 'generation: [0-9.]+' "${T}_accept.txt" | tail -1 | grep -oE '[0-9.]+' || echo "")

NCU_ARG=()
if [ "$DO_NCU" = 1 ]; then
  echo "## 4.5 ncu stalls (application replay; slow)"
  python3 "$HERE/ncu_stalls.py" --out "${T}_ncu.json" --kernels "$KERNELS" \
    --launch-skip 200 --launch-count 12 -- ./ds4 "${ARGS[@]}" >/dev/null 2>&1 || true
  [ -s "${T}_ncu.json" ] && NCU_ARG=(--ncu "${T}_ncu.json")
fi

echo "## 5/5 gamut report"
python3 "$HERE/gamut.py" \
  --plain "${T}_p.sqlite" --metrics "${T}_gm.sqlite" \
  --ptxas "${T}_ptxas.txt" --accept "${T}_accept.txt" "${NCU_ARG[@]}" \
  ${PREFILL:+--prefill-tps "$PREFILL"} ${DECODE:+--decode-tps "$DECODE"} \
  --label "$LABEL" --json "$RUNS/$LABEL.json" --html "$RUNS/$LABEL.html" \
  >"$RUNS/$LABEL.md"

echo "done → $RUNS/$LABEL.{md,json,html}"
echo "  prefill ${PREFILL:-?} t/s · decode ${DECODE:-?} t/s"
