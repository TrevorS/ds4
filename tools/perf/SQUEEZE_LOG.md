# SQUEEZE LOG — GB10 decode perf

Running log of perf experiments driven by the gamut suite. Each row is a
measured change (or a measured rejection). Baseline = `gb10-on-upstream`
(upstream + the 3 GB10 PRs), knight prompt, MTP draft=2.

Bottleneck (from the gamut report): decode is **memory-latency-bound at moderate
occupancy** — SM-issue ~8%, ~38% of the measured 236 GB/s read ceiling. The
lever is occupancy (hide weight-read latency), not bandwidth.

## Experiments

| # | change | result | verdict |
|---|--------|--------|---------|
| 1 | `moe_down` `__launch_bounds__(256,2)` | regs 168→128, occ 16→31%, barrier 53→42%, `moe_down` −20% ms, **decode +3–5%** | ✅ **KEEP** (bit-safe, `--long-context` OK) |
| 2 | `moe_down` `__launch_bounds__(256,3)` | regs→≤85 → heavy spill; occ 31→46% but `moe_down` ms 307→743 (+142%), stall flips barrier→lg_throttle, **decode −18%** | ✗ reject — spill |
| 3 | server `--mtp-draft` default 1→2 | enables combined-forward on server (was off); output-identical | ✅ keep (free) |
| 4 | narrow-accumulator `moe_down` via tile4 (`DS4_CUDA_MOE_TILE4=1`) | **−19%** decode (16.06 vs 19.78 t/s); output bit-identical to tile8 | ✗ reject — 2× weight reads |

### #4 detail — why narrow accumulators don't help

tile4 is *also* 128 regs (the `[4]` accumulators help less than hoped — pointers
+ dot temps dominate), so **no occupancy gain** over bounded tile8, while halving
pairs/block doubles the blocks → **2× expert-weight reads** (the large weights
evict L2, so it's not absorbed). The genuine "narrow accumulators, same weight
traffic" variant is blocked structurally: `acc[8]` must persist across the whole
K-reduction loop, so shrinking to `[4]` forces either a 2nd weight pass (=tile4)
or smem-resident accumulators (overhead). **`moe_down(256,2)` is the occupancy
optimum for this dot-over-K structure.** Bit-identity confirmed (tiling only
repartitions work across blocks; per-output float order is unchanged).

## Ruled out (with evidence — launch_bounds avenue exhausted)

- **`moe_gate_up`** — occupancy-quantization-locked. 113 regs → 2 blocks; smem
  38.6 KB → 2 blocks (co-limited). Reaching 3 blocks needs ≤85 regs **and** a
  smem cut — compound spill risk; expected EV negative after experiment #2.
- **`moe_down_expert_*` family rollout** — the `tile16*` variants use `[16]`-wide
  accumulators (more regs than tile8's 168), so `(256,2)`'s ≤128 ceiling would
  spill them like #2. Not safe.
- **Other decode kernels** — `share_warp<3>` (48 regs, ~83% occ, at the 86%-BW
  wall), `preq_warp8` / `attention_decode_mixed` (40 regs, 75% occ), `rms_norm`
  (100% occ) have no register headroom. Only `moe_down`/`gate_up` were
  register-starved; #1 banked the one safe win.

## Prefill (profiled ctx 8192, promessi — `gamut --phase prefill`)

Different regime from decode: **SM-issue 33%, tensor 4.4%, SMs-active 99.6%** —
compute-occupied, not memory-latency-stalled. Top kernels: `moe_gate_up
…rowspan<1024>` (33%), `moe_down_expert_tile16_row2048` (26%, **232 regs → 12%
occ**), `attention_indexed_mixed<8,16>` (10%, already 67% warps). An f16 path is
present (`f32_to_f16` + `nvjet`/cuBLAS + cutlass), tensor cores underused (4.4%).

No easy launch_bounds win: `tile16_row2048` at 232 regs would need ≤128 (a
104-reg cut) for 2 blocks — far past the 168→85 cut that spilled in decode #2,
and prefill is more compute-occupied (SMiss 24.8%) so occupancy is a weaker
lever. Prefill is at a structural limit too. (Tooling note: `roofline_estimate`
lacks prefill kernel-name matchers, so prefill `%peakBW` shows `—`.)

## CUDA-graph decode (greenlit — in progress)

Goal: kill the per-token `argmax→embed` host round-trip (~7.7% idle). Built as a
**non-strict, gated (`DS4_GRAPH_DECODE=1`) plain-greedy-only** path (strict + MTP
untouched).

- **Stage 0** (done): `--default-stream per-thread` → all launches on a
  capturable `cudaStreamPerThread`. Verified bit-identical (`long-context` OK),
  ~neutral perf (19.58 vs 19.76 t/s).
- **Stage 1** (done): per-token `cudaStreamBeginCapture(forward)→instantiate→
  replay` in `metal_graph_eval_token_raw_swa`. **Result: capture fires (nsys: 15
  instantiate+launch for n=16), zero errors, output BIT-IDENTICAL to eager —
  no sm_121a captured-graph drift for plain greedy.** This clears the make-or-break
  drift gate. No speedup yet (per-token re-instantiate ~1.67 ms + still host-syncs).
- **Stage 2** (done): device-resident token feedback — argmax→`comp_selected`,
  `embed_tokens_hc`(device) + router device-token (kernels already read
  `tokens[t]`), forward chained through device buffer *views* so the host
  pipelines the whole burst with **one** sync (no per-token sync), then reads the
  token log. **Result: BIT-IDENTICAL to eager, but no speedup (16.47 vs 16.47).**

### CORRECTED finding: decode idle IS host-launch latency — graphs win +6.5%

Plain-greedy decode is 93.4% kernel / **6.6% idle**, and the idle is **73,232
sub-5µs gaps across 1562 kernels/token** — host-launch latency, not GPU stalls.
Decisive timing test (capture forward ONCE, launch the cached exec per token =
1 host submit vs 1562 eager launches): **16.49 → 17.55 t/s, +6.5%.**

My earlier "dead end" was WRONG. Stage 1 missed this because it re-instantiated
(1.67ms) AND re-captured per token — re-capture re-issues all 1562 launches, so
no win. Stage 2 (device-token, no per-token sync) only removed the ~80µs/token
boundary gap (2% of idle), not the intra-forward launch latency. The win needs
**capture-once + launch-cached** (proven above).

**DONE — correct +5% landed.** The winning recipe (`DS4_GRAPH_DECODE=1`,
plain-greedy): each token, re-capture the device-token forward, patch the cached
exec via **`cudaGraphExecUpdate`** (cheap — falls back to re-instantiate only on
topology change: compressor emit / indexer growth), launch as ONE submit, no
per-token sync. Two surprises that broke my earlier "dead end":
1. Recording 1562 kernels in capture mode is **cheaper** than eager-launching them.
2. `ExecUpdate` (vs the 1.67ms instantiate Stage 1 used) is cheap enough per token.

Measured: knight n=48 **16.50→17.41 t/s (+5.5%)**; essay n=192
**16.04→16.86 (+5.1%)** — both **BIT-IDENTICAL** to eager. Ceiling (launch cached,
no re-capture, wrong output) is 17.55 (+6.5%); re-capture costs ~1%.

**Lesson: I twice declared "dead end" prematurely.** Stage 1 (re-instantiate/token)
and Stage 2-eager (device-token, no graph) each tested only half; the win needed
capture-once-style *launch* (graph collapses 73k host-launch gaps) + cheap
per-token ExecUpdate. Scope to extend: the MTP/session decode path (this is wired
into the plain-greedy CLI loop only).

## Bigger levers (owner-level — proposals, not done)

The readily-available occupancy squeeze is extracted. Further decode gains need
structural change (surface as proposals per the contributor role):

- **Narrower-accumulator decode kernel** — a `[4]`-wide `moe_down` (≈half the
  regs) could hit 3 blocks without spilling, at the cost of more weight re-reads;
  bit-identity must be checked (different reduction grouping).
- **Launch-gap elimination** — `argmax → embed_token_hc` is 0.6–1.2 ms/token
  (~7.7% idle); a CUDA-graph or async-sample decode loop would reclaim it.
- **MTP adaptive cascade depth** (N=2↔3) — robustness on low-accept prose; the
  ~54% accept ceiling itself is MTP-head-quality bound (needs model work).
- **Prefill** — separate regime (n_tok≥128, `tile16_row2048`); not yet profiled.

## Method note

`(256,3)` raised *occupancy* but tanked throughput — the canonical reminder that
occupancy is a means, not the goal. The gamut diff caught it instantly: occ↑ but
ms↑↑, %peakBW↓, stall reason flipped to `lg_throttle` (spill traffic). Always
read ms + stall-reason alongside occupancy.
