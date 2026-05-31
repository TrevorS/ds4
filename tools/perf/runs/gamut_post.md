
<!-- json sidecar: /home/trevor/Projects/ds4/tools/perf/runs/gamut_post.json -->
# ds4 gamut — post

`GB10 sm_121a` · steady decode = 80 tokens (of 96, skipped 16 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   — t/s
- decode    — t/s

## GPU HW (decode-windowed, busy samples)

## Time split
- wall 8072.8 ms · kernel 7530.9 ms (93.3%) · idle 541.9 ms (6.7%)  ·  trace decode 9.9 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> no clear single bottleneck

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_f16_wmma_bi_kernel | 21.0 | 1579.4 | 16720 | — | — | — | — | — | — |
| matmul_q8_0_preq_warp8_kernel | 15.7 | 1180.7 | 3520 | — | — | — | — | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 13.0 | 978.5 | 6880 | — | — | — | — | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 9.9 | 743.2 | 3440 | — | — | — | — | — | — |
| moe_gate_up_mid_decode_lut_qwarp32_kernel | 8.6 | 646.5 | 3440 | — | — | — | — | 6.2 | 74% |
| matmul_q8_0_pair_preq_warp8_kernel | 7.7 | 576.4 | 6880 | — | — | — | — | — | — |
| attention_indexed_mixed_kernel | 7.2 | 543.0 | 1680 | — | — | — | — | — | — |
| moe_down_sum6_qwarp32_kernel | 5.2 | 390.5 | 3440 | — | — | — | — | 6.2 | 61% |
| matmul_f16_ordered_chunks_kernel | 3.1 | 236.1 | 6960 | — | — | — | — | — | — |
| attention_decode_mixed_kernel | 2.3 | 174.6 | 1760 | — | — | — | — | — | — |
| rms_norm_plain_kernel | 1.6 | 122.9 | 6960 | — | — | — | — | 0.5 | 1% |
| hc_split_weighted_sum_norm_fused_kernel | 1.3 | 98.3 | 6880 | — | — | — | — | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 472.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 470.5 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 467.6 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 465.7 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 454.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 452.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
