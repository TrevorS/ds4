# ds4 gamut — prod-sampled-16k

`GB10 sm_121a` · steady decode = 88 tokens (of 96, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   339.8 t/s
- decode    12.05 t/s

## GPU HW (decode-windowed, busy samples)
- sms_active     87.1%
- sm_issue       7.4%  ← stalled
- tensor_active  0.0%
- compute_warps  46.3%

## Time split
- wall 3648.7 ms · kernel 3468.3 ms (95.1%) · idle 180.4 ms (4.9%)  ·  trace decode 24.1 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_warp8_kernel | 15.8 | 547.0 | 2068 | 40 | 75% | 5.7 | 68.1 | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 11.8 | 409.0 | 3872 | 40 | 75% | 4.4 | 51.5 | — | — |
| matmul_f16_ordered_chunks_kernel | 9.7 | 336.2 | 7568 | 35 | 88% | 4.1 | 23.9 | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 9.3 | 321.0 | 1936 | 48 | 62% | 5.4 | 59.3 | — | — |
| moe_gate_up_mid_decode_lut_qwarp32_kernel | 8.8 | 305.8 | 1892 | 40 | 75% | 17.8 | 31.0 | 6.2 | 86% |
| matmul_q8_0_pair_preq_warp8_kernel | 7.0 | 241.4 | 3872 | 48 | 62% | 4.3 | 49.6 | — | — |
| attention_decode_mixed_kernel | 6.9 | 240.6 | 1012 | 40 | 75% | 3.3 | 18.9 | — | — |
| matmul_f16_pair_ordered_chunks_kernel | 6.6 | 229.7 | 2728 | 38 | 75% | 1.7 | 21.6 | — | — |
| moe_down_sum6_qwarp32_kernel | 5.6 | 193.8 | 1892 | 40 | 75% | 4.9 | 43.1 | 6.2 | 68% |
| attention_indexed_mixed_kernel | 4.8 | 165.1 | 924 | 40 | 75% | 11.5 | 24.2 | — | — |
| indexer_score_one_direct_kernel | 3.2 | 110.4 | 924 | 28 | 100% | 32.2 | 76.1 | — | — |
| rms_norm_plain_kernel | 1.9 | 65.6 | 3960 | 16 | 100% | 1.0 | 1.1 | 0.5 | 1% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 405.8 | argmax_kernel | embed_token_hc_kernel |
| 382.6 | argmax_kernel | embed_token_hc_kernel |
| 367.3 | argmax_kernel | embed_token_hc_kernel |
| 352.5 | argmax_kernel | embed_token_hc_kernel |
| 350.2 | argmax_kernel | embed_token_hc_kernel |
| 337.8 | argmax_kernel | embed_token_hc_kernel |
