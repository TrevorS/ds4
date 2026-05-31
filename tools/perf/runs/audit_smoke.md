# ds4 gamut — audit_smoke

`GB10 sm_121a` · steady decode = 15 tokens (of 23, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   8.5 t/s
- decode    13.19 t/s

## GPU HW (decode-windowed, busy samples)
- sms_active     87.3%
- sm_issue       5.1%  ← stalled
- tensor_active  0.1%
- compute_warps  45.7%

## Time split
- wall 1194.2 ms · kernel 1115.1 ms (93.4%) · idle 79.1 ms (6.6%)  ·  trace decode 12.6 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_f16_wmma_bi_kernel | 18.1 | 201.8 | 2505 | 40 | 75% | 1.2 | 3.1 | — | — |
| matmul_q8_0_preq_warp8_kernel | 16.8 | 187.6 | 660 | 40 | 75% | 4.5 | 64.7 | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 15.7 | 175.6 | 1290 | 40 | 75% | 3.4 | 55.9 | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 11.9 | 133.0 | 645 | 48 | 62% | 4.3 | 59.3 | — | — |
| moe_gate_up_mid_decode_lut_qwarp32_kernel | 10.4 | 116.5 | 645 | 40 | 75% | 16.0 | 31.1 | 6.2 | 77% |
| matmul_q8_0_pair_preq_warp8_kernel | 8.5 | 94.5 | 1290 | 48 | 62% | 3.3 | 49.4 | — | — |
| moe_down_sum6_qwarp32_kernel | 5.9 | 66.1 | 645 | 40 | 75% | 4.7 | 44.4 | 6.2 | 68% |
| matmul_f16_ordered_chunks_kernel | 3.7 | 41.2 | 1305 | 35 | 88% | 1.2 | 5.8 | — | — |
| attention_decode_mixed_kernel | 3.0 | 33.7 | 645 | 40 | 75% | 3.2 | 31.2 | — | — |
| rms_norm_plain_kernel | 1.9 | 21.3 | 1305 | 16 | 100% | 1.0 | 1.0 | 0.5 | 1% |
| hc_split_weighted_sum_norm_fused_kernel | 1.5 | 16.9 | 1290 | 40 | 75% | 2.0 | 15.3 | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 88.6 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 72.7 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 69.9 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 69.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 68.4 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 68.3 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
