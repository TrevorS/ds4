# ds4 gamut — prod-greedy-16k

`GB10 sm_121a` · steady decode = 31 tokens (of 39, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   351.8 t/s
- decode    16.86 t/s
- accept    56.8%   tokens/iter 2.14

## GPU HW (decode-windowed, busy samples)
- sms_active     89.7%
- sm_issue       8.9%  ← stalled
- tensor_active  1.2%
- compute_warps  37.4%

## Time split
- wall 2085.9 ms · kernel 1943.1 ms (93.2%) · idle 142.8 ms (6.8%)  ·  trace decode 14.9 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_gate_up_mid_expert_tile8_row32_kernel occupancy-starved (96 regs → 25% occ, 32% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 21.0 | 407.7 | 4530 | 48 | 62% | 6.7 | 57.6 | 1.9 | 84% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 15.3 | 298.2 | 688 | 96 | 25% | 19.3 | 30.8 | 6.2 | 32% |
| moe_down_expert_tile8_row32_kernel | 15.1 | 294.3 | 688 | 128 | 25% | 7.1 | 30.9 | 6.2 | 16% |
| cutlass·wmma·f16·32x32 | 11.4 | 221.1 | 688 | — | — | 3.3 | 18.2 | — | — |
| indexer_scores_wmma128_kernel | 5.0 | 97.7 | 336 | 48 | 62% | 11.4 | 19.7 | — | — |
| cutlass·wmma·f16·16x16 | 5.0 | 97.0 | 4384 | — | — | 2.0 | 5.9 | — | — |
| matmul_q8_0_preq_warp8_kernel | 4.4 | 86.0 | 93 | 40 | 75% | 5.6 | 73.3 | — | — |
| attention_indexed_mixed_heads…ne_kernel<8, 16> | 4.3 | 83.7 | 336 | — | — | 9.0 | 8.4 | — | — |
| attention_decode_mixed_kernel | 4.3 | 82.7 | 383 | 40 | 75% | 15.3 | 30.0 | — | — |
| compressor_update_pool_kernel | 1.7 | 32.8 | 440 | 48 | 62% | 1.7 | 5.5 | — | — |
| matmul_q8_0_preq_batch_share_warp_kernel<2> | 1.4 | 26.5 | 302 | 48 | 62% | — | — | 1.9 | 86% |
| rms_norm_plain_kernel | 1.3 | 24.8 | 1485 | 16 | 100% | 1.0 | 2.0 | 0.5 | 1% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1618.8 | argmax_kernel | rms_norm_plain_kernel |
| 1470.2 | argmax_kernel | rms_norm_plain_kernel |
| 1150.9 | argmax_kernel | rms_norm_plain_kernel |
| 665.6 | argmax_kernel | embed_token_hc_kernel |
| 662.7 | argmax_kernel | embed_token_hc_kernel |
| 655.2 | argmax_kernel | embed_token_hc_kernel |
