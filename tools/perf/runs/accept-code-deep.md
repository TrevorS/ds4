# ds4 gamut — accept-code-deep

`GB10 sm_121a` · steady decode = 80 tokens (of 88, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   378.1 t/s
- decode    19.76 t/s
- accept    70.0%   tokens/iter 2.40

## GPU HW (decode-windowed, busy samples)
- sms_active     89.3%
- sm_issue       8.6%  ← stalled
- tensor_active  1.0%
- compute_warps  37.5%

## Time split
- wall 7262.8 ms · kernel 6901.5 ms (95.0%) · idle 361.3 ms (5.0%)  ·  trace decode 11.0 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_gate_up_mid_expert_tile8_row32_kernel occupancy-starved (96 regs → 25% occ, 12% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| moe_gate_up_mid_expert_tile8_row32_kernel | 28.0 | 1932.0 | 1720 | 96 | 25% | 19.3 | 30.8 | 6.2 | 12% |
| moe_down_expert_tile8_row32_kernel | 21.1 | 1458.6 | 1720 | 128 | 25% | 7.2 | 30.9 | 6.2 | 8% |
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 15.8 | 1087.1 | 12080 | 48 | 62% | 6.7 | 57.5 | 1.9 | 84% |
| cutlass·wmma·f16·32x32 | 8.2 | 565.3 | 1720 | — | — | 3.3 | 18.2 | — | — |
| moe_gate_up_mid_decode_q4K_qwarp32_kernel | 3.8 | 264.1 | 80 | 96 | 25% | 2.6 | 26.0 | 6.2 | 4% |
| cutlass·wmma·f16·16x16 | 3.5 | 242.9 | 10960 | — | — | 2.0 | 5.9 | — | — |
| matmul_q8_0_preq_warp8_kernel | 3.2 | 222.4 | 240 | 40 | 75% | 5.6 | 73.3 | — | — |
| attention_indexed_mixed_heads…ne_kernel<8, 16> | 3.1 | 211.1 | 840 | — | — | 8.9 | 8.5 | — | — |
| moe_down_q4K_sum6_qwarp32_kernel | 2.5 | 171.4 | 80 | 43 | 62% | 3.1 | 40.8 | 6.2 | 3% |
| indexer_scores_wmma128_kernel | 2.1 | 143.1 | 840 | 48 | 62% | 8.3 | 9.9 | — | — |
| attention_decode_mixed_kernel | 2.1 | 141.9 | 960 | 40 | 75% | 14.4 | 28.8 | — | — |
| rms_norm_plain_kernel | 0.9 | 62.5 | 3720 | 16 | 100% | 1.0 | 1.7 | 0.5 | 1% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1661.3 | argmax_kernel | rms_norm_plain_kernel |
| 1634.3 | argmax_kernel | rms_norm_plain_kernel |
| 1563.0 | argmax_kernel | rms_norm_plain_kernel |
| 1405.9 | argmax_kernel | rms_norm_plain_kernel |
| 1208.5 | argmax_kernel | rms_norm_plain_kernel |
| 775.7 | argmax_kernel | rms_norm_plain_kernel |
