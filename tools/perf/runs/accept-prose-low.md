# ds4 gamut — accept-prose-low

`GB10 sm_121a` · steady decode = 95 tokens (of 103, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   55.5 t/s
- decode    17.49 t/s
- accept    43.6%   tokens/iter 1.86

## GPU HW (decode-windowed, busy samples)
- sms_active     89.8%
- sm_issue       8.4%  ← stalled
- tensor_active  0.9%
- compute_warps  38.6%

## Time split
- wall 5254.5 ms · kernel 4822.0 ms (91.8%) · idle 432.5 ms (8.2%)  ·  trace decode 18.1 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_down_expert_tile8_row32_kernel occupancy-starved (128 regs → 25% occ, 18% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 26.3 | 1270.2 | 14194 | 48 | 62% | 6.6 | 57.7 | 1.9 | 84% |
| moe_down_expert_tile8_row32_kernel | 16.8 | 808.7 | 2064 | 128 | 25% | 7.4 | 30.8 | 6.2 | 18% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.7 | 803.1 | 2064 | 96 | 25% | 19.5 | 30.6 | 6.2 | 36% |
| cutlass·wmma·f16·32x32 | 13.7 | 659.9 | 2064 | — | — | 3.3 | 18.3 | — | — |
| cutlass·wmma·f16·16x16 | 6.0 | 288.4 | 13152 | — | — | 2.0 | 6.0 | — | — |
| matmul_q8_0_preq_warp8_kernel | 5.3 | 256.5 | 285 | 40 | 75% | 5.4 | 73.2 | — | — |
| attention_decode_mixed_kernel | 3.9 | 189.6 | 2159 | 40 | 75% | 10.3 | 27.0 | — | — |
| rms_norm_plain_kernel | 1.5 | 73.7 | 4461 | 16 | 100% | 1.0 | 1.9 | 0.5 | 1% |
| rms_norm_weight_kernel | 1.1 | 51.8 | 5867 | 18 | 100% | 2.7 | 15.9 | 0.5 | 2% |
| moe_build_expert_tile_offsets_kernel | 0.7 | 33.4 | 2064 | 14 | 100% | 10.9 | 24.0 | — | — |
| moe_gate_up_mid_decode_q4K_qwarp32_kernel | 0.7 | 31.4 | 95 | 96 | 25% | 2.5 | 26.0 | 6.2 | 42% |
| matmul_q8_0_preq_batch_share_warp_kernel<2> | 0.6 | 26.5 | 302 | 48 | 62% | 5.5 | 57.5 | 1.9 | 86% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1688.7 | argmax_kernel | embed_token_hc_kernel |
| 1610.5 | argmax_kernel | rms_norm_plain_kernel |
| 1564.9 | argmax_kernel | rms_norm_plain_kernel |
| 1496.4 | argmax_kernel | rms_norm_plain_kernel |
| 1310.1 | argmax_kernel | rms_norm_plain_kernel |
| 956.5 | argmax_kernel | rms_norm_plain_kernel |
