# ds4 gamut — moe_down_lb3

`GB10 sm_121a` · steady decode = 40 tokens (of 48, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   7.2 t/s
- decode    16.47 t/s
- accept    54.5%   tokens/iter 2.09

## GPU HW (decode-windowed, busy samples)
- sms_active     91.9%
- sm_issue       6.8%  ← stalled
- tensor_active  0.7%
- compute_warps  43.0%

## Time split
- wall 2552.7 ms · kernel 2386.5 ms (93.5%) · idle 166.2 ms (6.5%)  ·  trace decode 15.7 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_gate_up_mid_expert_tile8_row32_kernel occupancy-starved (96 regs → 25% occ, 35% of peak BW, stall: long_scb 53%)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s · stall ← ncu)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW | stall |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|------:|
| moe_down_expert_tile8_row32_kernel | 31.1 | 742.7 | 817 | 80 | 38% | 3.9 | 46.5 | 6.2 | 8% | lg_throttle 36% |
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 22.0 | 526.1 | 5738 | 48 | 62% | 6.3 | 58.3 | 1.9 | 82% | long_scb 59% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 13.4 | 320.7 | 817 | 96 | 25% | 19.3 | 31.5 | 6.2 | 35% | long_scb 53% |
| cutlass·wmma·f16·32x32 | 11.2 | 267.3 | 817 | — | — | 3.2 | 18.3 | — | — | — |
| matmul_q8_0_preq_warp8_kernel | 4.9 | 117.0 | 161 | 40 | 75% | 5.3 | 73.0 | — | — | — |
| cutlass·wmma·f16·16x16 | 4.8 | 114.7 | 5206 | — | — | 1.9 | 6.1 | — | — | — |
| attention_decode_mixed_kernel | 1.9 | 46.2 | 899 | 40 | 75% | 6.2 | 24.7 | — | — | — |
| rms_norm_plain_kernel | 1.3 | 30.5 | 1857 | 16 | 100% | 1.0 | 1.8 | 0.5 | 1% | — |
| rms_norm_weight_kernel | 0.9 | 20.5 | 2485 | 18 | 100% | 2.7 | 17.0 | 0.5 | 2% | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 0.7 | 17.1 | 164 | 40 | 75% | 4.2 | 51.7 | — | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 0.6 | 13.5 | 82 | 48 | 62% | 5.4 | 59.8 | — | — | — |
| moe_build_expert_tile_offsets_kernel | 0.5 | 13.0 | 817 | 14 | 100% | 10.7 | 23.8 | — | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1380.1 | argmax_kernel | embed_token_hc_kernel |
| 744.0 | argmax_kernel | embed_token_hc_kernel |
| 636.6 | argmax_kernel | embed_token_hc_kernel |
| 629.7 | argmax_kernel | embed_token_hc_kernel |
| 624.8 | argmax_kernel | embed_token_hc_kernel |
| 615.2 | argmax_kernel | embed_token_hc_kernel |
