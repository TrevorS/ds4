# ds4 gamut — gb10-on-upstream @ c5b39429

`GB10 sm_121a` · steady decode = 40 tokens (of 48, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   408.9 t/s
- decode    16.32 t/s
- accept    54.5%   tokens/iter 2.09
- kvcache   52.2 MB

## GPU HW (decode-windowed, busy samples)
- sms_active     87.5%
- sm_issue       7.8%  ← stalled
- tensor_active  0.8%
- compute_warps  35.6%

## Time split
- wall 2167.7 ms · kernel 2001.3 ms (92.3%) · idle 166.4 ms (7.7%)  ·  trace decode 18.5 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_down_expert_tile8_row32_kernel occupancy-starved (168 regs → 12% occ, 15% of peak BW, stall: barrier 53%)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s · stall ← ncu)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW | stall |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 25.3 | 505.5 | 5738 | 48 | 62% | 6.7 | 57.9 | 1.9 | 86% | long_scb 59% |
| moe_down_expert_tile8_row32_kernel | 19.1 | 381.7 | 817 | 168 | 12% | 6.0 | 15.9 | 6.2 | 15% | barrier 53% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.1 | 322.0 | 817 | 96 | 25% | 19.4 | 29.8 | 6.2 | 35% | long_scb 50% |
| cutlass·wmma·f16·32x32 | 13.1 | 262.3 | 817 | — | — | 3.2 | 18.3 | — | — | — |
| matmul_q8_0_preq_warp8_kernel | 5.9 | 117.6 | 161 | 40 | 75% | 5.4 | 72.6 | — | — | — |
| cutlass·wmma·f16·16x16 | 5.7 | 114.7 | 5206 | — | — | 1.9 | 6.1 | — | — | — |
| attention_decode_mixed_kernel | 2.3 | 46.6 | 899 | 40 | 75% | 6.1 | 24.8 | — | — | — |
| rms_norm_plain_kernel | 1.5 | 30.6 | 1857 | 16 | 100% | 1.0 | 1.9 | 0.5 | 1% | — |
| rms_norm_weight_kernel | 1.0 | 20.6 | 2485 | 18 | 100% | 2.7 | 15.8 | 0.5 | 2% | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 0.9 | 17.1 | 164 | 40 | 75% | 4.4 | 52.4 | — | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 0.7 | 13.5 | 82 | 48 | 62% | 5.5 | 60.1 | — | — | — |
| moe_build_expert_tile_offsets_kernel | 0.7 | 13.1 | 817 | 14 | 100% | 11.4 | 24.4 | — | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1242.8 | argmax_kernel | embed_token_hc_kernel |
| 724.7 | argmax_kernel | embed_token_hc_kernel |
| 668.3 | argmax_kernel | embed_token_hc_kernel |
| 633.9 | argmax_kernel | embed_token_hc_kernel |
| 630.1 | argmax_kernel | embed_token_hc_kernel |
| 626.6 | argmax_kernel | embed_token_hc_kernel |
