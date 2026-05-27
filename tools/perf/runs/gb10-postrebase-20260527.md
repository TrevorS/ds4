# ds4 gamut — gb10-postrebase-20260527

`GB10 sm_121a` · steady decode = 40 tokens (of 48, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   8.0 t/s
- decode    19.61 t/s
- accept    54.5%   tokens/iter 2.09

## GPU HW (decode-windowed, busy samples)
- sms_active     89.7%
- sm_issue       8.1%  ← stalled
- tensor_active  0.9%
- compute_warps  39.0%

## Time split
- wall 2119.2 ms · kernel 1948.8 ms (92.0%) · idle 170.4 ms (8.0%)  ·  trace decode 18.9 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_down_expert_tile8_row32_kernel occupancy-starved (128 regs → 25% occ, 17% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 26.2 | 511.0 | 5738 | 48 | 62% | 6.6 | 57.8 | 1.9 | 85% |
| moe_down_expert_tile8_row32_kernel | 16.7 | 324.9 | 817 | 128 | 25% | 7.3 | 30.9 | 6.2 | 17% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.6 | 323.9 | 817 | 96 | 25% | 19.4 | 30.7 | 6.2 | 35% |
| cutlass·wmma·f16·32x32 | 13.3 | 258.5 | 817 | — | — | 3.2 | 18.5 | — | — |
| matmul_q8_0_preq_warp8_kernel | 6.0 | 117.3 | 161 | 40 | 75% | 5.4 | 72.9 | — | — |
| cutlass·wmma·f16·16x16 | 5.8 | 113.8 | 5206 | — | — | 1.9 | 6.0 | — | — |
| attention_decode_mixed_kernel | 2.4 | 46.3 | 899 | 40 | 75% | 6.2 | 24.5 | — | — |
| rms_norm_plain_kernel | 1.6 | 30.2 | 1857 | 16 | 100% | 1.0 | 1.8 | 0.5 | 1% |
| rms_norm_weight_kernel | 1.1 | 20.6 | 2485 | 18 | 100% | 2.7 | 16.6 | 0.5 | 2% |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 0.9 | 17.1 | 164 | 40 | 75% | 4.5 | 51.7 | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 0.7 | 13.5 | 82 | 48 | 62% | 5.5 | 60.6 | — | — |
| moe_build_expert_tile_offsets_kernel | 0.7 | 13.1 | 817 | 14 | 100% | 10.8 | 23.8 | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 912.1 | argmax_kernel | rms_norm_plain_kernel |
| 686.2 | argmax_kernel | embed_token_hc_kernel |
| 684.8 | argmax_kernel | rms_norm_plain_kernel |
| 682.8 | argmax_kernel | embed_token_hc_kernel |
| 679.4 | argmax_kernel | embed_token_hc_kernel |
| 678.7 | argmax_kernel | embed_token_hc_kernel |
