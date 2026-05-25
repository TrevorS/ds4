# ds4 gamut — moe_down_lb2

`GB10 sm_121a` · steady decode = 40 tokens (of 48, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   8.0 t/s
- decode    19.69 t/s
- accept    54.5%   tokens/iter 2.09

## GPU HW (decode-windowed, busy samples)
- sms_active     89.6%
- sm_issue       8.1%  ← stalled
- tensor_active  0.9%
- compute_warps  39.2%

## Time split
- wall 2099.0 ms · kernel 1932.7 ms (92.1%) · idle 166.2 ms (7.9%)  ·  trace decode 19.1 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_gate_up_mid_expert_tile8_row32_kernel occupancy-starved (96 regs → 25% occ, 35% of peak BW, stall: long_scb 50%)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s · stall ← ncu)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW | stall |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 26.4 | 510.7 | 5738 | 48 | 62% | 6.6 | 58.0 | 1.9 | 85% | long_scb 60% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.7 | 322.6 | 817 | 96 | 25% | 19.2 | 30.7 | 6.2 | 35% | long_scb 50% |
| moe_down_expert_tile8_row32_kernel | 15.9 | 306.8 | 817 | 128 | 25% | 7.7 | 30.8 | 6.2 | 18% | barrier 56% |
| cutlass·wmma·f16·32x32 | 13.7 | 264.4 | 817 | — | — | 3.2 | 18.2 | — | — | — |
| matmul_q8_0_preq_warp8_kernel | 6.1 | 117.2 | 161 | 40 | 75% | 5.3 | 72.8 | — | — | — |
| cutlass·wmma·f16·16x16 | 5.9 | 114.6 | 5206 | — | — | 2.0 | 6.1 | — | — | — |
| attention_decode_mixed_kernel | 2.4 | 46.2 | 899 | 40 | 75% | 6.2 | 24.8 | — | — | — |
| rms_norm_plain_kernel | 1.6 | 30.5 | 1857 | 16 | 100% | 1.0 | 1.7 | 0.5 | 1% | — |
| rms_norm_weight_kernel | 1.1 | 20.5 | 2485 | 18 | 100% | 2.5 | 15.0 | 0.5 | 2% | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 0.9 | 17.1 | 164 | 40 | 75% | 4.3 | 52.2 | — | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 0.7 | 13.4 | 82 | 48 | 62% | 5.5 | 59.8 | — | — | — |
| moe_gate_up_mid_decode_q4K_qwarp32_kernel | 0.7 | 13.0 | 39 | 96 | 25% | 2.5 | 25.8 | 6.2 | 42% | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1299.6 | argmax_kernel | embed_token_hc_kernel |
| 728.1 | argmax_kernel | embed_token_hc_kernel |
| 665.1 | argmax_kernel | embed_token_hc_kernel |
| 632.8 | argmax_kernel | embed_token_hc_kernel |
| 626.7 | argmax_kernel | embed_token_hc_kernel |
| 624.1 | argmax_kernel | embed_token_hc_kernel |
