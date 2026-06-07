# ds4 gamut — accept-code-high

`GB10 sm_121a` · steady decode = 75 tokens (of 83, skipped 8 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   67.2 t/s
- decode    21.68 t/s
- accept    68.8%   tokens/iter 2.38

## GPU HW (decode-windowed, busy samples)
- sms_active     89.8%
- sm_issue       8.4%  ← stalled
- tensor_active  0.9%
- compute_warps  38.6%

## Time split
- wall 4175.5 ms · kernel 3853.3 ms (92.3%) · idle 322.2 ms (7.7%)  ·  trace decode 18.0 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> memory-latency-bound at moderate occupancy (not compute/tensor-bound). top lever: moe_down_expert_tile8_row32_kernel occupancy-starved (128 regs → 25% occ, 17% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 26.0 | 1001.8 | 11174 | 48 | 62% | 6.6 | 57.5 | 1.9 | 84% |
| moe_down_expert_tile8_row32_kernel | 17.0 | 654.5 | 1591 | 128 | 25% | 7.3 | 30.9 | 6.2 | 17% |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.8 | 648.6 | 1591 | 96 | 25% | 19.4 | 30.7 | 6.2 | 34% |
| cutlass·wmma·f16·32x32 | 13.2 | 508.7 | 1591 | — | — | 3.2 | 18.1 | — | — |
| cutlass·wmma·f16·16x16 | 5.8 | 223.2 | 10138 | — | — | 1.9 | 6.0 | — | — |
| matmul_q8_0_preq_warp8_kernel | 5.5 | 211.8 | 266 | 40 | 75% | 5.6 | 73.1 | — | — |
| attention_decode_mixed_kernel | 4.0 | 155.0 | 1708 | 40 | 75% | 10.7 | 27.0 | — | — |
| rms_norm_plain_kernel | 1.5 | 58.4 | 3528 | 16 | 100% | 1.0 | 1.8 | 0.5 | 1% |
| rms_norm_weight_kernel | 1.0 | 40.3 | 4616 | 18 | 100% | 2.7 | 16.3 | 0.5 | 2% |
| moe_build_expert_tile_offsets_kernel | 0.7 | 25.4 | 1591 | 14 | 100% | 11.0 | 23.9 | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 0.6 | 24.3 | 234 | 40 | 75% | 4.4 | 52.4 | — | — |
| moe_gate_up_mid_decode_q4K_qwarp32_kernel | 0.6 | 24.2 | 74 | 96 | 25% | 2.5 | 26.0 | 6.2 | 42% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 1314.5 | argmax_kernel | rms_norm_plain_kernel |
| 1237.8 | argmax_kernel | rms_norm_plain_kernel |
| 988.7 | argmax_kernel | rms_norm_plain_kernel |
| 767.6 | argmax_kernel | rms_norm_plain_kernel |
| 701.2 | argmax_kernel | embed_token_hc_kernel |
| 691.5 | argmax_kernel | embed_token_hc_kernel |
