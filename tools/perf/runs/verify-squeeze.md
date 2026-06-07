# gamut · verify-squeeze  (GB10 sm_121a, decode)

## throughput
- prefill   4.7 t/s
- decode    20.03 t/s
- accept    73.3% (combined 73.6%)   tokens/iter 2.46
- mtp step  combined 119.64 ms

## windows · 165 tok (skip 8), wall 9498.5 ms · kernel 91.6% / idle 8.4%

## GPU HW (decode-windowed, busy samples)
- sms_active     91.4
- sm_issue       8.2  ← stalled
- tensor_active  0.1
- compute_warps  45.2

**verdict:** memory-latency-bound at moderate occupancy (not compute/tensor-bound); top lever: moe_gate_up_mid_expert_tile8_row32_kernel (19% time, occ 25%)

## top 12 kernels (decode)
| kernel | %t | ms | calls | avg µs | regs | occ | issue | class | %peakBW | stall |
| --- | --: | --: | --: | --: | --: | --: | --: | --- | --: | --- |
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 23.4 | 2039.0 | 23254 | 87.7 | 48 | 62.5 | 6.7 | i8-dp4a | 86.1 | — |
| moe_gate_up_mid_expert_tile8_row32_kernel | 18.8 | 1634.1 | 3354 | 487.2 | 96 | 25.0 | 15.5 | i8-dp4a | 28.5 | — |
| grouped_q8_0_a_preq_warp8_kernel | 18.4 | 1603.3 | 3553 | 451.2 | 48 | 62.5 | 5.8 | unknown | — | — |
| moe_down_expert_tile8_row32_kernel | 14.6 | 1273.6 | 3354 | 379.7 | 128 | 25.0 | 7.4 | i8-dp4a | 18.3 | — |
| cutlass·wmma·f16·16x16 | 5.5 | 481.7 | 21372 | 22.5 | — | — | 1.9 | f16-tc | — | — |
| matmul_q8_0_preq_warp8_kernel | 4.9 | 430.2 | 512 | 840.2 | 40 | 75.0 | 5.4 | unknown | — | — |
| attention_decode_mixed_kernel | 3.8 | 326.7 | 3553 | 92.0 | 40 | 75.0 | 12.4 | unknown | — | — |
| rms_norm_plain_kernel | 1.4 | 121.6 | 7341 | 16.6 | 16 | 100.0 | 1.0 | f32 | 0.8 | — |
| rms_norm_weight_kernel | 1.0 | 84.9 | 9669 | 8.8 | 18 | 100.0 | 2.6 | f32 | 1.6 | — |
| moe_gate_up_mid_q4K_qwarp32_kernel | 0.8 | 66.0 | 156 | 423.2 | 96 | 25.0 | 2.0 | i8-dp4a | 32.8 | — |
| moe_build_expert_tile_offsets_kernel | 0.6 | 53.8 | 3354 | 16.0 | 14 | 100.0 | 6.1 | unknown | — | — |
| quantize_q8_0_f32_kernel | 0.5 | 45.2 | 28573 | 1.6 | 20 | 100.0 | 5.5 | f32 | 8.8 | — |

## top launch gaps (host/sched idle)
- 10843.6 µs  argmax_kernel → embed_token_hc_kernel
- 10789.3 µs  argmax_kernel → embed_token_hc_kernel
- 10772.7 µs  argmax_kernel → embed_token_hc_kernel
- 10532.6 µs  argmax_kernel → embed_token_hc_kernel
- 10432.2 µs  argmax_kernel → embed_token_hc_kernel
- 10409.1 µs  argmax_kernel → embed_token_hc_kernel