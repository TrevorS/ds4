# gamut · readme-headline  (GB10 sm_121a, decode)

## throughput
- prefill   9.3 t/s
- decode    19.41 t/s
- accept    56.2% (combined 56.2%)   tokens/iter 2.11
- mtp step  combined 108.13 ms

## windows · 91 tok (skip 8), wall 4669.5 ms · kernel 92.4% / idle 7.6%

## GPU HW (decode-windowed, busy samples)
- sms_active     90.8
- sm_issue       8.8  ← stalled
- tensor_active  0.1
- compute_warps  46.0

**verdict:** memory-latency-bound at moderate occupancy (not compute/tensor-bound); top lever: moe_gate_up_mid_expert_tile8_row32_kernel (16% time, occ 38%)

## top 12 kernels (decode)
| kernel | %t | ms | calls | avg µs | regs | occ | issue | class | %peakBW | stall |
| --- | --: | --: | --: | --: | --: | --: | --: | --- | --: | --- |
| matmul_q8_0_preq_batch_share_warp_kernel<3> | 24.8 | 1071.1 | 12382 | 86.5 | 48 | 62.5 | 6.8 | i8-dp4a | 87.3 | — |
| grouped_q8_0_a_preq_warp8_kernel | 19.1 | 824.7 | 1889 | 436.6 | 48 | 62.5 | 6.2 | unknown | — | — |
| moe_gate_up_mid_expert_tile8_row32_kernel | 16.3 | 704.7 | 1806 | 390.2 | 78 | 37.5 | 19.9 | i8-dp4a | 35.5 | — |
| moe_down_expert_tile8_row32_kernel | 15.6 | 672.6 | 1806 | 372.4 | 128 | 25.0 | 7.7 | i8-dp4a | 18.6 | — |
| cutlass·wmma·f16·16x16 | 5.7 | 247.4 | 11508 | 21.5 | — | — | 2.0 | f16-tc | — | — |
| matmul_q8_0_preq_warp8_kernel | 4.9 | 213.0 | 249 | 855.4 | 40 | 75.0 | 5.8 | unknown | — | — |
| attention_decode_mixed_kernel | 2.9 | 123.8 | 1889 | 65.6 | 40 | 75.0 | 8.6 | unknown | — | — |
| rms_norm_plain_kernel | 1.5 | 64.5 | 3903 | 16.5 | 16 | 100.0 | 1.0 | f32 | 0.8 | — |
| rms_norm_weight_kernel | 1.0 | 44.7 | 5163 | 8.7 | 18 | 100.0 | 2.7 | f32 | 1.6 | — |
| moe_build_expert_tile_offsets_kernel | 0.7 | 29.0 | 1806 | 16.1 | 14 | 100.0 | 11.1 | unknown | — | — |
| moe_gate_up_mid_q4K_qwarp32_kernel | 0.6 | 27.3 | 83 | 329.0 | 96 | 25.0 | 2.5 | i8-dp4a | 42.1 | — |
| matmul_q8_0_preq_batch_share_warp_kernel<2> | 0.6 | 25.4 | 302 | 84.2 | 48 | 62.5 | 5.6 | i8-dp4a | 89.7 | — |

## top launch gaps (host/sched idle)
- 1571.8 µs  argmax_kernel → rms_norm_plain_kernel
- 1464.1 µs  argmax_kernel → rms_norm_plain_kernel
- 1324.4 µs  argmax_kernel → rms_norm_plain_kernel
- 1181.2 µs  argmax_kernel → rms_norm_plain_kernel
- 752.3 µs  argmax_kernel → rms_norm_plain_kernel
- 652.2 µs  argmax_kernel → embed_token_hc_kernel