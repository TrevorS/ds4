# ds4 gamut — prefill ctx8192 promessi

`GB10 sm_121a` · steady decode = 1 tokens (of 8, skipped 7 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   — t/s
- decode    — t/s

## GPU HW (decode-windowed, busy samples)
- sms_active     99.6%
- sm_issue       33.4%
- tensor_active  4.4%
- compute_warps  40.9%

## Time split
- wall 20550.2 ms · kernel 20044.8 ms (97.5%) · idle 505.4 ms (2.5%)  ·  trace decode 0.0 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> top lever: moe_down_expert_tile16_row2048_kernel occupancy-starved (232 regs → 12% occ, 0% of peak BW)

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| moe_gate_up_mid_expert_tile8_…pan_kernel<1024> | 33.0 | 6606.3 | 86 | — | — | 45.5 | 32.8 | 6.2 | 0% |
| moe_down_expert_tile16_row2048_kernel | 26.1 | 5228.8 | 86 | 232 | 12% | 24.8 | 15.9 | 6.2 | 0% |
| attention_indexed_mixed_heads…ne_kernel<8, 16> | 9.9 | 1980.9 | 42 | — | — | 62.5 | 66.8 | — | — |
| f32_to_f16_kernel | 3.5 | 704.3 | 1361 | 10 | 100% | 7.2 | 78.7 | — | — |
| nvjet_sm121_hss_mma_128x208x6…64_tmaAB_bz_TNNN | 3.5 | 696.5 | 214 | — | — | 7.2 | 16.7 | — | — |
| rms_norm_plain_kernel | 3.2 | 633.9 | 173 | 16 | 100% | 3.1 | 95.2 | 0.5 | 0% |
| hc_expand_kernel | 2.6 | 520.3 | 172 | 40 | 75% | 28.5 | 88.2 | — | — |
| cutlass·wmma·f16·64x256 | 2.2 | 438.4 | 86 | — | — | 8.5 | 8.3 | — | — |
| head_rms_norm_rope_tail_kernel | 1.9 | 379.1 | 86 | 34 | 88% | 21.8 | 94.3 | 0.5 | 0% |
| attention_decode_mixed_heads8_online_kernel | 1.8 | 370.5 | 22 | 71 | 38% | 54.7 | 49.9 | — | — |
| attention_pack_group_heads_f16_kernel | 1.5 | 308.5 | 86 | 24 | 100% | 23.7 | 81.8 | — | — |
| attention_static_mixed_heads8_online_kernel | 1.5 | 304.4 | 22 | 70 | 38% | 52.8 | 49.9 | — | — |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 13819.8 | dequant_q8_0_to_f16_kernel | attention_p…heads_f16_kernel |
| 6950.6 | cutlass·wmma·f16·16x16 | cublasLt::s…t, float, float, |
| 4235.1 | fill_f32_kernel | f32_to_f16_kernel |
| 3649.0 | attention_p…heads_f16_kernel | cutlass·wmma·f16·64x256 |
| 1818.7 | dsv4_qkv_rm…norm_rows_kernel | dequant_q8_0_to_f16_kernel |
| 1086.1 | f32_to_f16_kernel | nvjet_sm121…64_tmaAB_bz_TNNN |
