# ds4 gamut — postfast

`GB10 sm_121a` · steady decode = 80 tokens (of 96, skipped 16 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   — t/s
- decode    — t/s

## GPU HW (decode-windowed, busy samples)

## Time split
- wall 6904.6 ms · kernel 6489.5 ms (94.0%) · idle 415.1 ms (6.0%)  ·  trace decode 11.6 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> no clear single bottleneck

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_warp8_kernel | 15.5 | 1008.6 | 3520 | — | — | — | — | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 14.3 | 925.3 | 6880 | — | — | — | — | — | — |
| grouped_q8_0_a_preq_warp8_kernel | 10.8 | 698.9 | 3440 | — | — | — | — | — | — |
| moe_gate_up_mid_decode_lut_qwarp32_kernel | 9.7 | 629.0 | 3440 | — | — | — | — | 6.2 | 76% |
| matmul_f16_wmma_bi_kernel | 8.9 | 576.7 | 6800 | — | — | — | — | — | — |
| matmul_q8_0_pair_preq_warp8_kernel | 7.8 | 503.6 | 6880 | — | — | — | — | — | — |
| attention_indexed_mixed_kernel | 7.6 | 493.4 | 1680 | — | — | — | — | — | — |
| matmul_f16_pair_ordered_chunks_kernel | 7.4 | 480.5 | 4960 | — | — | — | — | — | — |
| moe_down_sum6_qwarp32_kernel | 5.5 | 355.2 | 3440 | — | — | — | — | 6.2 | 67% |
| matmul_f16_ordered_chunks_kernel | 3.5 | 226.8 | 6960 | — | — | — | — | — | — |
| attention_decode_mixed_kernel | 2.4 | 158.0 | 1760 | — | — | — | — | — | — |
| rms_norm_plain_kernel | 1.8 | 116.3 | 6960 | — | — | — | — | 0.5 | 1% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 395.2 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 394.6 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 393.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 386.0 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 376.1 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 375.5 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
