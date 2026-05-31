# ds4 gamut — prefast

`GB10 sm_121a` · steady decode = 80 tokens (of 96, skipped 16 warmup) · roofline peak 236 GB/s (measured read)

## Throughput
- prefill   — t/s
- decode    — t/s

## GPU HW (decode-windowed, busy samples)

## Time split
- wall 6507.3 ms · kernel 6051.1 ms (93.0%) · idle 456.1 ms (7.0%)  ·  trace decode 12.3 t/s
  *(kernel = Σ durations; concurrent kernels not de-overlapped)*

## Verdict
> no clear single bottleneck

## Per-kernel (time ← plain trace · HW ← gb20b windowed · occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s)
| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |
|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|
| matmul_q8_0_preq_warp8_kernel | 12.7 | 768.5 | 3520 | — | — | — | — | — | — |
| matmul_f16_wmma_bi_kernel | 12.5 | 755.3 | 6800 | — | — | — | — | — | — |
| matmul_q8_0_hc_expand_preq_warp8_kernel | 12.0 | 728.0 | 6880 | — | — | — | — | — | — |
| moe_gate_up_mid_decode_lut_qwarp32_kernel | 9.8 | 590.9 | 3440 | — | — | — | — | 6.2 | 81% |
| grouped_q8_0_a_preq_warp8_kernel | 9.4 | 569.4 | 3440 | — | — | — | — | — | — |
| attention_indexed_mixed_kernel | 8.6 | 517.5 | 1680 | — | — | — | — | — | — |
| matmul_q8_0_pair_preq_warp8_kernel | 7.1 | 428.0 | 6880 | — | — | — | — | — | — |
| matmul_f16_pair_ordered_chunks_kernel | 7.0 | 423.6 | 4960 | — | — | — | — | — | — |
| moe_down_sum6_qwarp32_kernel | 6.0 | 360.8 | 3440 | — | — | — | — | 6.2 | 66% |
| matmul_f16_ordered_chunks_kernel | 4.7 | 283.6 | 6960 | — | — | — | — | — | — |
| attention_decode_mixed_kernel | 2.8 | 171.9 | 1760 | — | — | — | — | — | — |
| rms_norm_plain_kernel | 2.0 | 121.2 | 6960 | — | — | — | — | 0.5 | 1% |

## Top launch gaps (steady decode, host/scheduling idle)
| gap µs | prev kernel | next kernel |
|-------:|-------------|-------------|
| 411.2 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 394.2 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 389.2 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 389.2 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 384.5 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
| 377.8 | matmul_q8_0…req_warp8_kernel | embed_token_hc_kernel |
