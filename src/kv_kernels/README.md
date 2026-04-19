# `kv_kernels` (Phase F)

Package name **`kv_kernels`** (not `kernels`) to avoid clashing with Hugging Face Transformers’ internal `kernels` imports.

## Interfaces

| Component | Draft path | Target path |
|-----------|------------|-------------|
| Q·K on history | `qk_draft_hist_triton` / `qk_scores_draft_upper_only` | `qk_target_hist_triton` / `qk_scores_target_upper_plus_lower` |
| Packed layout | `packed_hist[s, d]` `uint8`: low nibble = lower code, high nibble = upper code | Same |
| Metadata | `k_scale_u`, `k_zp_u` shape `[S, n_g]` (channel groups) | + `k_scale_l`, `k_zp_l` for lower |

## Precision

- **Draft:** loads **high nibble only** + upper `(scale, zero_point)` per channel group.
- **Target:** `dequant(upper) + dequant(lower)` with separate lower metadata.

## Hardware

- Triton pack + attention kernels expect **CUDA**. CPU / MPS fall back to reference PyTorch.

## Decoder hook

`SpeculativeDecoderDenseHierarchical(..., kv_kernel_backend="reference"|"triton", validate_triton_kernels_at_start=True)` runs `validate_qk_kernels_cuda()` once; full transformer attention still uses Hugging Face until a custom forward calls these ops.

## Microbench

`PYTHONPATH=src python benchmarks/kernels_microbench.py`
