"""INT4 grouped packing roundtrip (CPU)."""

from __future__ import annotations

import torch

from mlsys_kv.cache.quantization_int4 import (
    _DEFAULT_GROUP,
    symmetric_dequantize_int4_grouped_packed,
    symmetric_quantize_int4_grouped_packed,
)


def test_int4_quantize_dequantize_roundtrip() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 64, dtype=torch.float32)
    packed, scales, pad_amt, orig_shape = symmetric_quantize_int4_grouped_packed(
        x, group_size=_DEFAULT_GROUP
    )
    assert packed.dtype == torch.uint8
    assert pad_amt == 0
    y = symmetric_dequantize_int4_grouped_packed(
        packed,
        scales,
        original_shape=orig_shape,
        pad_amt=pad_amt,
        group_size=_DEFAULT_GROUP,
        out_dtype=x.dtype,
        out_device=x.device,
    )
    assert y.shape == x.shape
    max_err = (y - x).abs().max().item()
    assert max_err < 0.35


def test_int4_padded_dim_unpacks_to_original_shape() -> None:
    x = torch.randn(1, 1, 2, 50, dtype=torch.float16)
    packed, scales, pad_amt, orig_shape = symmetric_quantize_int4_grouped_packed(
        x, group_size=_DEFAULT_GROUP
    )
    assert pad_amt > 0
    y = symmetric_dequantize_int4_grouped_packed(
        packed,
        scales,
        original_shape=orig_shape,
        pad_amt=pad_amt,
        group_size=_DEFAULT_GROUP,
        out_dtype=x.dtype,
        out_device=x.device,
    )
    assert y.shape == (1, 1, 2, 50)
