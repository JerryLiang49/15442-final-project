"""Phase 15: strict labeled grid and schema helpers."""

from __future__ import annotations

import pytest

from mlsys_kv.benchmarks.experiment_runner import expand_sweep_grid
from mlsys_kv.benchmarks.experiment_schema import (
    SWEEP_MODE_AUTOREGRESSIVE,
    SWEEP_MODE_QUANT_ONLY,
    SWEEP_MODE_SPARSE_ONLY,
    SWEEP_MODE_SPARSE_QUANT,
    SWEEP_MODE_SPECULATIVE_FP16,
    benchmark_label_for_canonical_mode,
    canonical_sweep_mode,
    quantization_type_for_row,
)


def test_canonical_aliases_round_trip() -> None:
    assert canonical_sweep_mode("ar") == SWEEP_MODE_AUTOREGRESSIVE
    assert canonical_sweep_mode("spec_sparse_quant_memonly") == SWEEP_MODE_SPARSE_QUANT
    assert benchmark_label_for_canonical_mode(SWEEP_MODE_QUANT_ONLY) == "spec_quant_memonly"


def test_quantization_type_row() -> None:
    assert quantization_type_for_row(canonical_mode=SWEEP_MODE_AUTOREGRESSIVE, quant_bits_effective=-1) == "none"
    assert quantization_type_for_row(canonical_mode=SWEEP_MODE_SPECULATIVE_FP16, quant_bits_effective=16) == "none"
    assert quantization_type_for_row(canonical_mode=SWEEP_MODE_QUANT_ONLY, quant_bits_effective=4) == "memory_only"
    assert quantization_type_for_row(canonical_mode=SWEEP_MODE_QUANT_ONLY, quant_bits_effective=16) == "none"


def test_expand_strict_no_duplicate_ar_quant_axis() -> None:
    modes = [
        SWEEP_MODE_AUTOREGRESSIVE,
        SWEEP_MODE_SPECULATIVE_FP16,
        SWEEP_MODE_QUANT_ONLY,
    ]
    g = expand_sweep_grid(modes, [1, 2], [0.1, 0.9], [4, 8], strict_labeled_grid=True)
    ar = [c for c in g if c[0] == SWEEP_MODE_AUTOREGRESSIVE]
    assert len(ar) == 1
    assert ar[0][1:] == (0, 0.0, -1)
    fp = [c for c in g if c[0] == SWEEP_MODE_SPECULATIVE_FP16]
    assert len(fp) == 2
    assert all(c[2] == 0.0 and c[3] == 16 for c in fp)
    qo = [c for c in g if c[0] == SWEEP_MODE_QUANT_ONLY]
    assert len(qo) == 2 * 2


def test_expand_sparse_quant_product() -> None:
    g = expand_sweep_grid(
        [SWEEP_MODE_SPARSE_QUANT],
        [1],
        [0.5],
        [4],
        strict_labeled_grid=True,
    )
    assert g == [(SWEEP_MODE_SPARSE_QUANT, 1, 0.5, 4)]


def test_expand_non_strict_factorial_size() -> None:
    modes = [SWEEP_MODE_AUTOREGRESSIVE, SWEEP_MODE_SPECULATIVE_FP16]
    g = expand_sweep_grid(modes, [1], [0.0], [8], strict_labeled_grid=False)
    assert len(g) == 2


@pytest.mark.parametrize(
    "raw",
    ["spec_quant_runtime", "spec_sparse_quant_runtime"],
)
def test_runtime_labels_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        canonical_sweep_mode(raw)
