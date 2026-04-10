"""Human-readable mode labels for plots/tables (honest quantization semantics)."""

from __future__ import annotations

# Display names: quantization is memory-only unless runtime_accelerated appears in metadata.
BENCHMARK_LABEL_DISPLAY: dict[str, str] = {
    "ar": "AR (baseline)",
    "spec_fp16": "Spec FP16 draft",
    "spec_quant_memonly": "Spec + INT KV (Memory-Only)",
    "spec_sparse": "Spec + Sparse draft",
    "spec_sparse_quant_memonly": "Spec + Sparse + INT KV (Memory-Only)",
}


def display_benchmark_label(label: str, *, quantization_type: str | None = None) -> str:
    """Append explicit memory-only note when CSV says ``memory_only``."""

    base = BENCHMARK_LABEL_DISPLAY.get(str(label), str(label))
    if quantization_type == "memory_only" and "Memory-Only" not in base:
        return f"{base} [Memory-Only KV]"
    return base


QUANT_HONESTY_FOOTNOTE = (
    "**Memory-only INT KV:** weights/KV are packed for footprint; attention consumes **dequantized** "
    "activations. Lack of speedup vs FP16 draft is consistent with **dequant + standard attention** "
    "overhead, not a failure of the memory story."
)
