"""Hot-path kernels (Triton prototypes): packed KV append, draft/target Q·K on quantized history.

**Backends**

* ``reference`` — PyTorch dequant + matmul (baseline for :func:`kv_kernels.integration.validate_qk_kernels_cuda`).
* ``triton`` — explicit ``uint8`` nibble loads in attention; Triton elementwise pack.

**Integration**

* :class:`decoding.speculative_dense_hierarchical.SpeculativeDecoderDenseHierarchical` accepts
  ``kv_kernel_backend`` and optional ``validate_triton_kernels_at_start``. Full model attention still
  uses Hugging Face until a custom forward wires these kernels in.
"""

from .append_hist import append_hist_packed_buffer, quantize_and_pack_key_hist_chunk
from .backend import KVKernelBackend, normalize_backend
from .integration import validate_qk_kernels_cuda
from .reference_attention import (
    pack_upper_lower_int4,
    qk_scores_draft_upper_only,
    qk_scores_target_upper_plus_lower,
)
from .triton_attention import qk_draft_dispatch, qk_target_dispatch, qk_draft_hist_triton, qk_target_hist_triton
from .triton_pack import append_packed_concat, pack_int4_pair_triton
from .triton_runtime import triton_available, require_triton
from .fused_draft_decode import (
    fused_draft_decode_attention,
    fused_draft_decode_attention_reference,
    fused_draft_decode_attention_reference_two_pass,
)
from .fused_verifier_block_attention import (
    fused_verifier_block_attention,
    fused_verifier_block_attention_reference,
    fused_verifier_block_attention_reference_logits,
)
from .fused_cache_mutation import append_quantized_cf1_to_hist, quantize_cf1_fp16_to_int4
from .runtime_options import RuntimePerfFlags, flags_from_env
from .tuning import (
    KernelTuningConfig,
    active_kernel_tuning,
    get_preset_config,
    kernel_tuning_scope,
    list_tuning_profiles,
    resolve_tuning_profile,
    set_active_kernel_tuning,
    set_kernel_tuning_from_spec,
)
from .parity_harness import (
    DEFAULT_LAYERWISE_TOLERANCES,
    ParityTolerances,
    TensorParityReport,
    assert_tensor_parity,
    tensor_parity_report,
    triage_fused_verifier_mismatch,
    triage_json_dumps,
)

__all__ = [
    "KVKernelBackend",
    "append_hist_packed_buffer",
    "append_packed_concat",
    "normalize_backend",
    "pack_int4_pair_triton",
    "pack_upper_lower_int4",
    "qk_draft_dispatch",
    "qk_draft_hist_triton",
    "qk_scores_draft_upper_only",
    "qk_scores_target_upper_plus_lower",
    "qk_target_dispatch",
    "qk_target_hist_triton",
    "quantize_and_pack_key_hist_chunk",
    "require_triton",
    "triton_available",
    "validate_qk_kernels_cuda",
    "fused_draft_decode_attention",
    "fused_draft_decode_attention_reference",
    "fused_draft_decode_attention_reference_two_pass",
    "fused_verifier_block_attention",
    "fused_verifier_block_attention_reference",
    "fused_verifier_block_attention_reference_logits",
    "append_quantized_cf1_to_hist",
    "quantize_cf1_fp16_to_int4",
    "DEFAULT_LAYERWISE_TOLERANCES",
    "ParityTolerances",
    "TensorParityReport",
    "assert_tensor_parity",
    "tensor_parity_report",
    "triage_fused_verifier_mismatch",
    "triage_json_dumps",
    "KernelTuningConfig",
    "RuntimePerfFlags",
    "active_kernel_tuning",
    "flags_from_env",
    "get_preset_config",
    "kernel_tuning_scope",
    "list_tuning_profiles",
    "resolve_tuning_profile",
    "set_active_kernel_tuning",
    "set_kernel_tuning_from_spec",
]
