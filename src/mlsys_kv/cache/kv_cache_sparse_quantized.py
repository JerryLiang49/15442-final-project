"""Joint draft KV: **sparsify first** (recent + heavy hitters), then **INT8-quantize** retained K/V only.

**Composition order** (required for report / benchmarks)

1. Clone full HF ``past_key_values`` from one decode step.
2. Run the same token-level retention policy as :class:`~mlsys_kv.cache.kv_cache_sparse.KVCacheSparse`
   (scores, periodic refresh, ``select_retained_token_indices``).
3. :func:`~mlsys_kv.cache.kv_cache_sparse.gather_retained_kv_layers` — gather K/V to retained length ``R``.
4. Apply :func:`~mlsys_kv.cache.quantization.symmetric_quantize_int8` **per tensor** (each gathered
   ``keys`` / ``values``), identical to :class:`~mlsys_kv.cache.kv_cache_quantized.KVCacheQuantized`,
   storing **only** ``int8`` codes + **FP32 scale** (×2 per layer for K and V).

**Memory accounting** (:meth:`memory_bytes`, :meth:`stats`)

* **payload_bytes_int8**: retained key/value quantization codes (one byte per stored element).
* **metadata_bytes_sparse**: global ``int64`` retained index list, optional FP32 score buffer, fixed
  refresh bookkeeping (see sparse cache).
* **metadata_bytes_quant**: two FP32 scales per quantized layer, plus sliding-window auxiliary tensor
  bytes when present (matches quant-only cache).

The **verifier** stays full FP16; lossy draft does not change committed tokens under greedy verification.

Combined **runtime** overhead in experiments: ``cumulative_refresh_time_s`` (selector) +
``cumulative_dequant_time_s`` (rebuild ``past_key_values`` for the next forward), surfaced via
:class:`~mlsys_kv.decoding.speculative.SpeculativeMetrics` and :meth:`stats`.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mlsys_kv.cache.hf_kv_clone import (
    clone_past_key_values,
    _clone_dynamic_layer,
    past_sequence_length,
    strip_last_position_from_past,
)
from mlsys_kv.cache.heavy_hitter_selector import (
    SparseRetentionConfig,
    attention_mass_from_last_token,
    build_full_length_scores_from_attention_prefix,
    key_norm_token_scores,
    select_retained_token_indices,
)
from mlsys_kv.cache.kv_cache_base import KVCacheBase
from mlsys_kv.cache.kv_cache_quantized import _QuantKVPair, _QuantSlidePair
from mlsys_kv.cache.kv_cache_sparse import gather_retained_kv_layers
from mlsys_kv.cache.quantization import symmetric_dequantize_int8, symmetric_quantize_int8


class KVCacheSparseQuantized(KVCacheBase):
    """Draft-only: SnapKV-style retention → symmetric INT8 on retained tensors only."""

    __slots__ = (
        "_config",
        "_model",
        "_fmt",
        "_tuple_entries",
        "_dynamic_entries",
        "_full_seq_len",
        "_retained_indices",
        "_token_scores_cpu",
        "_append_calls",
        "_last_token",
        "_cumulative_refresh_s",
        "_refresh_events",
        "_sum_sparsity",
        "_sparsity_samples",
        "_cumulative_dequant_s",
    )

    def __init__(self, config: SparseRetentionConfig, model: PreTrainedModel | None = None) -> None:
        self._config = config
        self._model = model
        self._fmt: str | None = None
        self._tuple_entries: list[_QuantKVPair] | None = None
        self._dynamic_entries: list[_QuantKVPair | _QuantSlidePair | Any] | None = None

        self._full_seq_len: int = 0
        self._retained_indices: list[int] = []
        self._token_scores_cpu: torch.Tensor | None = None

        self._append_calls: int = 0
        self._last_token: torch.Tensor | None = None

        self._cumulative_refresh_s: float = 0.0
        self._refresh_events: int = 0

        self._sum_sparsity: float = 0.0
        self._sparsity_samples: int = 0

        self._cumulative_dequant_s: float = 0.0

    def note_forward_token(self, token_ids: torch.Tensor) -> None:
        self._last_token = token_ids.detach()

    def append_from_forward_output(self, past_key_values: Any) -> None:
        full = clone_past_key_values(past_key_values)
        if full is None:
            self._fmt = None
            self._tuple_entries = None
            self._dynamic_entries = None
            self._full_seq_len = 0
            self._retained_indices = []
            return

        L = past_sequence_length(full)
        self._full_seq_len = L

        need_refresh = self._token_scores_cpu is None or (
            self._append_calls % self._config.refresh_interval == 0
        )
        if need_refresh:
            t0 = time.perf_counter()
            self._token_scores_cpu = self._compute_scores_cpu(full, L).cpu()
            self._cumulative_refresh_s += time.perf_counter() - t0
            self._refresh_events += 1
        elif self._token_scores_cpu is not None:
            old = self._token_scores_cpu
            if old.shape[0] < L:
                padded = torch.zeros(L, dtype=old.dtype, device=old.device)
                c = int(old.shape[0])
                padded[:c] = old
                padded[L - 1] = torch.tensor(float("inf"), dtype=old.dtype, device=old.device)
                self._token_scores_cpu = padded
            elif old.shape[0] > L:
                self._token_scores_cpu = old[:L].clone()

        assert self._token_scores_cpu is not None
        idx = select_retained_token_indices(
            L,
            self._token_scores_cpu,
            recent_window=self._config.recent_window,
            heavy_hitter_budget=self._config.heavy_hitter_budget,
        )
        self._retained_indices = idx
        if L > 0:
            self._sum_sparsity += 1.0 - (len(idx) / float(L))
            self._sparsity_samples += 1

        # --- Order: sparsify (gather) then quantize retained tensors only ---
        fmt, tup_fp16, dyn_fp16 = gather_retained_kv_layers(full, idx)
        self._fmt = fmt
        if fmt == "tuple":
            self._dynamic_entries = None
            assert tup_fp16 is not None
            self._tuple_entries = []
            for k, v in tup_fp16:
                qk, sk = symmetric_quantize_int8(k)
                qv, sv = symmetric_quantize_int8(v)
                self._tuple_entries.append(
                    _QuantKVPair(
                        qk=qk,
                        qv=qv,
                        scale_k=sk,
                        scale_v=sv,
                        kv_dtype=k.dtype,
                        kv_device=k.device,
                    )
                )
        else:
            assert fmt == "dynamic"
            self._tuple_entries = None
            assert dyn_fp16 is not None
            self._dynamic_entries = []
            for layer in dyn_fp16:
                if isinstance(layer, DynamicSlidingWindowLayer):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        qk, sk = symmetric_quantize_int8(k)
                        qv, sv = symmetric_quantize_int8(v)
                        self._dynamic_entries.append(
                            _QuantSlidePair(
                                qk=qk,
                                qv=qv,
                                scale_k=sk,
                                scale_v=sv,
                                kv_dtype=k.dtype,
                                kv_device=k.device,
                                cumulative_length=int(layer.cumulative_length),
                                sliding_window=int(layer.sliding_window),
                                sliding_window_tensor_cpu=layer._sliding_window_tensor.detach().cpu(),
                            )
                        )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                elif isinstance(layer, DynamicLayer):
                    if layer.is_initialized and layer.keys is not None:
                        k, v = layer.keys, layer.values
                        qk, sk = symmetric_quantize_int8(k)
                        qv, sv = symmetric_quantize_int8(v)
                        self._dynamic_entries.append(
                            _QuantKVPair(
                                qk=qk,
                                qv=qv,
                                scale_k=sk,
                                scale_v=sv,
                                kv_dtype=k.dtype,
                                kv_device=k.device,
                            )
                        )
                    else:
                        self._dynamic_entries.append(_clone_dynamic_layer(layer))
                else:
                    self._dynamic_entries.append(_clone_dynamic_layer(layer))

        self._append_calls += 1

    def _compute_scores_cpu(self, full_past: Any, L: int) -> torch.Tensor:
        if L == 0:
            return torch.zeros(0)

        use_attention = (
            self._config.scoring == "attention"
            and self._model is not None
            and self._last_token is not None
            and L > 1
        )
        if use_attention:
            try:
                prefix = strip_last_position_from_past(full_past)
                if past_sequence_length(prefix) == 0:
                    return key_norm_token_scores(full_past)
                s_pre = attention_mass_from_last_token(self._model, prefix, self._last_token)
                return build_full_length_scores_from_attention_prefix(s_pre, total_len=L)
            except Exception:
                return key_norm_token_scores(full_past)

        return key_norm_token_scores(full_past)

    def get_attention_kv(self) -> Any | None:
        if self._fmt is None:
            return None
        t0 = time.perf_counter()
        try:
            if self._fmt == "tuple":
                assert self._tuple_entries is not None
                layers: list[tuple[torch.Tensor, torch.Tensor]] = []
                for ent in self._tuple_entries:
                    k = symmetric_dequantize_int8(
                        ent.qk, ent.scale_k, out_dtype=ent.kv_dtype, out_device=ent.kv_device
                    )
                    v = symmetric_dequantize_int8(
                        ent.qv, ent.scale_v, out_dtype=ent.kv_dtype, out_device=ent.kv_device
                    )
                    layers.append((k, v))
                return tuple(layers)

            assert self._dynamic_entries is not None
            new_cache = DynamicCache()
            new_layers: list[Any] = []
            for entry in self._dynamic_entries:
                if isinstance(entry, _QuantSlidePair):
                    k = symmetric_dequantize_int8(
                        entry.qk,
                        entry.scale_k,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    v = symmetric_dequantize_int8(
                        entry.qv,
                        entry.scale_v,
                        out_dtype=entry.kv_dtype,
                        out_device=entry.kv_device,
                    )
                    nl = DynamicSlidingWindowLayer(sliding_window=entry.sliding_window)
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    nl.cumulative_length = entry.cumulative_length
                    nl._sliding_window_tensor = entry.sliding_window_tensor_cpu.to(device=k.device)
                    new_layers.append(nl)
                elif isinstance(entry, _QuantKVPair):
                    k = symmetric_dequantize_int8(
                        entry.qk, entry.scale_k, out_dtype=entry.kv_dtype, out_device=entry.kv_device
                    )
                    v = symmetric_dequantize_int8(
                        entry.qv, entry.scale_v, out_dtype=entry.kv_dtype, out_device=entry.kv_device
                    )
                    nl = DynamicLayer()
                    nl.keys = k
                    nl.values = v
                    nl.is_initialized = True
                    nl.dtype = k.dtype
                    nl.device = k.device
                    new_layers.append(nl)
                else:
                    new_layers.append(entry)
            new_cache.layers = new_layers
            return new_cache
        finally:
            self._cumulative_dequant_s += time.perf_counter() - t0

    def _sparse_metadata_bytes(self) -> int:
        meta = 0
        if self._retained_indices:
            meta += len(self._retained_indices) * 8
        if self._token_scores_cpu is not None:
            meta += int(self._token_scores_cpu.numel()) * int(self._token_scores_cpu.element_size())
        meta += 3 * 8
        return int(meta)

    def _quant_metadata_bytes(self) -> int:
        payload_ignored, meta = self._quant_payload_and_meta()
        return int(meta)

    def _quant_payload_and_meta(self) -> tuple[int, int]:
        """INT8 payload element count (as bytes) + quant-side metadata (scales, slide aux)."""
        payload = 0
        metadata = 0
        float32_scale_bytes = 4

        if self._tuple_entries:
            for ent in self._tuple_entries:
                payload += int(ent.qk.numel()) + int(ent.qv.numel())
                metadata += 2 * float32_scale_bytes

        if self._dynamic_entries:
            for entry in self._dynamic_entries:
                if isinstance(entry, _QuantSlidePair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    st = entry.sliding_window_tensor_cpu
                    metadata += 2 * float32_scale_bytes + int(st.numel() * st.element_size())
                elif isinstance(entry, _QuantKVPair):
                    payload += int(entry.qk.numel()) + int(entry.qv.numel())
                    metadata += 2 * float32_scale_bytes

        return int(payload), int(metadata)

    def memory_bytes(self) -> int:
        p, qmeta = self._quant_payload_and_meta()
        sm = self._sparse_metadata_bytes()
        return int(p + qmeta + sm)

    def stats(self) -> dict[str, Any]:
        retained = len(self._retained_indices)
        mean_sparsity = (
            (self._sum_sparsity / self._sparsity_samples) if self._sparsity_samples > 0 else 0.0
        )
        payload_int8, metadata_quant = self._quant_payload_and_meta()
        metadata_sparse = self._sparse_metadata_bytes()
        n_layers = (
            len(self._tuple_entries)
            if self._tuple_entries
            else (len(self._dynamic_entries) if self._dynamic_entries else 0)
        )

        seq_len: int | None = None
        if self._tuple_entries and self._tuple_entries:
            if self._tuple_entries[0].qk.dim() >= 2:
                seq_len = int(self._tuple_entries[0].qk.shape[-2])
        elif self._dynamic_entries:
            for e in self._dynamic_entries:
                if isinstance(e, _QuantKVPair):
                    if e.qk.dim() >= 2:
                        seq_len = int(e.qk.shape[-2])
                    break

        dense_fp16_est = 0
        if self._full_seq_len > 0 and retained > 0:
            p_fp16 = payload_int8 * 2
            dense_fp16_est = int(p_fp16 * (self._full_seq_len / float(retained)))

        return {
            "type": "KVCacheSparseQuantized",
            "composition_order": "sparsify_then_quantize_retained",
            "quant_scheme": "symmetric_int8_per_tensor_kv",
            "quantization_kv_bits": 8,
            "selection_scope": "token_level_shared_across_heads_per_layer",
            "recent_window": int(self._config.recent_window),
            "heavy_hitter_budget": int(self._config.heavy_hitter_budget),
            "refresh_interval_decode_steps": int(self._config.refresh_interval),
            "scoring_mode": self._config.scoring,
            "full_sequence_length": int(self._full_seq_len),
            "retained_sequence_length": int(retained),
            "sparsity_ratio_point_estimate": (
                (1.0 - retained / float(self._full_seq_len)) if self._full_seq_len > 0 else 0.0
            ),
            "mean_sparsity_ratio_over_appends": float(mean_sparsity),
            "payload_bytes_int8": int(payload_int8),
            "metadata_bytes_sparse": int(metadata_sparse),
            "metadata_bytes_quant": int(metadata_quant),
            "metadata_bytes": int(metadata_sparse + metadata_quant),
            "memory_bytes_logical": int(payload_int8 + metadata_sparse + metadata_quant),
            "dense_fp16_kv_bytes_est": int(dense_fp16_est),
            "num_layers": int(n_layers),
            "sequence_length_est": seq_len,
            "refresh_events": int(self._refresh_events),
            "append_calls": int(self._append_calls),
            "cumulative_refresh_time_s": float(self._cumulative_refresh_s),
            "cumulative_dequant_time_s": float(self._cumulative_dequant_s),
        }
