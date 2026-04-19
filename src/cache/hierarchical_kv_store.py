"""Shared hierarchical KV storage (QuantSpec-style): one physical store, two logical views.

**Quantization (see :mod:`cache.quant_spec_kv`)**

* **Keys** — channel-wise asymmetric INT4 upper/lower along head-dimension groups.
* **Values** — token-wise asymmetric INT4 upper/lower along sequence groups.
* **Draft view** — dequant **upper** only for K and V (same packed tensors as target).
* **Target view** — dequant **upper + lower** for K and V.

**Buffers**

* **CF1** — committed recent FP16 (settled prefix for this decode policy).
* **CF2** — speculative FP16 tail; draft appends here; verification rejection trims **only CF2**.

**Mutation rules**

* Rejection trims **CF2** only; rollover quantizes **CF1** into historical INT4 and shifts CF2→CF1.

**Phase L (fast path, default)**

* **CF2** uses preallocated ``[B,H,CAP,D]`` buffers with valid length ``_cf2_len``; appends ``copy_`` into
  the tail; **trim** only updates ``_cf2_len`` (no tensor realloc); **commit** shifts the retained
  tail with a cloned slice (correctness over overlapping self-copies).
* **Rollover** quantize+append is shared with :mod:`cache.kv_cache_ops`; CF2→CF1 is **pointer move**
  (no KV memcpy). Optional :class:`~cache.cache_mutation_profile.CacheMutationProfile` timings.

See :mod:`cache.recent_buffer` for the first-class API, instrumentation, and exact rollover semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .cache_mutation_profile import (
    CacheMutationProfile,
    timer_append,
    timer_pack,
    timer_rollback,
    timer_rollover,
)
from .quant_spec_kv import (
    quantize_fp16_kv_to_upper_lower,
    reconstruct_key_draft,
    reconstruct_key_target,
    reconstruct_value_draft,
    reconstruct_value_target,
)


def _n_key_groups(head_dim: int, group_size: int) -> int:
    if head_dim % group_size != 0:
        raise ValueError(f"head_dim {head_dim} not divisible by group_size {group_size}")
    return head_dim // group_size


def _pad_seq_to_multiple(k: torch.Tensor, v: torch.Tensor, g: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad along dim 2 so length is divisible by ``g`` (value token groups). Returns (k,v, orig_len)."""
    s = int(k.shape[2])
    orig = s
    if s % g == 0:
        return k, v, orig
    pad = g - (s % g)
    k = torch.nn.functional.pad(k, (0, 0, 0, pad))
    v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    return k, v, orig


def append_hist_key(
    cur_uq: torch.Tensor,
    cur_us: torch.Tensor,
    cur_uzp: torch.Tensor,
    cur_lq: torch.Tensor,
    cur_ls: torch.Tensor,
    cur_lzp: torch.Tensor,
    nuq: torch.Tensor,
    nus: torch.Tensor,
    nuzp: torch.Tensor,
    nlq: torch.Tensor,
    nls: torch.Tensor,
    nlzp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cur_uq.shape[2] == 0:
        return nuq, nus, nuzp, nlq, nls, nlzp
    return (
        torch.cat([cur_uq, nuq], dim=2),
        torch.cat([cur_us, nus], dim=2),
        torch.cat([cur_uzp, nuzp], dim=2),
        torch.cat([cur_lq, nlq], dim=2),
        torch.cat([cur_ls, nls], dim=2),
        torch.cat([cur_lzp, nlzp], dim=2),
    )


def append_hist_value(
    cur_vuq: torch.Tensor,
    cur_vus: torch.Tensor,
    cur_vuzp: torch.Tensor,
    cur_vlq: torch.Tensor,
    cur_vls: torch.Tensor,
    cur_vlzp: torch.Tensor,
    nuq: torch.Tensor,
    nus: torch.Tensor,
    nuzp: torch.Tensor,
    nlq: torch.Tensor,
    nls: torch.Tensor,
    nlzp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cur_vuq.shape[2] == 0:
        return nuq, nus, nuzp, nlq, nls, nlzp
    return (
        torch.cat([cur_vuq, nuq], dim=2),
        torch.cat([cur_vus, nus], dim=2),
        torch.cat([cur_vuzp, nuzp], dim=2),
        torch.cat([cur_vlq, nlq], dim=2),
        torch.cat([cur_vls, nls], dim=2),
        torch.cat([cur_vlzp, nlzp], dim=2),
    )


@dataclass
class HierarchicalKVView:
    """Read-only bundle of tensors for one logical view (per-layer list)."""

    layers_k: list[torch.Tensor]
    layers_v: list[torch.Tensor]
    upper_only: bool
    """If True, lower residual was omitted (draft view)."""


class HierarchicalKVStore:
    """Single physical KV store: INT4 history + CF1 + CF2 FP16 buffers."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
        G: int = 64,
        quant_group_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        use_fast_cf2: bool = True,
        enable_mutation_profiling: bool = False,
        mutation_sync_cuda: bool = False,
    ) -> None:
        """
        Args:
            num_layers: Transformer layers.
            num_heads: Attention heads.
            head_dim: Head dimension ``D``.
            batch_size: Batch (only ``1`` tested).
            G: Recent-window policy unit; CF1 may hold up to ``2 * G`` FP16 tokens.
            quant_group_size: INT4 group size for **both** K (channel groups) and V (token groups).
                Default ``head_dim`` (one channel group per position; ``S`` must be divisible by this
                for value quantization on a slice — use padding internally when needed).
            device: Device for tensors.
            dtype: FP16 dtype for CF1/CF2 and quant metadata.
            use_fast_cf2: When True (default), CF2 uses preallocated buffers with copy-into-tail appends,
                metadata-only trim, and in-place tail shift on commit. Set False for legacy ``torch.cat``/slice
                behavior (tests / bisect).
            enable_mutation_profiling: Populate :attr:`mutation_profile` with last/total timings.
            mutation_sync_cuda: If True, synchronize CUDA before stopping timers (more accurate GPU time).
        """
        if G < 1:
            raise ValueError("G must be >= 1")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.batch_size = int(batch_size)
        self.G = int(G)
        self.cf1_max_tokens = 2 * self.G
        self.quant_group_size = int(quant_group_size) if quant_group_size is not None else self.head_dim
        self._n_gk = _n_key_groups(self.head_dim, self.quant_group_size)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._use_fast_cf2 = bool(use_fast_cf2)
        self._mutation_profile: CacheMutationProfile | None = None
        if enable_mutation_profiling:
            self._mutation_profile = CacheMutationProfile(sync_cuda=mutation_sync_cuda)

        z_k = torch.zeros(batch_size, num_heads, 0, head_dim, device=self.device, dtype=torch.int8)
        z_ks = torch.zeros(batch_size, num_heads, 0, self._n_gk, device=self.device, dtype=dtype)
        z_kzp = z_ks.clone()
        z_v = z_k.clone()
        z_vs = torch.zeros(batch_size, num_heads, 0, head_dim, device=self.device, dtype=dtype)
        z_vzp = z_vs.clone()

        self._upper_k: list[torch.Tensor] = []
        self._upper_k_scale: list[torch.Tensor] = []
        self._upper_k_zp: list[torch.Tensor] = []
        self._lower_k: list[torch.Tensor] = []
        self._lower_k_scale: list[torch.Tensor] = []
        self._lower_k_zp: list[torch.Tensor] = []

        self._upper_v: list[torch.Tensor] = []
        self._upper_v_scale: list[torch.Tensor] = []
        self._upper_v_zp: list[torch.Tensor] = []
        self._lower_v: list[torch.Tensor] = []
        self._lower_v_scale: list[torch.Tensor] = []
        self._lower_v_zp: list[torch.Tensor] = []

        self._hist_len: int = 0

        self._cf1_k: list[torch.Tensor | None] = [None] * num_layers
        self._cf1_v: list[torch.Tensor | None] = [None] * num_layers
        self._cf1_len: int = 0

        self._cf2_k: list[torch.Tensor | None] = [None] * num_layers
        self._cf2_v: list[torch.Tensor | None] = [None] * num_layers
        self._cf2_len: int = 0
        self._cf2_capacity: list[int] = [0] * num_layers

        for _ in range(num_layers):
            self._upper_k.append(z_k.clone())
            self._upper_k_scale.append(z_ks.clone())
            self._upper_k_zp.append(z_kzp.clone())
            self._lower_k.append(z_k.clone())
            self._lower_k_scale.append(z_ks.clone())
            self._lower_k_zp.append(z_kzp.clone())
            self._upper_v.append(z_k.clone())
            self._upper_v_scale.append(z_vs.clone())
            self._upper_v_zp.append(z_vzp.clone())
            self._lower_v.append(z_k.clone())
            self._lower_v_scale.append(z_vs.clone())
            self._lower_v_zp.append(z_vzp.clone())

    @property
    def hist_len(self) -> int:
        return self._hist_len

    @property
    def cf1_len(self) -> int:
        return self._cf1_len

    @property
    def cf2_len(self) -> int:
        return self._cf2_len

    @property
    def mutation_profile(self) -> CacheMutationProfile | None:
        return self._mutation_profile

    def committed_fp16_len(self) -> int:
        return self._cf1_len

    def speculative_fp16_len(self) -> int:
        return self._cf2_len

    def logical_committed_seq_len(self) -> int:
        return self._hist_len + self._cf1_len

    def logical_draft_seq_len(self) -> int:
        return self._hist_len + self._cf1_len + self._cf2_len

    def logical_target_seq_len(self) -> int:
        return self.logical_draft_seq_len()

    def debug_state(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "G": self.G,
            "quant_group_size": self.quant_group_size,
            "n_key_groups": self._n_gk,
            "cf1_max_tokens": self.cf1_max_tokens,
            "hist_len": self._hist_len,
            "cf1_len": self._cf1_len,
            "cf2_len": self._cf2_len,
            "logical_committed_seq_len": self.logical_committed_seq_len(),
            "logical_draft_seq_len": self.logical_draft_seq_len(),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    def memory_bytes_estimate(self) -> int:
        total = 0
        for name in (
            "_upper_k",
            "_upper_k_scale",
            "_upper_k_zp",
            "_lower_k",
            "_lower_k_scale",
            "_lower_k_zp",
            "_upper_v",
            "_upper_v_scale",
            "_upper_v_zp",
            "_lower_v",
            "_lower_v_scale",
            "_lower_v_zp",
        ):
            for t in getattr(self, name):
                if t.numel():
                    total += t.numel() * t.element_size()
        for lst_name in ("_cf1_k", "_cf1_v", "_cf2_k", "_cf2_v"):
            for t in getattr(self, lst_name):
                if t is not None and t.numel():
                    total += t.numel() * t.element_size()
        return int(total)

    def draft_view(self) -> HierarchicalKVView:
        """Upper INT4 only (reconstructed) + CF1 + CF2 FP16."""
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for i in range(self.num_layers):
            parts_k: list[torch.Tensor] = []
            parts_v: list[torch.Tensor] = []
            if self._hist_len > 0:
                parts_k.append(
                    reconstruct_key_draft(
                        self._upper_k[i], self._upper_k_scale[i], self._upper_k_zp[i]
                    ).to(self.dtype)
                )
                parts_v.append(
                    reconstruct_value_draft(
                        self._upper_v[i],
                        self._upper_v_scale[i],
                        self._upper_v_zp[i],
                        group_size=self.quant_group_size,
                        logical_seq_len=self._hist_len,
                    ).to(self.dtype)
                )
            if self._cf1_len > 0 and self._cf1_k[i] is not None:
                parts_k.append(self._cf1_k[i])
                parts_v.append(self._cf1_v[i])
            if self._cf2_len > 0 and self._cf2_k[i] is not None:
                parts_k.append(self._cf2_k[i][:, :, : self._cf2_len, :])
                parts_v.append(self._cf2_v[i][:, :, : self._cf2_len, :])
            if not parts_k:
                k_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
                v_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
            else:
                k_list.append(torch.cat(parts_k, dim=2))
                v_list.append(torch.cat(parts_v, dim=2))
        return HierarchicalKVView(layers_k=k_list, layers_v=v_list, upper_only=True)

    def target_view(self) -> HierarchicalKVView:
        """Upper + lower (reconstructed) + CF1 + CF2."""
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for i in range(self.num_layers):
            parts_k: list[torch.Tensor] = []
            parts_v: list[torch.Tensor] = []
            if self._hist_len > 0:
                parts_k.append(
                    reconstruct_key_target(
                        self._upper_k[i],
                        self._upper_k_scale[i],
                        self._upper_k_zp[i],
                        self._lower_k[i],
                        self._lower_k_scale[i],
                        self._lower_k_zp[i],
                    ).to(self.dtype)
                )
                parts_v.append(
                    reconstruct_value_target(
                        self._upper_v[i],
                        self._upper_v_scale[i],
                        self._upper_v_zp[i],
                        self._lower_v[i],
                        self._lower_v_scale[i],
                        self._lower_v_zp[i],
                        group_size=self.quant_group_size,
                        logical_seq_len=self._hist_len,
                    ).to(self.dtype)
                )
            if self._cf1_len > 0 and self._cf1_k[i] is not None:
                parts_k.append(self._cf1_k[i])
                parts_v.append(self._cf1_v[i])
            if self._cf2_len > 0 and self._cf2_k[i] is not None:
                parts_k.append(self._cf2_k[i][:, :, : self._cf2_len, :])
                parts_v.append(self._cf2_v[i][:, :, : self._cf2_len, :])
            if not parts_k:
                k_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
                v_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
            else:
                k_list.append(torch.cat(parts_k, dim=2))
                v_list.append(torch.cat(parts_v, dim=2))
        return HierarchicalKVView(layers_k=k_list, layers_v=v_list, upper_only=False)

    def draft_view_without_cf2(self) -> HierarchicalKVView:
        """Draft reconstruction of history + CF1 only (no speculative CF2 tail)."""
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for i in range(self.num_layers):
            parts_k: list[torch.Tensor] = []
            parts_v: list[torch.Tensor] = []
            if self._hist_len > 0:
                parts_k.append(
                    reconstruct_key_draft(
                        self._upper_k[i], self._upper_k_scale[i], self._upper_k_zp[i]
                    ).to(self.dtype)
                )
                parts_v.append(
                    reconstruct_value_draft(
                        self._upper_v[i],
                        self._upper_v_scale[i],
                        self._upper_v_zp[i],
                        group_size=self.quant_group_size,
                        logical_seq_len=self._hist_len,
                    ).to(self.dtype)
                )
            if self._cf1_len > 0 and self._cf1_k[i] is not None:
                parts_k.append(self._cf1_k[i])
                parts_v.append(self._cf1_v[i])
            if not parts_k:
                k_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
                v_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
            else:
                k_list.append(torch.cat(parts_k, dim=2))
                v_list.append(torch.cat(parts_v, dim=2))
        return HierarchicalKVView(layers_k=k_list, layers_v=v_list, upper_only=True)

    def target_view_without_cf2(self) -> HierarchicalKVView:
        """Target (upper+lower) reconstruction of history + CF1 only — verifier past before a draft block."""
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for i in range(self.num_layers):
            parts_k: list[torch.Tensor] = []
            parts_v: list[torch.Tensor] = []
            if self._hist_len > 0:
                parts_k.append(
                    reconstruct_key_target(
                        self._upper_k[i],
                        self._upper_k_scale[i],
                        self._upper_k_zp[i],
                        self._lower_k[i],
                        self._lower_k_scale[i],
                        self._lower_k_zp[i],
                    ).to(self.dtype)
                )
                parts_v.append(
                    reconstruct_value_target(
                        self._upper_v[i],
                        self._upper_v_scale[i],
                        self._upper_v_zp[i],
                        self._lower_v[i],
                        self._lower_v_scale[i],
                        self._lower_v_zp[i],
                        group_size=self.quant_group_size,
                        logical_seq_len=self._hist_len,
                    ).to(self.dtype)
                )
            if self._cf1_len > 0 and self._cf1_k[i] is not None:
                parts_k.append(self._cf1_k[i])
                parts_v.append(self._cf1_v[i])
            if not parts_k:
                k_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
                v_list.append(
                    torch.empty(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                )
            else:
                k_list.append(torch.cat(parts_k, dim=2))
                v_list.append(torch.cat(parts_v, dim=2))
        return HierarchicalKVView(layers_k=k_list, layers_v=v_list, upper_only=False)

    def prefill_from_fp16(
        self,
        layers_k: list[torch.Tensor],
        layers_v: list[torch.Tensor],
        *,
        recent_tokens_cap: int | None = None,
    ) -> None:
        if len(layers_k) != self.num_layers or len(layers_v) != self.num_layers:
            raise ValueError("layer list length must match num_layers")
        cap = int(recent_tokens_cap if recent_tokens_cap is not None else self.cf1_max_tokens)
        if cap < 1:
            raise ValueError("recent_tokens_cap must be >= 1")

        s0 = int(layers_k[0].shape[2])
        for i in range(self.num_layers):
            if int(layers_k[i].shape[2]) != s0 or int(layers_v[i].shape[2]) != s0:
                raise ValueError("all layers must share sequence length S")

        recent = min(s0, cap)
        old = s0 - recent

        self._hist_len = old
        self._cf1_len = recent
        self._cf2_len = 0
        gs = self.quant_group_size

        for i in range(self.num_layers):
            k = layers_k[i].to(device=self.device, dtype=self.dtype)
            v = layers_v[i].to(device=self.device, dtype=self.dtype)
            if old > 0:
                k_old = k[:, :, :old, :]
                v_old = v[:, :, :old, :]
                k_old, v_old, _ = _pad_seq_to_multiple(k_old, v_old, gs)
                (
                    kuq,
                    kus,
                    kuzp,
                    klq,
                    kls,
                    klzp,
                    vuq,
                    vus,
                    vuzp,
                    vlq,
                    vls,
                    vlzp,
                ) = quantize_fp16_kv_to_upper_lower(k_old, v_old, group_size=gs)
                # Trim pad from codes to original ``old`` (pad was at end of S).
                # Value token groups need length ``n_tg * gs``; trimming V codes to ``old`` when
                # ``old % gs != 0`` breaks ``reconstruct_value_*`` (see :mod:`cache.quant_spec_kv`).
                if kuq.shape[2] > old:
                    n_tg = (old + gs - 1) // gs
                    v_len = n_tg * gs
                    kuq = kuq[:, :, :old, :]
                    klq = klq[:, :, :old, :]
                    vuq = vuq[:, :, :v_len, :]
                    vlq = vlq[:, :, :v_len, :]
                    kus = kus[:, :, :old, :]
                    kuzp = kuzp[:, :, :old, :]
                    kls = kls[:, :, :old, :]
                    klzp = klzp[:, :, :old, :]
                    vus = vus[:, :, :n_tg, :]
                    vuzp = vuzp[:, :, :n_tg, :]
                    vls = vls[:, :, :n_tg, :]
                    vlzp = vlzp[:, :, :n_tg, :]
                self._upper_k[i] = kuq
                self._upper_k_scale[i] = kus
                self._upper_k_zp[i] = kuzp
                self._lower_k[i] = klq
                self._lower_k_scale[i] = kls
                self._lower_k_zp[i] = klzp
                self._upper_v[i] = vuq
                self._upper_v_scale[i] = vus
                self._upper_v_zp[i] = vuzp
                self._lower_v[i] = vlq
                self._lower_v_scale[i] = vls
                self._lower_v_zp[i] = vlzp
            else:
                z_k = torch.zeros(
                    self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=torch.int8
                )
                z_ks = torch.zeros(
                    self.batch_size, self.num_heads, 0, self._n_gk, device=self.device, dtype=self.dtype
                )
                z_vs = torch.zeros(self.batch_size, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                self._upper_k[i] = z_k.clone()
                self._upper_k_scale[i] = z_ks.clone()
                self._upper_k_zp[i] = z_ks.clone()
                self._lower_k[i] = z_k.clone()
                self._lower_k_scale[i] = z_ks.clone()
                self._lower_k_zp[i] = z_ks.clone()
                self._upper_v[i] = z_k.clone()
                self._upper_v_scale[i] = z_vs.clone()
                self._upper_v_zp[i] = z_vs.clone()
                self._lower_v[i] = z_k.clone()
                self._lower_v_scale[i] = z_vs.clone()
                self._lower_v_zp[i] = z_vs.clone()

            if recent > 0:
                self._cf1_k[i] = k[:, :, old:s0, :].contiguous()
                self._cf1_v[i] = v[:, :, old:s0, :].contiguous()
            else:
                self._cf1_k[i] = None
                self._cf1_v[i] = None
            self._cf2_k[i] = None
            self._cf2_v[i] = None
            self._cf2_capacity[i] = 0

    def append_cf2_fp16(self, layers_k: list[torch.Tensor], layers_v: list[torch.Tensor]) -> None:
        if len(layers_k) != self.num_layers or len(layers_v) != self.num_layers:
            raise ValueError("layer list length must match num_layers")
        t = int(layers_k[0].shape[2])
        if t < 1:
            raise ValueError("need at least one token to append")
        for i in range(self.num_layers):
            if int(layers_k[i].shape[2]) != t or int(layers_v[i].shape[2]) != t:
                raise ValueError("inconsistent T across layers")

        dev = self.device
        with timer_append(self._mutation_profile, device=dev):
            if not self._use_fast_cf2:
                for i in range(self.num_layers):
                    nk = layers_k[i].to(device=self.device, dtype=self.dtype)
                    nv = layers_v[i].to(device=self.device, dtype=self.dtype)
                    if self._cf2_len == 0:
                        self._cf2_k[i] = nk
                        self._cf2_v[i] = nv
                    else:
                        assert self._cf2_k[i] is not None and self._cf2_v[i] is not None
                        self._cf2_k[i] = torch.cat([self._cf2_k[i], nk], dim=2)
                        self._cf2_v[i] = torch.cat([self._cf2_v[i], nv], dim=2)
                    self._cf2_capacity[i] = 0
                self._cf2_len += t
                return

            for i in range(self.num_layers):
                nk = layers_k[i].to(device=self.device, dtype=self.dtype)
                nv = layers_v[i].to(device=self.device, dtype=self.dtype)
                if self._cf2_k[i] is None:
                    cap = max(self.cf1_max_tokens, t)
                    self._cf2_k[i] = torch.empty(
                        self.batch_size,
                        self.num_heads,
                        cap,
                        self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    self._cf2_v[i] = torch.empty_like(self._cf2_k[i])
                    self._cf2_capacity[i] = cap
                elif self._cf2_len + t > self._cf2_capacity[i]:
                    old_cap = self._cf2_capacity[i]
                    new_cap = max(self._cf2_len + t, old_cap * 2 if old_cap > 0 else self._cf2_len + t)
                    new_k = torch.empty(
                        self.batch_size,
                        self.num_heads,
                        new_cap,
                        self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    new_v = torch.empty_like(new_k)
                    if self._cf2_len > 0:
                        new_k[:, :, : self._cf2_len, :].copy_(self._cf2_k[i][:, :, : self._cf2_len, :])
                        new_v[:, :, : self._cf2_len, :].copy_(self._cf2_v[i][:, :, : self._cf2_len, :])
                    self._cf2_k[i] = new_k
                    self._cf2_v[i] = new_v
                    self._cf2_capacity[i] = new_cap

                assert self._cf2_k[i] is not None and self._cf2_v[i] is not None
                self._cf2_k[i][:, :, self._cf2_len : self._cf2_len + t, :].copy_(nk)
                self._cf2_v[i][:, :, self._cf2_len : self._cf2_len + t, :].copy_(nv)
            self._cf2_len += t

    def trim_cf2(self, keep_prefix_tokens: int) -> None:
        if keep_prefix_tokens < 0:
            raise ValueError("keep_prefix_tokens must be non-negative")
        if keep_prefix_tokens > self._cf2_len:
            raise ValueError("keep_prefix_tokens exceeds cf2_len")
        if keep_prefix_tokens == self._cf2_len:
            return
        dev = self.device
        with timer_rollback(self._mutation_profile, device=dev):
            if self._use_fast_cf2:
                if keep_prefix_tokens == 0:
                    self._cf2_len = 0
                    return
                self._cf2_len = keep_prefix_tokens
                return

            if keep_prefix_tokens == 0:
                for i in range(self.num_layers):
                    self._cf2_k[i] = None
                    self._cf2_v[i] = None
                    self._cf2_capacity[i] = 0
                self._cf2_len = 0
                return
            for i in range(self.num_layers):
                assert self._cf2_k[i] is not None and self._cf2_v[i] is not None
                self._cf2_k[i] = self._cf2_k[i][:, :, :keep_prefix_tokens, :].contiguous()
                self._cf2_v[i] = self._cf2_v[i][:, :, :keep_prefix_tokens, :].contiguous()
                self._cf2_capacity[i] = 0
            self._cf2_len = keep_prefix_tokens

    def clear_cf2(self) -> None:
        self.trim_cf2(0)

    def commit_cf2_prefix_to_cf1(self, num_tokens: int) -> None:
        if num_tokens < 0 or num_tokens > self._cf2_len:
            raise ValueError("num_tokens out of range for CF2")
        if num_tokens == 0:
            return
        rest = self._cf2_len - num_tokens
        for i in range(self.num_layers):
            assert self._cf2_k[i] is not None and self._cf2_v[i] is not None
            head_k = self._cf2_k[i][:, :, :num_tokens, :].contiguous()
            head_v = self._cf2_v[i][:, :, :num_tokens, :].contiguous()
            if self._cf1_len == 0 or self._cf1_k[i] is None:
                self._cf1_k[i] = head_k
                self._cf1_v[i] = head_v
            else:
                self._cf1_k[i] = torch.cat([self._cf1_k[i], head_k], dim=2)
                self._cf1_v[i] = torch.cat([self._cf1_v[i], head_v], dim=2)
            if rest == 0:
                if not self._use_fast_cf2:
                    self._cf2_k[i] = None
                    self._cf2_v[i] = None
                    self._cf2_capacity[i] = 0
            else:
                if self._use_fast_cf2 and self._cf2_capacity[i] > 0:
                    assert self._cf2_k[i] is not None and self._cf2_v[i] is not None
                    tail_k = self._cf2_k[i][:, :, num_tokens : self._cf2_len, :].clone()
                    tail_v = self._cf2_v[i][:, :, num_tokens : self._cf2_len, :].clone()
                    self._cf2_k[i][:, :, :rest, :].copy_(tail_k)
                    self._cf2_v[i][:, :, :rest, :].copy_(tail_v)
                else:
                    self._cf2_k[i] = self._cf2_k[i][:, :, num_tokens:, :].contiguous()
                    self._cf2_v[i] = self._cf2_v[i][:, :, num_tokens:, :].contiguous()
                    self._cf2_capacity[i] = 0
        self._cf1_len += num_tokens
        self._cf2_len -= num_tokens

    def rollover(self) -> None:
        if self._cf1_len == 0 and self._cf2_len == 0:
            return

        from . import kv_cache_ops as _ops

        dev = self.device
        with timer_rollover(self._mutation_profile, device=dev):
            cf1_n = self._cf1_len
            gs = self.quant_group_size

            with timer_pack(self._mutation_profile, device=dev):
                for i in range(self.num_layers):
                    if cf1_n > 0 and self._cf1_k[i] is not None:
                        ck = self._cf1_k[i]
                        cv = self._cf1_v[i]
                        assert ck is not None and cv is not None
                        kuq, kus, kuzp, klq, kls, klzp, vuq, vus, vuzp, vlq, vls, vlzp = (
                            _ops.quantize_cf1_fp16_to_int4(ck, cv, cf1_n, gs)
                        )

                        if self._hist_len == 0:
                            self._upper_k[i] = kuq
                            self._upper_k_scale[i] = kus
                            self._upper_k_zp[i] = kuzp
                            self._lower_k[i] = klq
                            self._lower_k_scale[i] = kls
                            self._lower_k_zp[i] = klzp
                            self._upper_v[i] = vuq
                            self._upper_v_scale[i] = vus
                            self._upper_v_zp[i] = vuzp
                            self._lower_v[i] = vlq
                            self._lower_v_scale[i] = vls
                            self._lower_v_zp[i] = vlzp
                        else:
                            (
                                self._upper_k[i],
                                self._upper_k_scale[i],
                                self._upper_k_zp[i],
                                self._lower_k[i],
                                self._lower_k_scale[i],
                                self._lower_k_zp[i],
                                self._upper_v[i],
                                self._upper_v_scale[i],
                                self._upper_v_zp[i],
                                self._lower_v[i],
                                self._lower_v_scale[i],
                                self._lower_v_zp[i],
                                _new_hist_len,
                            ) = _ops.append_quantized_cf1_to_hist(
                                self._hist_len,
                                self._upper_k[i],
                                self._upper_k_scale[i],
                                self._upper_k_zp[i],
                                self._lower_k[i],
                                self._lower_k_scale[i],
                                self._lower_k_zp[i],
                                self._upper_v[i],
                                self._upper_v_scale[i],
                                self._upper_v_zp[i],
                                self._lower_v[i],
                                self._lower_v_scale[i],
                                self._lower_v_zp[i],
                                kuq,
                                kus,
                                kuzp,
                                klq,
                                kls,
                                klzp,
                                vuq,
                                vus,
                                vuzp,
                                vlq,
                                vls,
                                vlzp,
                            )
                            if int(_new_hist_len) != self._hist_len + cf1_n:
                                raise RuntimeError("hist append length mismatch")

            if cf1_n > 0:
                self._hist_len += cf1_n

            for i in range(self.num_layers):
                self._cf1_k[i] = self._cf2_k[i]
                self._cf1_v[i] = self._cf2_v[i]
                self._cf2_k[i] = None
                self._cf2_v[i] = None
                self._cf2_capacity[i] = 0

            self._cf1_len = self._cf2_len
            self._cf2_len = 0
