"""Phase N — practical roofline summary from Phase H / Phase N CSV (schema v4).

Reads rows with ``status=ok`` and aggregates mean decode throughput + traffic proxies by
``comparison_mode`` × ``context_length_target_tokens`` (and optionally ``gamma``).

Usage::

    PYTHONPATH=src python -m benchmarks.phase_n_report --csv outputs/phase_n/smoke.csv --out outputs/phase_n/report.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _f(x: str) -> float | None:
    t = (x or "").strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _i(x: str) -> int | None:
    t = (x or "").strip()
    if t == "":
        return None
    try:
        return int(t)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return sorted summary rows: group key + mean tps_decode + mean bytes proxy + mean implied GB/s."""
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("warmup", "").lower() in ("true", "1", "yes"):
            continue
        mode = r.get("comparison_mode") or r.get("baseline", "")
        ctx = _i(r.get("context_length_target_tokens"))
        gamma = r.get("gamma") or ""
        g = (mode, ctx, gamma)
        groups[g].append(r)

    out: list[dict[str, Any]] = []
    for key, g_rows in sorted(groups.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0, str(x[0][2]))):
        mode, ctx, gamma = key
        tps = [x for x in (_f(r.get("tokens_per_sec_decode_phase")) for r in g_rows) if x is not None]
        bpt = [
            x
            for x in (_f(r.get("estimated_kv_traffic_bytes_per_output_token")) for r in g_rows)
            if x is not None
        ]
        gbps = [x for x in (_f(r.get("implied_bandwidth_gbps_decode_proxy")) for r in g_rows) if x is not None]
        out.append(
            {
                "comparison_mode": mode,
                "context_length_target_tokens": ctx,
                "gamma": gamma,
                "n_rows": len(g_rows),
                "mean_tokens_per_sec_decode_phase": mean(tps) if tps else None,
                "mean_estimated_kv_traffic_b_per_tok": mean(bpt) if bpt else None,
                "mean_implied_bandwidth_gbps_proxy": mean(gbps) if gbps else None,
            }
        )
    return out


def aggregate_extended(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Stratify by ``model_name`` × ``gpu_torch_name`` × ``comparison_mode`` × context (for cross-GPU / model tables)."""
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("warmup", "").lower() in ("true", "1", "yes"):
            continue
        mode = r.get("comparison_mode") or r.get("baseline", "")
        ctx = _i(r.get("context_length_target_tokens"))
        gamma = r.get("gamma") or ""
        model = r.get("model_name") or ""
        gpu = r.get("gpu_torch_name") or ""
        g = (model, gpu, mode, ctx, gamma)
        groups[g].append(r)

    out: list[dict[str, Any]] = []
    for key, g_rows in sorted(
        groups.items(),
        key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]), x[0][3] or 0, str(x[0][4])),
    ):
        model, gpu, mode, ctx, gamma = key
        tps = [x for x in (_f(r.get("tokens_per_sec_decode_phase")) for r in g_rows) if x is not None]
        tps_e2e = [x for x in (_f(r.get("tokens_per_sec_e2e_timed")) for r in g_rows) if x is not None]
        bpt = [
            x
            for x in (_f(r.get("estimated_kv_traffic_bytes_per_output_token")) for r in g_rows)
            if x is not None
        ]
        gbps = [x for x in (_f(r.get("implied_bandwidth_gbps_decode_proxy")) for r in g_rows) if x is not None]
        out.append(
            {
                "model_name": model,
                "gpu_torch_name": gpu,
                "comparison_mode": mode,
                "context_length_target_tokens": ctx,
                "gamma": gamma,
                "n_rows": len(g_rows),
                "mean_tokens_per_sec_decode_phase": mean(tps) if tps else None,
                "mean_tokens_per_sec_e2e_timed": mean(tps_e2e) if tps_e2e else None,
                "mean_estimated_kv_traffic_b_per_tok": mean(bpt) if bpt else None,
                "mean_implied_bandwidth_gbps_proxy": mean(gbps) if gbps else None,
            }
        )
    return out


def write_markdown(summaries: list[dict[str, Any]], dest: Path, *, csv_path: Path) -> None:
    lines = [
        "# Phase N — roofline-style summary\n",
        "\n",
        f"**Source CSV:** `{csv_path}`\n",
        "\n",
        "Definitions: **decode tok/s** uses `tokens_per_sec_decode_phase` (prefill excluded). "
        "**Traffic proxy** uses `estimated_kv_traffic_bytes_per_output_token` (logical store or verifier KV "
        "/ sequence length — not a hardware counter). **Implied GB/s** = tok/s × bytes/tok.\n",
        "\n",
        "| comparison_mode | context_tokens | γ | rows | mean decode tok/s | mean B/token (proxy) | mean implied GB/s |\n",
        "|---|---:|---|---:|---:|---:|---:|\n",
    ]
    for s in summaries:
        lines.append(
            "| {mode} | {ctx} | {gam} | {n} | {tps} | {bpt} | {gbps} |\n".format(
                mode=s["comparison_mode"],
                ctx=s["context_length_target_tokens"] if s["context_length_target_tokens"] is not None else "",
                gam=s["gamma"] if s["gamma"] != "" else "—",
                n=s["n_rows"],
                tps=f"{s['mean_tokens_per_sec_decode_phase']:.4f}"
                if s["mean_tokens_per_sec_decode_phase"] is not None
                else "",
                bpt=f"{s['mean_estimated_kv_traffic_b_per_tok']:.1f}"
                if s["mean_estimated_kv_traffic_b_per_tok"] is not None
                else "",
                gbps=f"{s['mean_implied_bandwidth_gbps_proxy']:.3f}"
                if s["mean_implied_bandwidth_gbps_proxy"] is not None
                else "",
            )
        )
    lines.append(
        "\n**Interpretation:** If hierarchical fused shows higher decode tok/s with lower implied GB/s proxy "
        "versus hierarchical_ref, reduced HBM traffic is plausibly consistent with the speedup (qualitative; "
        "validate with profiler on a subset).\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines), encoding="utf-8")


def write_markdown_extended(
    extended: list[dict[str, Any]],
    dest: Path,
    *,
    csv_path: Path,
) -> None:
    """Append a second table: model × GPU × mode × context with e2e decode + decode-only columns."""
    lines = [
        "\n---\n\n# Extended: model × GPU × regime\n\n",
        f"**Source CSV:** `{csv_path}`\n\n",
        "`mean e2e tok/s` uses `tokens_per_sec_e2e_timed` (prefill + decode). "
        "`mean decode tok/s` uses `tokens_per_sec_decode_phase` (prefill excluded).\n\n",
        "| model | GPU | comparison_mode | ctx | γ | rows | mean e2e tok/s | mean decode tok/s | mean B/token | mean implied GB/s |\n",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for s in extended:
        lines.append(
            "| {m} | {g} | {mode} | {ctx} | {gam} | {n} | {e2e} | {dec} | {bpt} | {gbps} |\n".format(
                m=s["model_name"][:40] if s["model_name"] else "",
                g=s["gpu_torch_name"][:24] if s["gpu_torch_name"] else "",
                mode=s["comparison_mode"],
                ctx=s["context_length_target_tokens"] if s["context_length_target_tokens"] is not None else "",
                gam=s["gamma"] if s["gamma"] != "" else "—",
                n=s["n_rows"],
                e2e=f"{s['mean_tokens_per_sec_e2e_timed']:.4f}"
                if s["mean_tokens_per_sec_e2e_timed"] is not None
                else "",
                dec=f"{s['mean_tokens_per_sec_decode_phase']:.4f}"
                if s["mean_tokens_per_sec_decode_phase"] is not None
                else "",
                bpt=f"{s['mean_estimated_kv_traffic_b_per_tok']:.1f}"
                if s["mean_estimated_kv_traffic_b_per_tok"] is not None
                else "",
                gbps=f"{s['mean_implied_bandwidth_gbps_proxy']:.3f}"
                if s["mean_implied_bandwidth_gbps_proxy"] is not None
                else "",
            )
        )
    lines.append(
        "\n**Note:** Compare `hierarchical_fused` vs `hf_ar` / `hierarchical_ref` at **longer context** and "
        "**larger models** — the gap between baseline and fused path often **narrows or reverses** when KV "
        "traffic dominates (see profiler notes on launch vs kernel time at short contexts).\n"
    )
    base = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    dest.write_text(base + "".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase N roofline report from Phase H CSV")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Markdown output path")
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Append model × GPU × comparison_mode table (for multi-GPU / multi-model sweeps)",
    )
    args = ap.parse_args()
    rows = load_rows(args.csv)
    summ = aggregate(rows)
    write_markdown(summ, args.out, csv_path=args.csv.resolve())
    if args.extended:
        ext = aggregate_extended(rows)
        write_markdown_extended(ext, args.out, csv_path=args.csv.resolve())
    print(f"[phase_n_report] wrote {args.out}")


if __name__ == "__main__":
    main()
