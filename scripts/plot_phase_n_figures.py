#!/usr/bin/env python3
"""Generate professor-facing figures from Phase N CSVs under results/phase_n/figures/."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase_n" / "figures"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _ok(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r.get("status") == "ok"]


def _f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def plot_longctx(csv_path: Path) -> None:
    rows = _ok(_read_rows(csv_path))
    # One row per baseline in longctx smoke
    order = ["ar", "ar_eager", "quant_spec"]
    by_base = {r["baseline"]: r for r in rows}
    labels = []
    decode = []
    mem_gb = []
    acc = []
    for b in order:
        r = by_base.get(b)
        if not r:
            continue
        labels.append(
            {
                "ar": "hf_ar (SDPA)",
                "ar_eager": "hf_ar_eager",
                "quant_spec": "hierarchical_fused",
            }[b]
        )
        decode.append(_f(r["tokens_per_sec_decode_phase"]))
        mem_gb.append(_f(r["gpu_peak_memory_bytes_after_run"]) / 1e9)
        a = r.get("acceptance_rate", "").strip()
        acc.append(_f(a) if a else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, decode, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Long-context smoke (~1920 tokens): decode phase")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "longctx_decode_throughput.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, mem_gb, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Peak GPU memory after run (GB, 1e9 bytes)")
    ax.set_title("Long-context smoke: peak memory")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "longctx_peak_memory.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = ["hierarchical_fused"]
    ax.bar(x, [acc[-1]], color="#C44E52")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Long-context hierarchical: acceptance (AR modes N/A)")
    fig.tight_layout()
    fig.savefig(OUT / "longctx_acceptance.png", dpi=150)
    plt.close(fig)


def plot_smoke(csv_path: Path, tag: str) -> None:
    rows = _ok(_read_rows(csv_path))
    labels = []
    decode = []
    mem_gb = []
    for r in rows:
        cm = r.get("comparison_mode") or r["baseline"]
        if r["baseline"] == "ar":
            labels.append("hf_ar")
        else:
            labels.append(str(cm))
        decode.append(_f(r["tokens_per_sec_decode_phase"]))
        mem_gb.append(_f(r["gpu_peak_memory_bytes_after_run"]) / 1e9)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, decode, color=["#4C72B0", "#C44E52"][: len(labels)])
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title(f"Short-context smoke ({tag})")
    fig.tight_layout()
    fig.savefig(OUT / f"smoke_{tag}_decode_throughput.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, mem_gb, color=["#4C72B0", "#C44E52"][: len(labels)])
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"Short-context smoke: peak memory ({tag})")
    fig.tight_layout()
    fig.savefig(OUT / f"smoke_{tag}_peak_memory.png", dpi=150)
    plt.close(fig)


def plot_sweep_median_decode(csv_path: Path) -> None:
    rows = _ok(_read_rows(csv_path))
    by_mode: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            by_mode[r["comparison_mode"]].append(_f(r["tokens_per_sec_decode_phase"]))
        except KeyError:
            continue

    order = ["hf_ar", "dense_self_spec", "hierarchical_ref", "hierarchical_fused"]
    labels = []
    values = []
    for m in order:
        xs = by_mode.get(m)
        if not xs:
            continue
        labels.append(m)
        values.append(median(xs))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#8172B3", "#CCB974", "#C44E52"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylabel("Median decode tok/s")
    ax.set_title("Modal sweep: median decode throughput by comparison_mode")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_modal_v2_median_decode_by_mode.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_longctx(ROOT / "results" / "phase_n" / "phase_n_llama_longctx_weaker_ar_v1.csv")
    plot_smoke(ROOT / "results" / "phase_n" / "phase_n_llama_ar_vs_fused_smoke_v7.csv", "v7_openllama7b")
    plot_sweep_median_decode(ROOT / "results" / "phase_n" / "phase_n_sweep_modal_v2.csv")
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
