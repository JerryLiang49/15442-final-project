"""Phase H benchmark schema and runner smoke tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.phase_h_schema import PHASE_H_CSV_FIELDNAMES, PHASE_H_SCHEMA_VERSION


def test_phase_h_schema_version_stable() -> None:
    assert PHASE_H_SCHEMA_VERSION == "5"
    assert "gamma" in PHASE_H_CSV_FIELDNAMES
    assert "tokens_per_sec_decode_phase" in PHASE_H_CSV_FIELDNAMES
    assert "comparison_mode" in PHASE_H_CSV_FIELDNAMES
    assert "context_length_target_tokens" in PHASE_H_CSV_FIELDNAMES


@pytest.mark.slow
def test_phase_h_runner_smoke_csv(tmp_path: Path) -> None:
    from benchmarks.phase_h_runner import run_phase_h_benchmark

    cfg = tmp_path / "smoke.yaml"
    out = tmp_path / "out.csv"
    cfg.write_text(
        """
sweep_id: t_smoke
model_name: gpt2
device: cpu
dtype: float32
seed: 1
warmup_trials: 0
num_trials: 1
verify_match: false
max_new_tokens: 4
min_prompt_tokens: 8
mt_bench_path: data/mt_bench_subset.json
max_prompts: 1
baselines: [ar]
gamma_values: [4]
G_values: [8]
quant_group_size_values: [null]
kv_kernel_backend_values: [reference]
output_csv: {out}
raw_jsonl: {raw}
resume: false
""".strip().format(out=out.as_posix(), raw=(tmp_path / "raw.jsonl").as_posix()),
        encoding="utf-8",
    )
    assert run_phase_h_benchmark(cfg) == 0
    assert out.is_file()
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["baseline"] == "ar"
    assert rows[0]["gamma"] == ""
    assert rows[0]["comparison_mode"] == "hf_ar"
    assert rows[0]["context_length_target_tokens"] == "8"
