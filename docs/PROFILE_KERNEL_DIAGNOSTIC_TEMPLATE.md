# QuantSpec / Triton — Kernel Diagnostic Report

Fill this in **after** running `scripts/profile_kernel.py` on a **CUDA** machine (local GPU or Modal).  
Keep one copy per **(GPU model, driver, PyTorch, Triton, commit SHA)**.

## 0. Environment

| Field | Value |
|-------|--------|
| Date | |
| GPU | (e.g. NVIDIA A10G) |
| Driver / CUDA runtime | |
| PyTorch | `python -c "import torch; print(torch.__version__)"` |
| Triton | `python -c "import triton; print(triton.__version__)"` |
| Repo commit | |
| Command line | (paste full `PYTHONPATH=src python scripts/profile_kernel.py ...`) |

## 1. Microbench (isolated kernel / reference)

Paste the **terminal output** block from `profile_kernel.py` (mean_ms / do_bench_ms).

| Path | ref mean_ms | triton mean_ms | triton do_bench_ms | Notes |
|------|-------------|----------------|---------------------|-------|
| draft Q·K hist | | | | |
| target Q·K hist | | | | |
| fused_verifier (if `--include-fused-verifier`) | | | | |

**Ratio** triton/ref (draft): ______  
**Ratio** triton/ref (target): ______  

## 2. Chrome trace (`--trace-out`)

| Field | Value |
|-------|--------|
| Trace file | `chrome://tracing` or Perfetto — attach `.json` |
| Profile path | draft / target (which Q·K kernel) |

### 2.1 Top CUDA kernels (paste 5–10 rows)

From script “Top events” or from trace: kernel names + self CUDA time.

| Kernel / op name | Approx % of CUDA time | Category (copy / launch / math) |
|------------------|------------------------|----------------------------------|
| | | |

### 2.2 Stacks (`with_stack=True`)

**Data movement:** List any `aten::copy_`, `slice`, `cat`, `contiguous`, `to()` **before** the Triton kernel:  
_______________________________________________________________________________

**Launch overhead:** Small grids (e.g. one program per hist position) vs batch — note sequence length **S**:  
_______________________________________________________________________________

**Compute:** Triton kernel name(s) for Q·K / fused verifier:  
_______________________________________________________________________________

## 3. Roofline (analytical, from script)

Paste script lines:

- `est_bytes_total` = ______  
- `est_flops` = ______  
- `arithmetic_intensity` (FLOP/byte) = ______  
- `if_memory_bound_at_600_GB/s` min time (µs) = ______  

**Interpretation:**  
Is AI low enough that **600 GB/s** HBM becomes the binding limit for the **Q·K hist** phase? ☐ Yes ☐ No ☐ Uncertain  

**Note:** Peak **compute** (TOPS) was not measured here; for a full roofline, add vendor peak FP32/FP16 throughput and compare `flops / peak_tops` vs `bytes / peak_gbps`.

## 4. Bottleneck hypothesis (Task 2 — checklist)

Mark what the evidence supports (can be multiple):

| Hypothesis | Evidence for | Evidence against |
|------------|--------------|------------------|
| **A. Separate dequant** — INT4→FP16 in Python/PyTorch before Triton | | |
| **B. Triton tiling** — `BLOCK_D` / `num_warps` not ideal for A10G SM / shared mem | | |
| **C. Python glue** — acceptance / KV rollback → `cpu()` / sync / small launches | | |
| **D. Memory bandwidth** — roofline + profiler shows memcpy-bound | | |
| **E. Kernel launch tax** — many tiny kernels vs one fused pass | | |

**Primary suspect (1–2 sentences):**  
_______________________________________________________________________________

## 5. Implementation plan (Task 3)

### Step 1 — Immediate (fusion / fewer HBM roundtrips)

Actions:  
_______________________________________________________________________________

Expected impact: ☐ ~5–15% ☐ ~15–40% ☐ unknown  

### Step 2 — Structural (fewer syncs / larger fused regions)

Actions:  
_______________________________________________________________________________

Expected impact: _______________________________________________________________________________

### Step 3 — Scaling (“break-even” vs HF AR)

At what **context length** / **model width** do you expect hierarchical+Triton to match AR **decode tok/s**?  
_______________________________________________________________________________

**Back-of-envelope:** Use Phase N CSV `tokens_per_sec_decode_phase` vs `comparison_mode`, and this profile’s **triton/ref** ratio to extrapolate.

## 6. Follow-up experiments

- [ ] `CUDA_LAUNCH_BLOCKING=1` single iteration + narrow trace  
- [ ] `ncu` / Nsight Compute on Triton kernel (memory throughput %)  
- [ ] Sweep `--seq-len` {128, 256, 512, 1024} at fixed `D=1280`  
- [ ] Sweep `kv_kernels.tuning` presets: `default`, `a10g_balanced`, `verifier_wide`  

---

*Template version: 1 — pairs with `scripts/profile_kernel.py`.*
