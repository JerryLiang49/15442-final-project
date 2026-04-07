# Benchmark documentation

Canonical docs live under **`docs/`** (this file is only a stable entrypoint):

- **[`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md)** — pytest gate, checklist, frozen vocabulary  
- **[`docs/BENCHMARK_PHASE15.md`](docs/BENCHMARK_PHASE15.md)** — sweep mode labels, strict grid, CSV schema notes  

Run the gate from the repo root:

```bash
PYTHONPATH=src pytest -m benchmark_gate -q
```
