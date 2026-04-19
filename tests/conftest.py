"""Pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads HF weights / longer CPU tests")
    config.addinivalue_line(
        "markers",
        "parity_cuda: Phase M fused-kernel parity tests that require CUDA + Triton (skipped on CPU CI)",
    )
    config.addinivalue_line(
        "markers",
        "benchmark_gate: Phase 14 sweep gate (modes × prompts × K); run before Phase 15 benchmarks",
    )
    config.addinivalue_line(
        "markers",
        "benchmark_gate: Phase 14 end-to-end gate before benchmark sweeps (run with -m benchmark_gate)",
    )
