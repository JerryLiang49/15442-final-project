"""Pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads HF weights / longer CPU tests")
