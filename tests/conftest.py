from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def clear_service_caches():
    from exoplanet_detector.app import services

    services.get_run_context.cache_clear()
    services._load_example_dataset.cache_clear()
    yield
    services.get_run_context.cache_clear()
    services._load_example_dataset.cache_clear()

