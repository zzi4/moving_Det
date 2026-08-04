from pathlib import Path

import pytest

from moving_det.config import load_config


@pytest.fixture
def config():
    return load_config(Path("configs/poc.yaml"))
