"""Tests for baillarger.backend: device names, the default, and refusals."""

import pytest
import torch

from baillarger import backend


def test_device_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAILLARGER_DEVICE", raising=False)
    assert backend.device("numpy").is_numpy
    c = backend.device("cpu")
    assert c.name == "cpu" and c.torch == torch.device("cpu")
    assert backend.device().name == backend.available()[0]
    assert backend.available()[-2:] == ["cpu", "numpy"]
    monkeypatch.setenv("BAILLARGER_DEVICE", "numpy")
    assert backend.device().is_numpy
    with pytest.raises(ValueError, match="not one of"):
        backend.device("tpu")


def test_an_unavailable_device_is_an_error() -> None:
    for name in ("cuda", "mps"):
        if name not in backend.available():
            with pytest.raises(ValueError, match="not available"):
                backend.device(name)
