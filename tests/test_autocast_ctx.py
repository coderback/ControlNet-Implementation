import torch
import types
from src.training.train_controlnet import autocast_ctx


def test_autocast_ctx_cpu_smoke():
    ctx = autocast_ctx(True, "cpu")
    with ctx:
        x = torch.tensor([1.0], dtype=torch.float32)
        y = x + 1.0
    assert y.item() == 2.0


def test_autocast_ctx_cuda_monkeypatched(monkeypatch):
    class DummyAutocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "amp", types.SimpleNamespace(autocast=lambda: DummyAutocast())
    )

    ctx = autocast_ctx(True, "cuda")
    with ctx:
        pass
