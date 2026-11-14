"""Tests for the optional multi-GPU switch, ``train.data_parallel``.

None of these needs a GPU. They pin when the training forward pass gets split
across GPUs and when it quietly stays on one device, because a switch that
silently did nothing -- or wrapped on the wrong device -- would cost a long run
before anyone noticed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from morg2smiles.train import TrainConfig, wrap_data_parallel


@pytest.fixture
def model() -> nn.Module:
    return nn.Linear(4, 2)


def test_off_by_default_and_round_trips_through_config():
    assert TrainConfig().data_parallel is False
    cfg = TrainConfig.from_dict({"data_parallel": True})
    assert cfg.data_parallel is True
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_disabled_returns_the_model_itself(model):
    assert wrap_data_parallel(model, torch.device("cuda"), enabled=False) is model


def test_ignored_off_cuda(model):
    assert wrap_data_parallel(model, torch.device("cpu"), enabled=True) is model
    assert wrap_data_parallel(model, torch.device("mps"), enabled=True) is model


def test_ignored_with_fewer_than_two_gpus(model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert wrap_data_parallel(model, torch.device("cuda"), enabled=True) is model


def test_refuses_a_model_placed_on_a_later_gpu(model):
    """DataParallel gathers on the first visible GPU; the model must live there."""
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        wrap_data_parallel(model, torch.device("cuda", 1), enabled=True)


class _RecordingDataParallel(nn.Module):
    """Stands in for nn.DataParallel, whose own setup queries the real hardware."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


def test_wraps_when_several_gpus_are_visible(model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(nn, "DataParallel", _RecordingDataParallel)
    for device in (torch.device("cuda"), torch.device("cuda", 0)):
        net = wrap_data_parallel(model, device, enabled=True)
        assert isinstance(net, _RecordingDataParallel)
        # The optimiser and checkpointing use `model`; the wrapper must not copy it.
        assert net.module is model


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two or more CUDA GPUs")
def test_splits_a_real_batch_and_matches_one_device():
    torch.manual_seed(0)
    model = nn.Linear(16, 3).to("cuda:0")
    net = wrap_data_parallel(model, torch.device("cuda", 0), enabled=True)
    assert isinstance(net, nn.DataParallel)
    x = torch.randn(64, 16, device="cuda:0")
    out = net(x)
    assert out.device == torch.device("cuda", 0)
    torch.testing.assert_close(out, model(x))
    out.sum().backward()
    assert model.weight.grad is not None and model.weight.grad.device == torch.device("cuda", 0)
