"""Regression coverage for D1 final validation and NPY checkpoint diagnostics."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ultralytics.models.yolo.detect import foundation_train
from ultralytics.utils.errors import MoERouterError


class DummyModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def config_dict(self):
        return {}


def final_eval_fixture(tmp_path, monkeypatch, rank, result=None, failure=None):
    monkeypatch.setattr(foundation_train, "RANK", rank)
    monkeypatch.setattr(foundation_train, "LOCAL_RANK", rank)
    monkeypatch.setattr(foundation_train, "D1FoundationDetectionModel", DummyModel)
    monkeypatch.setattr(foundation_train, "torch_distributed_zero_first", lambda _: nullcontext())
    monkeypatch.setattr(foundation_train, "strip_optimizer", lambda *args, **kwargs: {})
    monkeypatch.setattr(foundation_train, "load_checkpoint", lambda *args, **kwargs: (DummyModel(), {}))
    trainer = object.__new__(foundation_train.D1FoundationDetectionTrainer)
    trainer.model = DummyModel()
    trainer.ema = SimpleNamespace(ema=DummyModel())
    trainer.args = SimpleNamespace(plots=False)
    trainer.device = torch.device("cpu")
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.last.touch()
    trainer.best.touch()
    trainer._select_final_eval_checkpoints = lambda: ([trainer.best], [])
    trainer._reset_non_checkpoint_moe_runtime_state = lambda: None
    trainer.metrics = {"previous": 1.0}
    trainer._d1_final_eval_checkpoint = None
    calls = []
    callbacks = []

    class Validator:
        args = SimpleNamespace(plots=False)

        def __call__(self, *, trainer):
            calls.append(trainer._d1_final_eval_checkpoint)
            assert trainer.ema.ema is None
            if failure is not None:
                raise failure
            return result

    trainer.validator = Validator()
    trainer.run_callbacks = lambda event: callbacks.append((event, trainer._d1_final_eval_checkpoint))
    return trainer, calls, callbacks


@pytest.mark.parametrize("rank", [-1, 0, 1, 2, 3, 4, 5])
def test_final_eval_all_ranks_validate_only_main_consumes_metrics(tmp_path, monkeypatch, rank):
    result = {"fitness": 0.2, "metrics/mAP50-95(B)": 0.1} if rank <= 0 else None
    trainer, calls, callbacks = final_eval_fixture(tmp_path, monkeypatch, rank, result=result)
    original_model, original_ema = trainer.model, trainer.ema.ema
    trainer.final_eval()
    assert calls == [trainer.best]
    assert trainer.model is original_model
    assert trainer.ema.ema is original_ema
    assert trainer._d1_final_eval_checkpoint is None
    if rank <= 0:
        assert trainer.metrics == {"metrics/mAP50-95(B)": 0.1}
        assert callbacks == [("on_fit_epoch_end", trainer.best)]
    else:
        assert trainer.metrics == {"previous": 1.0}
        assert callbacks == []


@pytest.mark.parametrize("failure", [None, RuntimeError("validator failed")])
def test_final_eval_main_rank_fails_closed_and_restores_state(tmp_path, monkeypatch, failure):
    trainer, _, callbacks = final_eval_fixture(tmp_path, monkeypatch, 0, failure=failure)
    original_model, original_ema = trainer.model, trainer.ema.ema
    exception = TypeError if failure is None else RuntimeError
    with pytest.raises(exception):
        trainer.final_eval()
    assert trainer.model is original_model
    assert trainer.ema.ema is original_ema
    assert trainer._d1_final_eval_checkpoint is None
    assert callbacks == []


def test_final_eval_fallback_records_the_checkpoint_actually_evaluated(tmp_path, monkeypatch):
    trainer, _, callbacks = final_eval_fixture(tmp_path, monkeypatch, 0)
    trainer._select_final_eval_checkpoints = lambda: ([trainer.best, trainer.last], [])

    class Validator:
        args = SimpleNamespace(plots=False)

        def __call__(self, *, trainer):
            if trainer._d1_final_eval_checkpoint == trainer.best:
                raise MoERouterError("unhealthy best router")
            return {"fitness": 0.1}

    trainer.validator = Validator()
    trainer.final_eval()
    assert callbacks == [("on_fit_epoch_end", trainer.last)]
    assert trainer._d1_final_eval_checkpoint is None
