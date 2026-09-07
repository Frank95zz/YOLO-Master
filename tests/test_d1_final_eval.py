"""Regression coverage for D1 final validation and NPY checkpoint diagnostics."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.d1 import diagnose_wp8, run_wp8_train
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


def test_final_telemetry_does_not_overwrite_last_epoch(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    telemetry = run_wp8_train.EpochTelemetry(tmp_path)
    trainer = SimpleNamespace(epoch=99, validator=SimpleNamespace(seen=5000), metrics={"ap": 0.1})
    telemetry.on_fit_epoch_end(trainer)
    epoch_path = tmp_path / "validation/epoch-099.json"
    epoch_bytes = epoch_path.read_bytes()
    trainer._d1_final_eval_checkpoint = Path("best.pt")
    trainer.metrics = {"ap": 0.2}
    telemetry.on_fit_epoch_end(trainer)
    assert epoch_path.read_bytes() == epoch_bytes
    final = json.loads((tmp_path / "validation/final-best.json").read_text())
    assert final["phase"] == "final"
    assert final["checkpoint"] == "best.pt"
    assert final["metrics"] == {"ap": 0.2}
    telemetry.rank = 1
    trainer._d1_final_eval_checkpoint = Path("last.pt")
    telemetry.on_fit_epoch_end(trainer)
    assert not (tmp_path / "validation/final-last.json").exists()


@pytest.mark.parametrize("mode", ["valid", "missing", "wrong_checkpoint", "incomplete"])
def test_summary_requires_checkpoint_specific_final_report(tmp_path, monkeypatch, mode):
    runner = run_wp8_train
    run_root, report_dir = tmp_path / "run", tmp_path / "reports"
    (run_root / "weights").mkdir(parents=True)
    checkpoint = run_root / "weights/best.pt"
    checkpoint.touch()
    runner.write_json(report_dir / "preflight.json", {"identity": {"test": True}})
    runner.write_json(
        report_dir / "epochs/rank-00-epoch-000.json",
        {
            "rank": 0,
            "data_wait_seconds": 1.0,
            "step_seconds": 9.0,
            "peak_gpu_bytes": 1,
        },
    )
    validation = {
        "phase": "final",
        "checkpoint": "last.pt" if mode == "wrong_checkpoint" else "best.pt",
        "validator_seen": 4999 if mode == "incomplete" else 5000,
        "metrics": {"ap": 0.2},
    }
    runner.write_json(report_dir / "validation/epoch-000.json", validation)
    if mode != "missing":
        runner.write_json(report_dir / "validation/final-best.json", validation)
    monkeypatch.setattr(
        runner,
        "load_contract",
        lambda _: {
            "acceptance": {"require_epochs": 1},
            "hardware": {"world_size": 1},
        },
    )
    monkeypatch.setattr(runner, "read_results_csv", lambda _: [{"epoch": 1, "ap": 0.1}])
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: (DummyModel(), {}))
    monkeypatch.setattr(runner, "D1FoundationDetectionModel", DummyModel)
    monkeypatch.setattr(runner, "initialize_mixture_loss_ema_buffer", lambda _: None)
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "routing_deltas", lambda *args: {})
    args = SimpleNamespace(config=None, report_dir=report_dir, run_root=run_root)
    if mode != "valid":
        with pytest.raises(RuntimeError, match="failed acceptance"):
            runner.summarize(args)
    else:
        report = runner.summarize(args)
        assert report["final_metrics"] == {"ap": 0.2}
        assert report["last_epoch_metrics"] == {"epoch": 1, "ap": 0.1}
        assert report["final_validation_report"] == "final-best.json"


@pytest.mark.parametrize("npy", [False, True])
def test_diagnostic_identity_uses_the_format_aware_reader(tmp_path, monkeypatch, npy):
    index = {
        "schema_version": "d1-npy-cache-v1" if npy else "d1-cache-index-v1",
        "contract_sha256": "a" * 64,
    }
    if npy:
        index["source_content_sha256"] = "b" * 64
    else:
        index.update(content_sha256="b" * 64, shards=[{}])
    reader = SimpleNamespace(index=index, records={"one": {}})
    monkeypatch.setattr(diagnose_wp8, "open_feature_cache", lambda *args, **kwargs: reader)
    identity = diagnose_wp8._cache_identity(tmp_path)
    assert identity["content_sha256"] == "b" * 64
    assert identity["sample_count"] == 1
    assert identity["shard_count"] == (0 if npy else 1)
