"""P5-only architecture, paired initialization and training-entry acceptance tests."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.d1 import run_p5_ablation as p5
from scripts.d1.p1p2_runtime import E1FrozenTrainer
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.modules import DINOFeaturePyramidAdapter
from ultralytics.utils import YAML


def features(grid=4, gradients=False):
    return {name: torch.randn(1, 384, grid, grid, requires_grad=gradients) for name in ("block4", "block8", "block12")}


def tensor_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}


@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_registered_parameter_counts_and_unchanged_interfaces(variant):
    model = p5.construct_model(variant)
    assert model.adapter.out_channels == (64, 128, 256)
    assert model.adapter.source_names == ("block4", "block8", "block12")
    assert tuple(model.stride.tolist()) == (8, 16, 32)
    assert model.detect.nc == 80 and model.detect.reg_max == 1 and model.end2end
    assert model.args.latent_aux_gain == 0.1
    assert all(m.value_fusion_mode == "weighted_sum" for m in model.mixtures.values())
    assert not any("teacher" in name.lower() for name, _ in model.named_parameters())


@pytest.mark.parametrize("variant", ("DW", "BN64"))
def test_formal_shapes_and_all_branch_gradients(variant):
    model = p5.construct_model(variant)
    adapter = model.adapter
    inputs = features(40, gradients=True)
    result = adapter(inputs)
    expected = {"p3": (1, 64, 80, 80), "p4": (1, 128, 40, 40), "p5": (1, 256, 20, 20)}
    assert tuple(result) == ("p3", "p4", "p5")
    for level, candidates in result.items():
        assert len(candidates) == 3
        for name, value in zip(adapter.source_names, candidates):
            assert tuple(value.shape) == expected[level] and bool(torch.isfinite(value).all())
            assert torch.equal(value, adapter.branches[level][name](inputs[name]))
    sum(value.square().mean() for group in result.values() for value in group).backward()
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in inputs.values())
    for group in adapter.branches.values():
        for branch in group.values():
            for module in branch.modules():
                if isinstance(module, nn.Conv2d):
                    assert module.weight.grad is not None
                    assert bool(torch.isfinite(module.weight.grad).all()) and module.weight.grad.abs().sum() > 0
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in adapter.modules())
    ids = [{id(p) for p in branch.parameters()} for group in adapter.branches.values() for branch in group.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(ids) for b in ids[i + 1:])
    branch = adapter.branches["p5"]["block4"]
    convs = [m for m in branch.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2 and all(m.bias is None for m in convs)
    if variant == "DW":
        assert (convs[0].groups, convs[0].stride, convs[0].padding) == (384, (2, 2), (1, 1))
        assert (convs[1].in_channels, convs[1].out_channels, convs[1].kernel_size) == (384, 256, (1, 1))
    else:
        assert (convs[0].in_channels, convs[0].out_channels) == (384, 64)
        assert (convs[1].in_channels, convs[1].out_channels, convs[1].groups) == (64, 256, 1)


@pytest.mark.parametrize("kwargs", [
    {"p5_mode": "typo"}, {"p5_mode": True}, {"p5_mode": ["conv"]},
    {"p5_mode": "bottleneck"}, {"p5_mode": "bottleneck", "p5_bottleneck_channels": True},
    {"p5_mode": "bottleneck", "p5_bottleneck_channels": 0},
    {"p5_mode": "bottleneck", "p5_bottleneck_channels": 384},
    {"p5_mode": "depthwise", "p5_bottleneck_channels": 64},
])
def test_invalid_p5_options_fail_closed(kwargs):
    with pytest.raises(ValueError):
        DINOFeaturePyramidAdapter(**kwargs)


def test_unknown_adapter_yaml_key_fails_closed():
    config = p5.model_config("DW")
    config["adapter"]["p5_mod"] = "conv"
    with pytest.raises(ValueError, match="unsupported adapter"):
        D1FoundationDetectionModel(config)


def test_default_state_dict_layout_remains_legacy_compatible():
    torch.manual_seed(0)
    old_style = DINOFeaturePyramidAdapter()
    torch.manual_seed(0)
    explicit = DINOFeaturePyramidAdapter(p5_mode="conv")
    assert set(old_style.state_dict()) == set(explicit.state_dict())
    for name in old_style.source_names:
        assert f"branches.p5.{name}.0.weight" in old_style.state_dict()
    explicit.load_state_dict(old_style.state_dict(), strict=True)
    assert sum(p.numel() for p in old_style.parameters()) == 2878080
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in old_style.state_dict().items())


@pytest.mark.parametrize("variant", ("DW", "BN64"))
def test_detection_loss_optimizer_and_strict_checkpoint_round_trip(variant, tmp_path):
    model = p5.construct_model(variant).train()
    inputs = features()
    batch = {
        "features": inputs, "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {k: v.detach().clone() for k, v in model.adapter.branches["p5"].named_parameters()}
    loss, items = model(batch)
    assert bool(torch.isfinite(loss).all()) and bool(torch.isfinite(items).all())
    assert model._mixture_aux_diagnostics["counts_by_kind"]["latent"] == 3
    loss.sum().backward()
    for name, parameter in model.adapter.branches["p5"].named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
    optimizer.step()
    for name, parameter in model.adapter.branches["p5"].named_parameters():
        assert not torch.equal(before[name], parameter), name
    model.eval()
    with torch.no_grad():
        expected = model(inputs)[0]
    payload = model.checkpoint_payload()
    path = tmp_path / "downstream.pt"
    torch.save(payload, path)
    restored = D1FoundationDetectionModel.from_checkpoint_payload(
        torch.load(path, weights_only=False, map_location="cpu")
    ).eval()
    with torch.no_grad():
        assert torch.equal(expected, restored(inputs)[0])
    assert restored.config_dict()["adapter"] == model.config_dict()["adapter"]
    wrong = p5.construct_model("BASE")
    with pytest.raises(RuntimeError):
        wrong.load_state_dict(model.state_dict(), strict=True)


def test_paired_initialization_copies_all_unchanged_tensors():
    source = tensor_state(p5.e1.construct_model("A"))
    shared = {}
    for variant in p5.VARIANTS:
        model = p5.construct_model(variant)
        initial = p5.paired_initial_state(model, source, variant)
        for key, value in initial.items():
            if variant == "BASE" or not key.startswith(p5.P5_PREFIX):
                assert torch.equal(value, source[key])
        shared[variant] = {k: v for k, v in initial.items() if not k.startswith(p5.P5_PREFIX)}
        assert all(m.value_fusion_mode == "weighted_sum" for m in model.mixtures.values())
    assert shared["DW"].keys() == shared["BN64"].keys() == shared["BASE"].keys()
    for variant in ("DW", "BN64"):
        assert all(torch.equal(v, shared[variant][k]) for k, v in shared["BASE"].items())


@pytest.mark.parametrize("corruption", ("missing", "shape", "nonfinite"))
def test_invalid_initial_state_rejected(corruption):
    source = tensor_state(p5.e1.construct_model("A"))
    key = next(iter(source))
    if corruption == "missing":
        source.pop(key)
    elif corruption == "shape":
        source[key] = torch.ones(2)
    else:
        source[key].fill_(float("nan"))
    with pytest.raises(ValueError):
        p5.paired_initial_state(p5.construct_model("DW"), source, "DW")


def test_p5_contract_and_model_changes_are_isolated(tmp_path, monkeypatch):
    original = deepcopy(p5.e1.load_contract())
    assert p5.load_contract()["screen_epochs"] == 50
    assert p5.e1.load_contract() == original
    contract = p5.load_contract()
    contract["screen_epochs"] = 100
    path = tmp_path / "bad.yaml"
    YAML.save(path, contract)
    with pytest.raises(ValueError):
        p5.load_contract(path)
    config = p5.model_config("DW")
    config["loss"]["latent_aux_gain"] = 0.0
    YAML.save(path, config)
    monkeypatch.setitem(p5.MODEL_FILES, "DW", path)
    with pytest.raises(ValueError, match="outside"):
        p5.model_config("DW")


def test_prepare_and_provenance_checks(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    (source_dir / "inputs").mkdir(parents=True)
    initial = tensor_state(p5.e1.construct_model("A"))
    torch.save(initial, source_dir / "inputs/initial-frozen.pt")
    (source_dir / "matrix.json").write_text("{}")
    source = {
        "data_root": "/nvme/coco", "cache_root": "/nvme/cache", "nvme_root": "/nvme",
        "data_receipt_sha256": "receipt", "data_files": {"E1": "/nvme/coco.yaml"},
    }
    source_identity = {"commit": "fixed", "source_sha256": "source", "contract_sha256": "contract"}
    monkeypatch.setattr(p5, "identity", lambda: source_identity)
    monkeypatch.setattr(p5.e1, "load_matrix", lambda *a, **k: source)
    work = tmp_path / "p5"
    matrix = p5.prepare(source_dir, work)
    assert matrix["training_started"] is False and not (work / "runs").exists()
    assert p5.load_matrix(work) == matrix
    overrides = []
    for variant in p5.VARIANTS:
        spec = p5.spec_for(matrix, variant)
        row = p5.overrides_for(matrix, spec)
        assert row["epochs"] == 100 and spec["window"] == 50
        assert row["batch"] == row["nbs"] == 384 and row["latent_aux_gain"] == 0.1
        assert row["workers"] == 4 and row["save_period"] == 5
        for key in ("name", "model"):
            row.pop(key)
        overrides.append(row)
    assert all(o == overrides[0] for o in overrides)
    with pytest.raises(FileExistsError):
        p5.prepare(source_dir, work)
    path = work / "inputs/model-DW.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="input changed"):
        p5.load_matrix(work)


def test_no_training_without_explicit_approval(monkeypatch):
    monkeypatch.setattr(p5, "load_matrix", lambda *a: pytest.fail("Must reject before accessing runtime"))
    with pytest.raises(ValueError, match="approved"):
        p5.train(SimpleNamespace(approved=False))
    with pytest.raises(SystemExit) as exc:
        p5.main(["train"])
    assert exc.value.code == 2


@pytest.mark.parametrize("variant", ("DW", "BN64"))
def test_trainer_model_factory_preserves_new_architecture_and_resume(variant, tmp_path):
    model = p5.construct_model(variant)
    path = tmp_path / "initial.pt"
    torch.save(tensor_state(model), path)
    trainer = object.__new__(p5.P5FrozenTrainer)
    trainer.data = {"nc": 80, "names": {i: str(i) for i in range(80)}}
    trainer.args = SimpleNamespace(cls_remap=True)
    trainer.e1 = {"seed": 0, "initial_state": str(path)}
    trainer.resume = False
    fresh = trainer.get_model(cfg=p5.model_config(variant), verbose=False)
    assert fresh.adapter.p5_mode == p5.VARIANTS[variant]["p5_mode"]
    model._mixture_loss_ema_buf.fill_(2.5)
    trainer.resume = True
    restored = trainer.get_model(cfg=p5.model_config(variant), weights=model, verbose=False)
    assert torch.equal(restored._mixture_loss_ema_buf, model._mixture_loss_ema_buf)
    assert p5.P5FrozenTrainer.evaluation_module == "scripts.d1.run_p5_ablation"
    assert E1FrozenTrainer.evaluation_module == "scripts.d1.run_p1p2"


def test_inspect_never_launches_training(monkeypatch):
    monkeypatch.setattr(p5.P5FrozenTrainer, "train", lambda *a: pytest.fail("Unexpected training"))
    result = p5.inspect()
    assert result["training_started"] is False and set(result["variants"]) == set(p5.VARIANTS)


@pytest.mark.parametrize("variant", ("DW", "BN64"))
def test_real_npy_and_labels_cpu_one_step(variant, tmp_path):
    cache_root = os.environ.get("D1_P5_CACHE_ROOT")
    rgb_root = os.environ.get("D1_P5_RGB_ROOT")
    if not cache_root or not rgb_root:
        pytest.skip("Optional real COCO NPY cache and RGB paths are not configured")
    from ultralytics.data.d1_cache import D1FeatureCacheDataset
    from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
    from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader
    from ultralytics.utils import DEFAULT_CFG_DICT

    reader = NpyFeatureCacheReader(Path(cache_root))
    sample_id = sorted(reader.records)[0]
    reader.verify_sample(sample_id)
    image_relative = Path(reader.records[sample_id]["image_path"])
    label_relative = Path("labels") / Path(sample_id).with_suffix(".txt")
    # Isolate the dataset's optional label-cache writes from the original dataset.
    for relative in (image_relative, label_relative):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(rgb_root) / relative, destination)
    listing = tmp_path / "one-image.txt"
    listing.write_text(str(tmp_path / image_relative) + "\n")
    dataset = D1FeatureCacheDataset(
        img_path=str(listing), cache_dir=Path(cache_root), data={"nc": 80, "names": dict(enumerate(map(str, range(80))))},
        imgsz=640, batch_size=1, hyp=SimpleNamespace(**DEFAULT_CFG_DICT), trusted_cache=True, prefetch_factor=1,
    )
    batch = dataset.collate_fn([dataset[0]])
    assert len(batch["cls"]) > 0 and all(v.dtype == torch.float16 for v in batch["features"].values())
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.device, trainer.amp = torch.device("cpu"), False
    batch = trainer.preprocess_batch(batch)
    model = p5.construct_model(variant).train()
    before = tensor_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss, items = model(batch)
    assert bool(torch.isfinite(loss).all()) and bool(torch.isfinite(items).all())
    loss.sum().backward()
    optimizer.step()
    changed = [not torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith(p5.P5_PREFIX)]
    assert all(changed)
    model.eval()
    restored = D1FoundationDetectionModel.from_checkpoint_payload(model.checkpoint_payload()).eval()
    with torch.no_grad():
        assert torch.equal(model(batch["features"])[0], restored(batch["features"])[0])
    reader.close()
