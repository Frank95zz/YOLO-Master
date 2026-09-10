"""Pre-registered three-seed BN64 matrix, isolated probes, and failure gates."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.d1 import run_e3 as e3
from scripts.d1.e3_mechanism import probe
from scripts.d1.p1p2_runtime import E1Policy, load_initial_tensors
from ultralytics.utils import DEFAULT_CFG_DICT, YAML


def fake_matrix(tmp_path):
    hashes = {f"initial-s{s}.pt": str(s) * 64 for s in (0, 1, 2)}
    for b in e3.EXPECTED["balance_candidates"]:
        for z in e3.EXPECTED["z_candidates"]:
            for gain in e3.EXPECTED["stage2_gains"]:
                cid = e3.key({"balance": b, "z": z, "gain": gain, "seed": 0})
                hashes[cid.rsplit("-s", 1)[0] + ".yaml"] = "a" * 64
    return {
        "workspace": str(tmp_path),
        "identity": {"commit": "b" * 40},
        "reference_sha256": "c" * 64,
        "inputs_sha256": hashes,
    }


def test_three_seed_complete_matrix():
    assert e3.load_contract() == e3.EXPECTED
    rows = e3.candidates()
    assert len(rows) == len({e3.key(c) for c in rows}) == 27
    assert {c["seed"] for c in rows} == {0, 1, 2}
    assert rows[:3] == [{"balance": 0.01, "z": 0.001, "gain": 0.1, "seed": s} for s in range(3)]
    assert 27 + 3 * 3 == 36
    assert all(e3.resolve_candidate(e3.key(c)) == c for c in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [0, 1, 2, 3]),
        ("screen_epochs", 50),
        ("global_batch", 384),
        ("architecture", "BASE"),
        ("official_backend", "coco"),
    ],
)
def test_contract_drift_rejected(tmp_path, field, value):
    contract = deepcopy(e3.EXPECTED)
    contract[field] = value
    path = tmp_path / "bad.yaml"
    YAML.save(path, contract)
    with pytest.raises(ValueError, match="Unregistered"):
        e3.load_contract(path)


@pytest.mark.parametrize(
    "field,value", [("seed", 3), ("seed", True), ("seed", 1.0), ("gain", -1), ("balance", float("nan")), ("z", 2)]
)
def test_bad_candidate_rejected(field, value):
    candidate = {**e3.candidates()[0], field: value}
    with pytest.raises(ValueError):
        e3.key(candidate)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paired_initialization_and_recipe(tmp_path, seed):
    torch.set_num_threads(1)
    baseline = {**e3.candidates()[0], "seed": seed}
    off = {**baseline, "balance": 0.0, "z": 0.0}
    a, b = e3.construct_model(baseline), e3.construct_model(off)
    tensors = {k: v for k, v in a.state_dict().items() if isinstance(v, torch.Tensor)}
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in tensors.items())
    load_initial_tensors(b, tensors)
    assert b.mixtures["p3"].balance_loss_coeff == 0
    assert a.detect.nc == 10 and a.detect.max_det == 500
    assert sum(p.numel() for p in a.parameters()) == e3.PARAMETERS
    assert not any("teacher" in n.lower() for n, _ in a.named_parameters())
    mat = fake_matrix(tmp_path)
    first, second = e3.spec_for(mat, baseline), e3.spec_for(mat, off)
    assert first["initial_state"] == second["initial_state"]
    assert first["identity"]["initial_sha256"] == second["identity"]["initial_sha256"]
    args = e3.overrides(mat, first)
    assert args["seed"] == seed and args["epochs"] == 300 and first["window"] == 60
    assert args["batch"] == args["nbs"] == 96 and args["workers"] == 4
    assert args["amp"] and args["deterministic"] and not args["pretrained"]
    assert all(args[k] == 0 for k in ("mosaic", "mixup", "fliplr", "multi_scale", "copy_paste"))
    assert E1Policy.required_schedule_epochs == 100
    assert E1Policy.required_global_batch == 384


def test_different_seeds_different_parameters():
    torch.set_num_threads(1)
    c = e3.candidates()[0]
    a, b = e3.construct_model(c), e3.construct_model({**c, "seed": 1})
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize(
    "balance,z,gain", [(0.01, 0.001, 0.1), (0, 0, 0.1), (0.01, 0, 0.1), (0, 0.001, 0.1), (0.01, 0.001, 0)]
)
def test_probe_actual_coefficients_and_source_isolation(balance, z, gain):
    torch.set_num_threads(1)
    c = {"balance": balance, "z": z, "gain": gain, "seed": 0}
    model = e3.construct_model(c).train()
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, "latent_aux_gain": gain, "mixture_aux_budget": 3.0})
    model.criterion = model.init_criterion()
    features = {name: torch.randn(1, 384, 8, 8) for name in ("block4", "block8", "block12")}
    batch = {
        "img": features,
        "features": features,
        "batch_idx": torch.zeros(1),
        "cls": torch.tensor([[9.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    original = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    report = probe(model, batch)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert all(p.grad is None for p in model.parameters())
    assert set(report["scales"]) == {"p3", "p4", "p5"}
    assert all(s["balance_coeff"] == balance and s["z_coeff"] == z for s in report["scales"].values())
    if gain == 0 or balance == z == 0:
        assert report["effective_aux"] == 0
        assert all(row["aux"] == 0 for row in report["gradients"].values())
    assert report["scales"]["p3"]["raw_z"] > 0


def ranking_rows():
    return [{**c, "AP_all": 0.2 + c["seed"] * 0.01, "training_gpu_hours": 1.0} for c in e3.candidates()]


def test_ranking_all_zero_can_win_but_next_stage_uses_nonzero():
    rows = ranking_rows()
    for row in rows:
        if row["balance"] == row["z"] == 0:
            row["AP_all"] = 0.9
    groups, winner = e3.rank_groups(rows)
    assert winner["balance"] == 0 and winner["z"] == 0.001
    assert winner["AP_mean"] == pytest.approx(0.21)
    assert winner["AP_sample_std"] == pytest.approx(0.01)
    assert max(g["AP_mean"] for g in groups) == 0.9


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nan", "percent", "seed3"])
def test_rank_fails_closed(failure):
    rows = ranking_rows()
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "seed3":
        rows[-1]["seed"] = 3
    else:
        rows[0]["AP_all"] = float("nan") if failure == "nan" else 20
    with pytest.raises(ValueError):
        e3.rank_groups(rows)


def test_no_e1_tolerance_in_ranking():
    rows = ranking_rows()
    for row in rows:
        if row["balance"] == 0.1 and row["z"] == 0.01:
            row["AP_all"] += 1e-7
    _, winner = e3.rank_groups(rows)
    assert (winner["balance"], winner["z"]) == (0.1, 0.01)


def test_suite_requires_approval(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        e3.suite(SimpleNamespace(workspace=tmp_path, approved=False))


def test_official_report_rejects_internal_map():
    with pytest.raises(ValueError, match="MATLAB"):
        e3.validate_official_report({"backend": "ultralytics", "metrics": {"AP_all": 0.5}})


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["/fins/.venv/bin/python", "/fins/.venv/bin/tensorboard", "--logdir=/fins/runs"], None),
        (["/fins/site-packages/tensorboard_data_server/bin/server", "--logdir=/fins/runs"], None),
        (["/fins/.venv/bin/python", "/fins/train.py"], "simulation"),
        (["/fins/Unity.x86_64"], "simulation"),
        (["python", "-m", "torch.distributed.run"], "training"),
        (["python", "gpu_keeper.py"], None),
    ],
)
def test_resource_check_distinguishes_viewers(argv, expected):
    assert e3.competing_job(argv) == expected
