"""Contracts for the D1 WP0 dataset, teacher, preprocessing, and provenance lock."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml"
CONTRACT = ROOT / "experiments/d1/manifests/p0-experiment-contract.json"
SCRIPT = ROOT / "scripts/d1/prepare_wp0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_wp0", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)


def test_wp0_recipe_locks_full_coco_and_vits16_without_random_augmentation() -> None:
    recipe = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert recipe["data"] == "coco.yaml"
    assert recipe["imgsz"] == 640
    assert recipe["seed"] == 0
    assert recipe["deterministic"] is True
    assert recipe["foundation_enabled"] is False
    assert recipe["foundation_teacher"] == "dinov3"
    assert recipe["foundation_model"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert recipe["foundation_teacher_dtype"] == "fp16"
    assert recipe["foundation_target_levels"] == ["p3", "p4", "p5"]
    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
    ):
        assert recipe[key] == 0.0

    tracked_text = CONFIG.read_text(encoding="utf-8").lower()
    assert "coco8" not in tracked_text
    assert "coco-mini" not in tracked_text
    assert "/data/" not in tracked_text


def test_wp0_contract_locks_preprocessing_blocks_and_cache_schema() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "d1-p0-contract-v1"
    assert contract["dataset"]["splits"] == {"train2017": 118287, "val2017": 5000}
    assert contract["dataset"]["class_count"] == 80
    assert contract["teacher"]["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert contract["features"] == {
        "grid_size": [40, 40],
        "hidden_size": 384,
        "output_blocks": [
            {"implementation_index": 3, "name": "block4", "ordinal": 4},
            {"implementation_index": 7, "name": "block8", "ordinal": 8},
            {"implementation_index": 11, "name": "block12", "ordinal": 12},
        ],
        "patch_size": 16,
        "raw_stride": 16,
    }
    assert contract["input"]["letterbox"] == {
        "auto": False,
        "center": True,
        "interpolation": "INTER_LINEAR",
        "padding_value": 114,
        "scale_fill": False,
        "scaleup": True,
        "stride": 32,
    }
    assert contract["input"]["normalize"]["mean"] == [0.485, 0.456, 0.406]
    assert contract["input"]["normalize"]["std"] == [0.229, 0.224, 0.225]
    assert contract["input"]["teacher_extra_crop"] is False
    assert contract["input"]["teacher_extra_resize"] is False
    assert contract["cache"]["schema_version"] == "d1-cache-v1"
    assert contract["cache"]["dtype"] == "float16"
    assert contract["cache"]["target_shard_bytes"] == 2 * 1024**3


def test_tracked_contract_has_no_host_paths_or_credentials() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    forbidden = ("/data/", "/root/", "10.210.", "password", "token", "authorization")
    for text in _walk_strings(contract):
        lowered = text.lower()
        assert not any(value in lowered for value in forbidden)


def test_split_list_is_sorted_stable_and_exact(tmp_path: Path) -> None:
    module = _load_script()
    module.EXPECTED_SPLITS = {"train2017": 3}
    image_dir = tmp_path / "images/train2017"
    image_dir.mkdir(parents=True)
    for name in ("000000000003.jpg", "000000000001.jpg", "000000000002.jpg"):
        (image_dir / name).write_bytes(name.encode())

    first = module.build_split_list(tmp_path, "train2017")
    second = module.build_split_list(tmp_path, "train2017")

    assert first == [
        "images/train2017/000000000001.jpg",
        "images/train2017/000000000002.jpg",
        "images/train2017/000000000003.jpg",
    ]
    assert first == second


@pytest.mark.parametrize("corrupt", [False, True])
def test_materialize_splits_preserves_provenance_and_fails_closed(tmp_path, corrupt):
    module = _load_script()
    module.EXPECTED_SPLITS = {"train2017": 2, "val2017": 1}
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    manifests = repo / "experiments/d1/manifests"
    manifests.mkdir(parents=True)
    expected = {}
    for split, ids in {"train2017": [1, 2], "val2017": [3]}.items():
        directory = data / "images" / split
        directory.mkdir(parents=True)
        lines = []
        for image_id in ids:
            filename = f"{image_id:012d}.jpg"
            (directory / filename).write_bytes(b"fixture")
            lines.append(f"images/{split}/{filename}")
        expected[split] = {
            "count": len(lines),
            "sha256": module.hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        }
    if corrupt:
        expected["val2017"]["sha256"] = "0" * 64
    source = manifests / "coco2017-splits.json"
    source.write_text(json.dumps({"splits": expected}))
    original = source.read_bytes()
    if corrupt:
        with pytest.raises(ValueError, match="published split contract"):
            module.materialize_splits(data, repo)
        assert not list(manifests.glob("*.txt"))
    else:
        module.materialize_splits(data, repo)
        first = {p.name: p.read_bytes() for p in manifests.glob("*.txt")}
        module.materialize_splits(data, repo)
        assert first == {p.name: p.read_bytes() for p in manifests.glob("*.txt")}
        assert len(first) == 2
    assert source.read_bytes() == original


def test_split_overlap_is_compared_by_image_filename() -> None:
    module = _load_script()
    train = ["images/train2017/000000000001.jpg"]
    val = ["images/val2017/000000000001.jpg"]
    try:
        module.assert_disjoint_splits(train, val)
    except ValueError as exc:
        assert "000000000001.jpg" in str(exc)
    else:
        raise AssertionError("same COCO image id must not appear in both splits")


def test_manifest_writers_are_deterministic(tmp_path: Path) -> None:
    from scripts.d1.artifacts import write_json

    module = _load_script()
    lines = ["images/train2017/000000000001.jpg", "images/train2017/000000000002.jpg"]
    list_path = tmp_path / "split.txt"
    first_hash = module.write_lines(list_path, lines)
    first_bytes = list_path.read_bytes()
    second_hash = module.write_lines(list_path, list(reversed(list(reversed(lines)))))
    assert list_path.read_bytes() == first_bytes
    assert second_hash == first_hash

    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"z": 1, "a": {"value": True}})
    first_bytes = manifest_path.read_bytes()
    write_json(manifest_path, {"a": {"value": True}, "z": 1})
    assert manifest_path.read_bytes() == first_bytes


def test_modelscope_vits16_contract_matches_expected_architecture() -> None:
    module = _load_script()
    assert module.MODEL_ID == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert module.EXPECTED_MODEL_CONFIG == {
        "hidden_size": 384,
        "model_type": "dinov3_vit",
        "num_attention_heads": 6,
        "num_hidden_layers": 12,
        "num_register_tokens": 4,
        "patch_size": 16,
    }
    assert module.MODEL_REVISION == "2e601320d0545509ab03374e2f8707f303e1de7a"
    assert module.MODEL_FILES["model.safetensors"] == (
        86_406_384,
        "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
    )


def test_teacher_verification_rejects_same_size_corruption(tmp_path, monkeypatch):
    module = _load_script()
    payloads = {
        "config.json": json.dumps(module.EXPECTED_MODEL_CONFIG).encode(),
        "model.safetensors": b"test-weights",
        "LICENSE.md": b"license",
        "README.md": b"readme",
    }
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(
        module,
        "MODEL_FILES",
        {name: (len(data), module.hashlib.sha256(data).hexdigest()) for name, data in payloads.items()},
    )
    assert module.verify_model(tmp_path, load_model=False)["model_loaded"] is False
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"X" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="SHA256"):
        module.verify_model(tmp_path, load_model=False)


def test_label_count_and_membership_are_checked(tmp_path):
    module = _load_script()
    directory = tmp_path / "labels/train2017"
    directory.mkdir(parents=True)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "instances_train2017.json").write_text("{}")
    label = directory / "a.txt"
    label.write_text("")
    lists = {"train2017": ["images/train2017/a.jpg", "images/train2017/b.jpg"]}
    assert module.verify_labels(tmp_path, lists, {"train2017": 1}) == {"train2017": 1}
    with pytest.raises(ValueError, match="count"):
        module.verify_labels(tmp_path, lists, {"train2017": 2})
    label.rename(directory / "unknown.txt")
    with pytest.raises(ValueError, match="membership"):
        module.verify_labels(tmp_path, lists, {"train2017": 1})


def test_prepare_rejects_removed_download_mode():
    module = _load_script()
    with pytest.raises(SystemExit):
        module.main(["--download"])
