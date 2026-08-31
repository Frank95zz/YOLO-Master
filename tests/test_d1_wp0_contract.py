"""Contracts for the D1 WP0 dataset, teacher, preprocessing, and provenance lock."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml"
CONTRACT = ROOT / "experiments/d1/manifests/p0-experiment-contract.json"
SCRIPT = ROOT / "scripts/prepare_d1_wp0.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_d1_wp0", SCRIPT)
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
    module = _load_script()
    lines = ["images/train2017/000000000001.jpg", "images/train2017/000000000002.jpg"]
    list_path = tmp_path / "split.txt"
    first_hash = module.write_lines(list_path, lines)
    first_bytes = list_path.read_bytes()
    second_hash = module.write_lines(list_path, list(reversed(list(reversed(lines)))))
    assert list_path.read_bytes() == first_bytes
    assert second_hash == first_hash

    manifest_path = tmp_path / "manifest.json"
    module.write_json(manifest_path, {"z": 1, "a": {"value": True}})
    first_bytes = manifest_path.read_bytes()
    module.write_json(manifest_path, {"a": {"value": True}, "z": 1})
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
    assert module.COCO_MIRROR_REVISION == "5466a7f1944225fcddb1896006508cad5be27b5b"
    assert module.COCO_FILES["train2017.zip"][2:] == (
        19_336_861_798,
        "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929",
    )
    assert module.COCO_FILES["val2017.zip"][2:] == (
        815_585_330,
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    )


def test_download_only_promotes_hash_verified_part_file(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    payload = b"verified payload"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    destination = tmp_path / "artifact.bin"
    expected_sha256 = module.hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/curl")
    module.download_file(
        source.as_uri(),
        destination,
        expected_size=len(payload),
        expected_sha256=expected_sha256,
    )
    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()


def test_extract_zip_is_repeatable_and_records_archive_digest(tmp_path: Path) -> None:
    module = _load_script()
    archive_path = tmp_path / "sample.zip"
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/a.jpg", b"a")
        archive.writestr("images/b.jpg", b"bb")
    (destination / "images").mkdir(parents=True)
    (destination / "images/a.jpg").write_bytes(b"a")

    module.extract_zip(archive_path, destination)
    module.extract_zip(archive_path, destination)

    assert (destination / "images/a.jpg").read_bytes() == b"a"
    assert (destination / "images/b.jpg").read_bytes() == b"bb"
    marker = destination / ".sample.zip.extracted"
    assert marker.read_text(encoding="utf-8").strip() == module.sha256_file(archive_path)
