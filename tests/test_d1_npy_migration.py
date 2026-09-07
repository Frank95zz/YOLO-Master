"""Destructive migration must fail closed and survive each commit boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.d1 import convert_cache_to_npy as migration


@pytest.fixture
def bundle(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    shape = (2, 4, 4)
    name = "train2017-r00-00000.safetensors"
    records, tensors = [], {}
    for i in range(2):
        sid = f"train2017/{i + 1:012d}"
        record = {"sample_id": sid, "split": "train2017", "shard": name, "tensors": {}}
        for j, feature in enumerate(migration.NAMES):
            key = f"sample{i}.{feature}"
            value = np.full(shape, i + j / 4, dtype=np.float16)
            tensors[key] = value
            record["tensors"][feature] = {
                "key": key,
                "shape": list(shape),
                "dtype": "float16",
                "nbytes": value.nbytes,
                "sha256": migration.sha(value.tobytes()),
            }
        records.append(record)
    path = source / name
    save_file(tensors, path)
    shard = {
        "filename": name,
        "split": "train2017",
        "sample_count": 2,
        "bytes": path.stat().st_size,
        "sha256": migration.file_sha(path),
    }
    return source, output, shard, records, "source-index-hash", shape


def run(bundle):
    source, output, shard, records, identity, shape = bundle
    return migration.convert_shard(source, output, shard, records, identity, shape=shape)


def test_lossless_conversion_retires_only_verified_shard(bundle):
    receipt = run(bundle)
    source, output, shard, records, _, shape = bundle
    assert receipt["state"] == "SOURCE_REMOVED"
    assert receipt["verified_tensor_count"] == 6
    assert not (source / shard["filename"]).exists()
    for record in records:
        migration.check_npy(migration.npy_path(output, record["sample_id"]), record, shape)
    assert run(bundle) == receipt


def test_source_checksum_failure_preserves_source(bundle):
    bundle[2]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        run(bundle)
    assert (bundle[0] / bundle[2]["filename"]).exists()
    assert not list(bundle[1].rglob("*.npy"))


def test_wrong_tensor_digest_preserves_source(bundle):
    bundle[3][1]["tensors"]["block8"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        run(bundle)
    assert (bundle[0] / bundle[2]["filename"]).exists()


def test_corrupt_existing_npy_fails_without_overwrite(bundle):
    path = migration.npy_path(bundle[1], bundle[3][0]["sample_id"])
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run(bundle)
    assert path.read_bytes() == b"corrupt"
    assert (bundle[0] / bundle[2]["filename"]).exists()


def test_partial_file_is_rebuilt(bundle):
    path = migration.npy_path(bundle[1], bundle[3][0]["sample_id"])
    path.parent.mkdir()
    path.with_name(path.name + ".part").write_bytes(b"interrupted")
    run(bundle)
    assert path.is_file()
    assert not list(bundle[1].rglob("*.part"))


def test_interrupt_before_receipt_never_deletes_source(bundle, monkeypatch):
    original = migration.atomic_json

    def fail_receipt(path, value):
        if value.get("state") == "VERIFIED":
            raise InterruptedError("before receipt")
        return original(path, value)

    monkeypatch.setattr(migration, "atomic_json", fail_receipt)
    with pytest.raises(InterruptedError):
        run(bundle)
    assert (bundle[0] / bundle[2]["filename"]).exists()
    monkeypatch.setattr(migration, "atomic_json", original)
    assert run(bundle)["state"] == "SOURCE_REMOVED"


def test_interrupt_after_deletion_resumes_from_verified_receipt(bundle, monkeypatch):
    original = migration.atomic_json

    def fail_after_delete(path, value):
        if value.get("state") == "SOURCE_REMOVED":
            raise InterruptedError("after deletion")
        return original(path, value)

    monkeypatch.setattr(migration, "atomic_json", fail_after_delete)
    with pytest.raises(InterruptedError):
        run(bundle)
    assert not (bundle[0] / bundle[2]["filename"]).exists()
    receipt = next((bundle[1] / "receipts").glob("*.json"))
    assert json.loads(receipt.read_text())["state"] == "VERIFIED"
    monkeypatch.setattr(migration, "atomic_json", original)
    assert run(bundle)["state"] == "SOURCE_REMOVED"


def test_corruption_after_source_retirement_fails_closed(bundle):
    run(bundle)
    path = migration.npy_path(bundle[1], bundle[3][0]["sample_id"])
    value = np.load(path, allow_pickle=False)
    value[0, 0, 0, 0] += 1
    np.save(path, value)
    with pytest.raises(ValueError, match="checksum"):
        run(bundle)


def test_missing_source_without_receipt_fails(bundle):
    (bundle[0] / bundle[2]["filename"]).unlink()
    with pytest.raises(FileNotFoundError, match="receipt"):
        run(bundle)


def test_receipt_source_identity_cannot_change(bundle):
    run(bundle)
    with pytest.raises(ValueError, match="another source"):
        migration.convert_shard(*bundle[:4], "different-index", shape=bundle[5])


def test_duplicate_records_fail(bundle):
    bundle[3][1] = bundle[3][0]
    with pytest.raises(ValueError, match="uniqueness"):
        run(bundle)


def test_extra_tensor_keys_fail(bundle):
    path = bundle[0] / bundle[2]["filename"]
    save_file({"unexpected": np.zeros(bundle[5], dtype=np.float16)}, path)
    bundle[2].update(bytes=path.stat().st_size, sha256=migration.file_sha(path))
    with pytest.raises(ValueError, match="keys"):
        run(bundle)


@pytest.mark.parametrize("sid", ["../escape", "/root/escape", "val2017/12", "train2017/000000000001/../x"])
def test_unsafe_sample_id_rejected(tmp_path, sid):
    with pytest.raises(ValueError):
        migration.npy_path(tmp_path, sid)


def test_symlink_output_rejected(bundle, tmp_path):
    (bundle[1] / "train2017").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        run(bundle)
    assert (bundle[0] / bundle[2]["filename"]).exists()


def test_nfs_output_rejected():
    with pytest.raises(ValueError, match="restricted"):
        migration.local_root(Path("/data/yingxi/anything"))


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "inf"])
def test_invalid_array_rejected(bundle, kind):
    value = np.zeros((3, *bundle[5]), dtype=np.float16)
    if kind == "dtype":
        value = value.astype(np.float32)
    elif kind == "shape":
        value = value[:, :, :, :2]
    else:
        value.flat[0] = np.nan if kind == "nan" else np.inf
    with pytest.raises(ValueError):
        migration.validate_array(value, bundle[3][0], bundle[5])


def test_coordinator_cancellation_before_retirement_preserves_source(bundle):
    def stop():
        raise InterruptedError("cancelled before source deletion")

    with pytest.raises(InterruptedError):
        migration.convert_shard(*bundle[:5], shape=bundle[5], before_retire=stop)
    assert (bundle[0] / bundle[2]["filename"]).exists()
    assert json.loads(next((bundle[1] / "receipts").glob("*.json")).read_text())["state"] == "VERIFIED"
    assert run(bundle)["state"] == "SOURCE_REMOVED"


def make_parallel_jobs(bundle, count=8):
    source, output, _, _, identity, shape = bundle
    jobs = []
    for index in range(count):
        name = f"train2017-r00-{index:05d}.safetensors"
        tensors, records = {}, []
        for sample in range(2):
            sid = f"train2017/{index * 10 + sample + 1:012d}"
            record = {"sample_id": sid, "split": "train2017", "shard": name, "tensors": {}}
            for layer, feature in enumerate(migration.NAMES):
                key = f"{index}-{sample}.{feature}"
                value = np.full(shape, index + sample + layer / 4, dtype=np.float16)
                tensors[key] = value
                record["tensors"][feature] = {
                    "key": key,
                    "shape": list(shape),
                    "dtype": "float16",
                    "nbytes": value.nbytes,
                    "sha256": migration.sha(value.tobytes()),
                }
            records.append(record)
        path = source / name
        save_file(tensors, path)
        shard = {
            "filename": name,
            "sample_count": 2,
            "split": "train2017",
            "bytes": path.stat().st_size,
            "sha256": migration.file_sha(path),
        }
        jobs.append((source, output, shard, records, identity, shape))
    return jobs


def test_eight_process_conversion_and_retired_revalidation(bundle):
    jobs = make_parallel_jobs(bundle)
    results = []
    migration.parallel_shards(jobs, bundle[1], 8, 0, results.append)
    assert len(results) == 8
    assert len({r["receipt"]["source_shard"]["filename"] for r in results}) == 8
    assert all(r["receipt"]["state"] == "SOURCE_REMOVED" for r in results)
    assert not list(bundle[0].glob("*.safetensors"))
    assert len(list(bundle[1].rglob("*.npy"))) == 16
    again = []
    migration.parallel_shards(jobs, bundle[1], 8, 0, again.append)
    assert len(again) == 8 and all(not r["source_existed"] for r in again)


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_invalid_parallel_workers(bundle, workers):
    with pytest.raises(ValueError, match="workers"):
        migration.parallel_shards(make_parallel_jobs(bundle, 1), bundle[1], workers, 0, lambda _: None)


def test_duplicate_parallel_job_is_rejected_before_deletion(bundle):
    jobs = make_parallel_jobs(bundle, 1)
    with pytest.raises(ValueError, match="duplicate"):
        migration.parallel_shards(jobs * 2, bundle[1], 8, 0, lambda _: None)
    assert (bundle[0] / jobs[0][2]["filename"]).is_file()


def test_parallel_space_gate_precedes_deletion(bundle):
    jobs = make_parallel_jobs(bundle, 1)
    with pytest.raises(RuntimeError, match="space"):
        migration.parallel_shards(jobs, bundle[1], 8, 1024**6, lambda _: None)
    assert (bundle[0] / jobs[0][2]["filename"]).is_file()


def test_parallel_corruption_preserves_bad_source_and_shuts_down(bundle):
    jobs = make_parallel_jobs(bundle, 8)
    bad = migration.npy_path(bundle[1], jobs[0][3][0]["sample_id"])
    bad.parent.mkdir()
    bad.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        migration.parallel_shards(jobs, bundle[1], 8, 0, lambda _: None)
    assert (bundle[0] / jobs[0][2]["filename"]).is_file()
    assert bad.read_bytes() == b"corrupt"
