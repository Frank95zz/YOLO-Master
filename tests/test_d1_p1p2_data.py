"""Non-destructive copy, mount and partial-file failure tests."""

import json

import pytest

from scripts.d1 import p1p2_data as data


@pytest.fixture
def fake_nvme(tmp_path, monkeypatch):
    root = tmp_path / "nvme"
    root.mkdir()
    data._nvme_device.cache_clear()
    monkeypatch.setattr(
        data.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps(
            {
                "filesystems": [{"source": "/dev/nvme0n1", "fstype": "ext4", "target": str(root)}],
            }
        ),
    )
    yield root
    data._nvme_device.cache_clear()


def test_mount_and_escape_checks(fake_nvme):
    assert data.nvme_path(fake_nvme / "new/data", fake_nvme) == fake_nvme / "new/data"
    with pytest.raises(ValueError):
        data.nvme_path(fake_nvme.parent / "elsewhere", fake_nvme)


def test_nfs_rejected(tmp_path, monkeypatch):
    data._nvme_device.cache_clear()
    monkeypatch.setattr(
        data.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps(
            {
                "filesystems": [{"source": "server:/data", "fstype": "nfs4", "target": str(tmp_path)}],
            }
        ),
    )
    with pytest.raises(ValueError):
        data.nvme_path(tmp_path, tmp_path)
    data._nvme_device.cache_clear()


def test_copy_resume_and_idempotency(fake_nvme, tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"same RGB bytes")
    target = fake_nvme / "image.jpg"
    partial = fake_nvme / "image.jpg.d1-copy.part"
    partial.write_bytes(b"same")
    result = data.copy_verified(source, target, fake_nvme)
    assert target.read_bytes() == source.read_bytes()
    assert not partial.exists()
    assert result == data.copy_verified(source, target, fake_nvme)
    assert source.read_bytes() == b"same RGB bytes"


def test_existing_different_file_not_overwritten(fake_nvme, tmp_path):
    source, target = tmp_path / "source", fake_nvme / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    with pytest.raises(ValueError):
        data.copy_verified(source, target, fake_nvme)
    assert target.read_bytes() == b"old"


def test_bad_partial_preserved(fake_nvme, tmp_path):
    source, target = tmp_path / "source", fake_nvme / "target"
    source.write_bytes(b"abcdef")
    partial = fake_nvme / "target.d1-copy.part"
    partial.write_bytes(b"BAD")
    with pytest.raises(ValueError):
        data.copy_verified(source, target, fake_nvme)
    assert partial.exists() and not target.exists()


def test_empty_label(fake_nvme):
    target = fake_nvme / "empty.txt"
    result = data.copy_verified(None, target, fake_nvme)
    assert result["empty_label_materialized"] and result["bytes"] == 0
    assert target.read_bytes() == b""
