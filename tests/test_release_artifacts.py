"""Artifact restoration must reject tampering and preserve existing runs."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import pytest

spec = importlib.util.spec_from_file_location("restore_artifacts", Path(__file__).parents[1] / "scripts/restore_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(tmp_path, name="data/sample.jsonl", payload=b'{"example": 1}\n'):
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    manifest = {"sha256": module.digest(archive), "files": [
        {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    return archive, manifest


def test_roundtrip_and_idempotence(tmp_path):
    archive, manifest = make_archive(tmp_path)
    root = tmp_path / "output"
    assert module.restore(archive, manifest, root, verify_only=True) == 1
    assert not root.exists()
    assert module.restore(archive, manifest, root) == 1
    assert module.restore(archive, manifest, root) == 1
    assert (root / "data/sample.jsonl").read_bytes() == b'{"example": 1}\n'


def test_tampered_archive_rejected(tmp_path):
    archive, manifest = make_archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Archive SHA-256"):
        module.restore(archive, manifest, tmp_path / "output")


def test_member_hash_rejected(tmp_path):
    archive, manifest = make_archive(tmp_path)
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Member SHA-256"):
        module.restore(archive, manifest, tmp_path / "output")


def test_existing_run_preserved(tmp_path):
    archive, manifest = make_archive(tmp_path)
    root = tmp_path / "output"
    (root / "data").mkdir(parents=True)
    target = root / "data/sample.jsonl"
    target.write_text("my new run")
    with pytest.raises(FileExistsError):
        module.restore(archive, manifest, root)
    assert target.read_text() == "my new run"
    module.restore(archive, manifest, root, overwrite=True)
    assert target.read_bytes() == b'{"example": 1}\n'


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "data/../../escape"])
def test_path_traversal_rejected(tmp_path, name):
    archive, manifest = make_archive(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        module.restore(archive, manifest, tmp_path / "output")


def test_symlink_parent_rejected(tmp_path):
    archive, manifest = make_archive(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    (root / "data").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.restore(archive, manifest, root)
