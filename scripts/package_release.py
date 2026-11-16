"""Maintainer utility: package the original experiment via an explicit allowlist."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("release_assets"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/release_manifest.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    groups = {"data": [], "checkpoints": [], "evidence": []}
    groups["data"] = [p for p in (source / "data").iterdir() if p.suffix in (".json", ".jsonl")]
    for method in ("sft", "grpo_A", "grpo_B", "grpo_C"):
        base = source / "checkpoints" / method
        groups["checkpoints"].extend(p for p in (base / "final").iterdir()
                                      if p.name != "training_args.bin" and p.is_file())
        groups["evidence"].extend(p for p in base.iterdir() if p.suffix in (".json", ".jsonl"))
    groups["evidence"].append(source / "checkpoints/grpo_smoke/resume_audit.json")
    results = source / "results"
    groups["evidence"].extend(p for p in results.iterdir() if p.suffix in (".csv", ".json"))
    groups["evidence"].append(results / "pytest.log")
    for method in ("baselines", "qwen_zero", "qwen_sft", "qwen_grpo_A", "qwen_grpo_B",
                   "qwen_grpo_C", "sft_fixed_validation", "sft_validation", "figures"):
        groups["evidence"].extend(p for p in (results / method).iterdir()
                                  if p.suffix in (".csv", ".json", ".jsonl", ".png", ".pdf"))
    manifest = {"schema_version": 1, "release": "v0.1.0", "artifacts": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for group, paths in groups.items():
        filename = f"kkt-{group}-v0.1.0.tar.gz"
        target = args.output / filename
        if target.exists():
            raise FileExistsError(f"Refusing to replace {target}")
        members = []
        with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                     compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(set(paths)):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"Expected regular file: {path}")
                    relative = path.relative_to(source).as_posix()
                    members.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    info.pax_headers = {}
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        manifest["artifacts"][group] = {"filename": filename, "bytes": target.stat().st_size,
                                         "sha256": digest(target), "files": members}
        print(f"{filename}: {len(members)} files, {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text("".join(
        f"{value['sha256']}  {value['filename']}\n" for value in manifest["artifacts"].values()))


if __name__ == "__main__":
    main()
