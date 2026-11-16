"""Verify release archive/member hashes and restore their working directory layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_target(root, name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or path.parts[0] not in ("data", "checkpoints", "results")):
        raise ValueError(f"Unsafe archive path: {name}")
    target = root.joinpath(*path.parts)
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Refusing to follow symlink: {candidate}")
    return target


def restore(archive_path, spec, root, overwrite=False, verify_only=False):
    root = Path(root).resolve()
    if digest(archive_path) != spec["sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {archive_path.name}")
    expected = {item["path"]: item for item in spec["files"]}
    if len(expected) != len(spec["files"]):
        raise ValueError("Duplicate path in manifest")
    pending = []
    with tarfile.open(archive_path, "r:gz") as archive:
        seen = set()
        # Preflight all members and destinations before writing any file.
        for member in archive.getmembers():
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            seen.add(member.name)
            target = checked_target(root, member.name)
            item = expected[member.name]
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if member.size != item["bytes"] or actual != item["sha256"]:
                raise ValueError(f"Member SHA-256/size mismatch: {member.name}")
            if target.exists():
                if not target.is_file():
                    raise FileExistsError(f"Not a regular file: {target}")
                if digest(target) == item["sha256"]:
                    continue
                if not overwrite:
                    raise FileExistsError(f"Different file exists: {target}; use a clean directory or --overwrite")
            pending.append((member, target))
        if seen != set(expected):
            raise ValueError("Archive is missing manifest members")
        if not verify_only:
            for member, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        with archive.extractfile(member) as stream:
                            shutil.copyfileobj(stream, output)
                    temporary.chmod(0o644)
                    os.replace(temporary, target)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=["data", "checkpoints", "evidence"])
    parser.add_argument("--directory", type=Path, default=Path("downloads"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/release_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for group in args.groups:
        spec = manifest["artifacts"][group]
        count = restore(args.directory / spec["filename"], spec, args.root,
                        args.overwrite, args.verify_only)
        print(f"{group}: {count} files verified" + ("" if args.verify_only else " and restored"))


if __name__ == "__main__":
    main()
