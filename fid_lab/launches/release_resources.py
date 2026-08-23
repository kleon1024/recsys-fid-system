"""Content-bound resources shared by simulated surface releases."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def resource(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": file_sha256(root / relative)}


def source_resources(
    root: Path, package_relative: str, shared: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    package = root / package_relative
    relatives = {
        path.relative_to(root).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    relatives.update(shared)
    return [resource(root, relative) for relative in sorted(relatives)]


def bundle_identifier(bundle: dict) -> str:
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"


def verified_artifact(
    root: Path, artifact_dir: str, evidence: dict, hash_field: str = "sha256",
) -> dict[str, str]:
    relative = f"{artifact_dir}/{evidence['artifact_file']}"
    artifact = resource(root, relative)
    if artifact["sha256"] != evidence[hash_field]:
        raise ValueError(f"model artifact hash mismatch: {relative}")
    return artifact
