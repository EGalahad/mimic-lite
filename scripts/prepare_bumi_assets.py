#!/usr/bin/env python3
"""Validate and stage the Bumi MJCF asset in Active Adaptation's cache."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from uuid import uuid4


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"dir\0")
    return digest.hexdigest()


def _install_staged_tree(stage: Path, output: Path, *, force: bool) -> str:
    if output.exists():
        if _tree_fingerprint(stage) == _tree_fingerprint(output):
            return "unchanged"
        if not force:
            raise FileExistsError(
                f"Output exists with different content: {output}. Pass --force to replace it."
            )

        backup = output.parent / f".{output.name}.backup-{uuid4().hex}"
        output.rename(backup)
        try:
            stage.rename(output)
        except Exception:
            backup.rename(output)
            raise
        shutil.rmtree(backup)
        return "replaced"

    stage.rename(output)
    return "created"


def prepare_bumi_assets(
    source: Path, output: Path, *, force: bool = False
) -> dict[str, int | str]:
    import mujoco

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    xml_path = source / "bumi.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Expected Bumi MJCF at {xml_path}")

    xml_root = ET.parse(xml_path).getroot()
    compiler = xml_root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    if meshdir != "assets":
        raise ValueError(f"Expected <compiler meshdir='assets'>, got {meshdir!r}")

    mesh_file_set: set[str] = set()
    for mesh in xml_root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is not None:
            mesh_file_set.add(mesh_file)
    mesh_files = sorted(mesh_file_set)
    if not mesh_files:
        raise ValueError(f"No mesh file references found in {xml_path}")
    missing_meshes = [
        name for name in mesh_files if not (source / meshdir / name).is_file()
    ]
    if missing_meshes:
        raise FileNotFoundError(
            "Missing Bumi mesh files: "
            + ", ".join(str(name) for name in missing_meshes)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.stage-", dir=output.parent
    ) as temp_dir:
        stage = Path(temp_dir) / output.name
        stage.mkdir()
        shutil.copy2(xml_path, stage / "bumi.xml")
        shutil.copytree(source / meshdir, stage / meshdir)

        mj_spec_cls = getattr(mujoco, "MjSpec")
        joint_type = getattr(mujoco, "mjtJoint")
        spec = mj_spec_cls.from_file(str(stage / "bumi.xml"))
        model = spec.compile()
        hinge_count = int((model.jnt_type == joint_type.mjJNT_HINGE).sum())
        free_count = int((model.jnt_type == joint_type.mjJNT_FREE).sum())
        body_count = int(model.nbody - 1)
        if (hinge_count, free_count, body_count) != (21, 1, 22):
            raise ValueError(
                "Unexpected Bumi model topology: "
                f"hinges={hinge_count}, freejoints={free_count}, bodies={body_count}"
            )

        status = _install_staged_tree(stage, output, force=force)

    result: dict[str, int | str] = {
        "status": status,
        "source": str(source),
        "output": str(output),
        "meshes": len(mesh_files),
        "hinge_joints": hinge_count,
        "free_joints": free_count,
        "bodies": body_count,
    }
    print(
        "Prepared Bumi asset:",
        " ".join(f"{key}={value}" for key, value in result.items()),
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing bumi.xml and assets/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / ".cache"
        / "aa-robot-models"
        / "bumi",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace a different existing cache"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_bumi_assets(args.source, args.output, force=args.force)


if __name__ == "__main__":
    main()
