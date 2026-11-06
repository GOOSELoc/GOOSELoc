"""Check that a GOOSELoc download is complete.

    gooseloc-check --root data/GOOSELoc [--split test] [--modalities aerial lidar]

Verifies that every scene of the selected splits has all files of the selected
modalities and, with ``--deep``, that they can be decoded and have the expected
shapes. With ``--checksums`` the SHA-256 sums in ``<root>/CHECKSUMS.sha256`` are
verified as well (slow: reads every byte).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from gooseloc.constants import CAMERA_HW, CAMERAS, MODALITIES, TILE_SIZE_PX
from gooseloc.data.io import (
    aerial_path,
    camera_path,
    lidar_path,
    read_image,
    read_lidar,
    resolve_root,
)
from gooseloc.data.metadata import load_scenes, select


def expected_files(root: Path, scene_ids, modalities) -> list[Path]:
    files = []
    for sid in scene_ids:
        if "aerial" in modalities:
            files.append(aerial_path(root, sid))
        if "cameras" in modalities:
            files.extend(camera_path(root, sid, cam) for cam in CAMERAS)
        if "lidar" in modalities:
            files.append(lidar_path(root, sid))
    return files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="all")
    p.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    p.add_argument("--deep", action="store_true", help="decode every file and check shapes")
    p.add_argument("--checksums", action="store_true", help="verify CHECKSUMS.sha256")
    args = p.parse_args(argv)

    root = resolve_root(args.root)
    scenes = select(load_scenes(root), args.split)
    files = expected_files(root, scenes["scene_id"], args.modalities)
    missing = [f for f in files if not f.exists()]
    print(f"{len(scenes)} scenes, {len(files)} files expected, {len(missing)} missing")
    for f in missing[:20]:
        print(f"  missing: {f}")

    bad = []
    if args.deep:
        for f in files:
            if not f.exists():
                continue
            try:
                if f.suffix == ".bin":
                    pts = read_lidar(f)
                    ok = pts.ndim == 2 and pts.shape[0] > 0
                else:
                    size = read_image(f).size[::-1]
                    ok = size == ((TILE_SIZE_PX, TILE_SIZE_PX) if f.parent.name == "aerial" else CAMERA_HW)
            except Exception:
                ok = False
            if not ok:
                bad.append(f)
        print(f"{len(bad)} files failed to decode or have unexpected shapes")
        for f in bad[:20]:
            print(f"  bad: {f}")

    mismatched = []
    if args.checksums:
        sums_file = root / "CHECKSUMS.sha256"
        wanted = {str(f.relative_to(root)) for f in files}
        for line in sums_file.read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*")
            if rel in wanted and (root / rel).exists() and _sha256(root / rel) != digest:
                mismatched.append(rel)
        print(f"{len(mismatched)} checksum mismatches")

    return 1 if (missing or bad or mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
