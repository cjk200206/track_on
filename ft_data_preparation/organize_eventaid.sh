#!/usr/bin/env bash
# Unpack Dropbox EventAid bundles into independent scene directories.
set -euo pipefail
command -v python3 >/dev/null

python3 - "$@" <<'PY'
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def ignored(name):
    return any(part in {"__MACOSX", ".DS_Store", "Thumbs.db"} or part.startswith("._")
               for part in PurePosixPath(name).parts)


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def scene_path(bundle, member):
    parts = list(safe_path(member.filename).with_suffix("").parts)
    if parts and parts[0] == bundle.stem:
        parts.pop(0)
    return Path(*(part.replace("controlled experiment", "controlled_experiment") for part in parts))


def organize(bundle, dry_run):
    with zipfile.ZipFile(bundle) as outer:
        scenes = [item for item in outer.infolist()
                  if not item.is_dir() and item.filename.lower().endswith(".zip")
                  and not ignored(item.filename)]
        if not scenes:
            raise ValueError(f"No scene ZIPs found in {bundle}")
        print(f"{bundle.name}: {len(scenes)} scene archives", flush=True)
        for member in scenes:
            destination = bundle.parent / bundle.stem / scene_path(bundle, member)
            marker = destination / ".eventaid_complete.json"
            identity = {"version": 1, "bundle": bundle.name, "member": member.filename,
                        "crc": member.CRC, "size": member.file_size}
            if dry_run:
                print(f"  {member.filename} -> {destination}", flush=True)
                continue
            if marker.is_file():
                try:
                    if json.loads(marker.read_text()) == identity:
                        print(f"  Skip completed: {destination}", flush=True)
                        continue
                except json.JSONDecodeError:
                    pass

            print(f"  Extracting: {destination}", flush=True)
            # Keep at most one temporary scene ZIP on the dataset filesystem.
            with tempfile.TemporaryDirectory(prefix=".eventaid-", dir=bundle.parent) as temporary:
                scene_zip = Path(temporary) / "scene.zip"
                with outer.open(member) as source, scene_zip.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                with zipfile.ZipFile(scene_zip) as inner:
                    files = [item for item in inner.infolist()
                             if not item.is_dir() and not ignored(item.filename)]
                    # D-wall has an extra wrapper around its actual scene data.
                    strip_wrapper = bool(files) and all(
                        item.filename.startswith("yyds/") for item in files
                    )
                    for item in files:
                        name = item.filename[5:] if strip_wrapper else item.filename
                        output = destination / safe_path(name)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with inner.open(item) as source, output.open("wb") as target:
                            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                    print(f"  Done: {len(files)} files"
                          + ("; removed yyds wrapper" if strip_wrapper else ""), flush=True)
            destination.mkdir(parents=True, exist_ok=True)
            temporary_marker = marker.with_suffix(".tmp")
            temporary_marker.write_text(json.dumps(identity, indent=2) + "\n")
            temporary_marker.replace(marker)


parser = argparse.ArgumentParser(
    prog="organize_eventaid.sh",
    description="Unpack EventAid scene ZIPs, retain source bundles and skip completed scenes.",
)
parser.add_argument("data_root", nargs="?", default="/remote-home/share/cjk/datasets/EventAid")
parser.add_argument("--subsets", nargs="+", choices=list("BDFRS"), default=None,
                    help="Only organize the selected subsets; default: all downloaded bundles")
parser.add_argument("--dry-run", action="store_true", help="List destinations without extracting")
args = parser.parse_args()
root = Path(args.data_root).expanduser().resolve()
if not root.is_dir():
    parser.error(f"Dataset directory does not exist: {root}")
bundles = ([root / f"EventAid-{subset}.zip" for subset in dict.fromkeys(args.subsets)]
           if args.subsets else sorted(root.glob("EventAid-[BDFRS].zip")))
if not bundles:
    parser.error(f"No EventAid bundle ZIPs found in {root}")
for bundle in bundles:
    if not bundle.is_file():
        parser.error(f"Missing bundle: {bundle}")
for bundle in bundles:
    organize(bundle, args.dry_run)
print("Preview complete." if args.dry_run else "Finished. Original bundle ZIPs retained.", flush=True)
PY
