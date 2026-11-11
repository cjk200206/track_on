#!/usr/bin/env bash
# Download and organize original DSEC training sequences for left-camera FE use.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash $0 [data_root] [all|download|organize]"
    echo "Downloads left events, image/exposure timestamps and calibration."
    echo "Remapped images and detection extension sequences are not downloaded."
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${1:-$script_dir/../data/DSEC}"
mode="${2:-all}"
case "$mode" in
    all|download|organize) ;;
    *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac
command -v python3 >/dev/null
mkdir -p "$data_root/downloads"
data_root="$(cd -- "$data_root" && pwd)"

if [[ "$mode" != "organize" ]]; then
    command -v wget >/dev/null
    python3 - "$data_root" <<'PY'
import html
import re
import sys
from pathlib import Path
from urllib.request import urlopen

root = Path(sys.argv[1])
with urlopen("https://dsec.ifi.uzh.ch/dsec-datasets/download/", timeout=60) as response:
    page = response.read().decode()
suffixes = (
    "_events_left.zip", "_calibration.zip",
    "_image_timestamps.txt", "_image_exposure_timestamps_left.txt",
)
urls = sorted({
    html.unescape(url)
    for url in re.findall(r'href="([^"]+)"', page)
    if url.startswith("https://download.ifi.uzh.ch/rpg/DSEC/train/")
    and url.endswith(suffixes)
})
if not urls:
    raise RuntimeError("No original DSEC training download links found")
(root / "download_urls.txt").write_text("\n".join(urls) + "\n")
sequences = {url.split("/")[-2] for url in urls}
print(f"Download list: {len(sequences)} sequences, {len(urls)} files", flush=True)
PY

    wget -c --timeout=60 --tries=20 --waitretry=10 --retry-connrefused \
        -i "$data_root/download_urls.txt" -P "$data_root/downloads"
fi

if [[ "$mode" != "download" ]]; then
    python3 - "$data_root" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
rules = {
    "_events_left.zip": "events/left",
    "_calibration.zip": "calibration",
    "_image_timestamps.txt": "images/timestamps.txt",
    "_image_exposure_timestamps_left.txt": "images/exposure_timestamps_left.txt",
}
for url in (root / "download_urls.txt").read_text().splitlines():
    if not url:
        continue
    source = root / "downloads" / url.rsplit("/", 1)[-1]
    sequence = url.split("/")[-2]
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = next(suffix for suffix in rules if source.name.endswith(suffix))
    target = root / "train" / sequence / rules[suffix]
    print(f"Organizing {sequence}: {source.name}", flush=True)
    if suffix.endswith(".zip"):
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
print(f"Organized data: {root / 'train'}", flush=True)
print("Download archives retained. Extract the remap ZIP into the data root.")
PY
fi
