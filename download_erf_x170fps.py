"""Download ERF-X170FPS archives using validated, resumable Google Drive ranges."""

import argparse
import fcntl
import logging
import time
from pathlib import Path

import requests


FILES = {
    "train": ("1Bsf9qreziPcVEuf0_v3kjdPUh27zsFXK", 234270237378),
    "test": ("1Dk7jVQD29HqRVV11e8vxg5bDOh6KxrzL", 81372206342),
}
CHUNK_SIZE = 4 * 1024 * 1024
RETRY_DELAY = 60
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def read_chunk(session, file_id, start, end, total):
    with session.get(
        DOWNLOAD_URL,
        params={"id": file_id, "export": "download", "confirm": "t"},
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=(30, 120),
    ) as response:
        expected = f"bytes {start}-{end}/{total}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError(
                f"Unexpected response: HTTP {response.status_code}, "
                f"type={response.headers.get('Content-Type')}, "
                f"range={response.headers.get('Content-Range')}"
            )
        data = response.content
        if len(data) != end - start + 1:
            raise ValueError("Incomplete chunk")
        if start == 0 and not data.startswith(b"PK\x03\x04"):
            raise ValueError("Response is not ZIP data")
        return data


def download(session, path, file_id, total):
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:
            valid = existing.read(4) == b"PK\x03\x04"
        if not valid:
            backup = path.with_name(f"{path.name}.invalid-{time.time_ns()}")
            path.rename(backup)
            logging.warning("Preserved invalid file as %s", backup)

    offset = path.stat().st_size if path.exists() else 0
    if offset > total:
        raise ValueError(f"{path} exceeds expected size; check the file before resuming")
    logging.info("%s: starting at %s / %s bytes", path.name, offset, total)

    with path.open("ab") as output:
        while offset < total:
            end = min(offset + CHUNK_SIZE, total) - 1
            try:
                data = read_chunk(session, file_id, offset, end, total)
            except (requests.RequestException, ValueError) as error:
                logging.warning("%s: %s; retrying in %s seconds", path.name, error, RETRY_DELAY)
                time.sleep(RETRY_DELAY)
                continue

            output.write(data)
            output.flush()
            offset += len(data)
            logging.info("%s: %.3f / %.3f GB", path.name, offset / 1e9, total / 1e9)

    logging.info("%s: download complete", path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/remote-home/share/cjk/dataset/ERF-X170FPS"),
    )
    parser.add_argument("--proxy", default="http://127.0.0.1:17890")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A download is already running in this output directory")

        with requests.Session() as session:
            session.trust_env = False
            session.proxies = {"http": args.proxy, "https": args.proxy}
            session.headers["Accept-Encoding"] = "identity"
            splits = FILES if args.split == "all" else (args.split,)
            for split in splits:
                file_id, total = FILES[split]
                download(session, args.output_dir / f"{split}.zip", file_id, total)

    logging.info("All requested downloads complete")


if __name__ == "__main__":
    main()
