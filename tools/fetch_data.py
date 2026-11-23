"""Download and verify the published NAV2 data without modifying the TAR files."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse

REPO = "WaltzZ105/multiroute-state-shards"
REVISION = "218857bc3632d6895e2b16560af4cd03d74236e4"
CHECKSUM_INDEX_SHA256 = "81b5ab816f6be36415c383ed1a0fc1049e67d75245a034660a23aed9d3ed01fd"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, sha256, size=None):
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError(f"Missing file or incorrect size: {path}")
    if file_sha256(path) != sha256:
        raise ValueError(f"Hugging Face checksum mismatch: {path}")


def download_from_resolver(endpoint, root, name, sha256, size=None):
    path = root / name
    if path.is_file():
        verify(path, sha256, size)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    url = f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/{quote(name)}"
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
        "--continue-at", "-", "--output", str(partial), url,
    ], check=True)
    verify(partial, sha256, size)
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--local-only", action="store_true", help="Verify files staged in an existing snapshot.")
    source.add_argument("--resolve-endpoint", help="Fetch public files over HTTPS with curl instead of the Hub API.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--list-shards", action="store_true")
    modes.add_argument("--manifests-only", action="store_true")
    modes.add_argument("--archive", action="append", help="A release-relative TAR path (repeatable).")
    modes.add_argument("--all", action="store_true")
    args = parser.parse_args()

    index = args.root / "CHECKSUMS.json"
    endpoint = None
    if args.resolve_endpoint:
        endpoint = args.resolve_endpoint.rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            parser.error("--resolve-endpoint must be HTTPS and may not contain credentials")
        download_from_resolver(endpoint, args.root, index.name, CHECKSUM_INDEX_SHA256)
    elif not args.local_only:
        from huggingface_hub import hf_hub_download

        options = dict(repo_id=REPO, repo_type="dataset", revision=REVISION, local_dir=args.root, token=False)
        hf_hub_download(filename=index.name, **options)
    verify(index, CHECKSUM_INDEX_SHA256)
    rows = {row["path"]: row for row in json.loads(index.read_text())["files"]}
    archives = sorted(name for name in rows if name.endswith(".tar"))
    if args.list_shards:
        print("\n".join(archives))
        return
    selected = [
        name for name in rows
        if name == "README.md" or name.endswith(("/all_train.jsonl", "/all_dev.jsonl"))
    ]
    if args.archive:
        if any(name not in rows or not name.endswith(".tar") for name in args.archive):
            parser.error("unknown TAR path, run --list-shards first")
        selected.extend(args.archive)
    if args.all:
        selected = sorted(rows)

    patterns = ["README.md", "r2r/**", "rxr/**"] if args.all else selected
    if endpoint:
        for name in selected:
            download_from_resolver(endpoint, args.root, name, rows[name]["sha256"], rows[name]["size"])
    elif not args.local_only:
        from huggingface_hub import snapshot_download

        snapshot_download(allow_patterns=patterns, **options)
    if not endpoint:
        for name in selected:
            verify(args.root / name, rows[name]["sha256"], rows[name]["size"])
    print(f"{REPO}@{REVISION}: verified {len(selected)} files")


if __name__ == "__main__":
    main()
