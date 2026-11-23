"""Expose NAV2 observation rows to the official SimpleVLN cache exporter."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    root = Path(protocol["observation_root"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(protocol["manifest"], "rb") as source, args.output.open("x", encoding="utf-8") as output:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            sample = json.loads(line)
            row = {
                "messages": [
                    {"role": "system", "content": sample["system"]},
                    {"role": "user", "content": sample["messages"][0]["value"]},
                ],
                "images": [str(root / name) for name in sample["images"]],
                "channel": f"nav2:{protocol['manifest_sha256']}:{offset}",
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
