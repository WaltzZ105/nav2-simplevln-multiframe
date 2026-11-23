"""Replay the published action prefixes and export SimpleVLN observation inputs."""

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

SIMPLEVLN_REVISION = "0216a1aa2d78e5f093595265d28bd90b29e0c87e"


def episode_groups(manifest):
    with manifest.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        seen = set()
        for archive, group in itertools.groupby(rows, key=lambda row: row["archive"]):
            if archive in seen:
                raise ValueError(f"Archive rows are not contiguous in {manifest}: {archive}")
            seen.add(archive)
            yield archive, list(group)


def index_episode(rows):
    by_prefix = {}
    identity = {(row["scene"], str(row["episode"])) for row in rows}
    if len(identity) != 1:
        raise ValueError("Mixed episode identities in a TAR manifest")
    for row in rows:
        prefix = tuple(row["prefix_actions"])
        if row["step"] != len(prefix) or any(action not in (1, 2, 3) for action in prefix):
            raise ValueError(f"Invalid executed action prefix for {row['node_id']}")
        pose = (row["position"], row["rotation_xyzw"])
        if len(pose[0]) != 3 or len(pose[1]) != 4 or not all(map(math.isfinite, pose[0] + pose[1])):
            raise ValueError(f"Invalid archived pose for {row['node_id']}")
        if prefix in by_prefix:
            assert_pose(by_prefix[prefix], *pose)
        else:
            by_prefix[prefix] = row
    return by_prefix


def episode_starts(paths):
    starts = {}
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as source:
            episodes = json.load(source)["episodes"]
        for episode in episodes:
            key = Path(episode["scene_id"]).stem, str(episode["episode_id"])
            value = episode["start_position"], episode["start_rotation"]
            if key in starts and starts[key] != value:
                raise ValueError(f"Conflicting source episode poses: {key}")
            starts[key] = value
    return starts


def assert_pose(row, position, rotation):
    expected_position = row["position"]
    expected_rotation = row["rotation_xyzw"]
    position_error = max(abs(a - b) for a, b in zip(expected_position, position))
    rotation_error = min(
        max(abs(a - sign * b) for a, b in zip(expected_rotation, rotation))
        for sign in (1, -1)
    )
    if not (position_error <= 1e-3 and rotation_error <= 1e-3):
        raise ValueError(
            f"Replay pose mismatch for {row['node_id']}: "
            f"position={position_error:.6g} m, quaternion={rotation_error:.6g}"
        )


def history_positions(step, upstream):
    return upstream.normalize_pair(upstream.sample_history_plus_current(list(range(step + 1))))


def render_episode(renderer, by_prefix, frame_dir):
    root = by_prefix.get(())
    if root is None:
        raise ValueError("No step-zero pose in the published rows")
    frame_dir.mkdir(parents=True)
    renderer.start(root)
    children = {}
    for prefix in sorted(by_prefix):
        for step, action in enumerate(prefix):
            children.setdefault(prefix[:step], set()).add(action)
    saved = {}

    def visit(prefix, rgb):
        if prefix in by_prefix:
            assert_pose(by_prefix[prefix], *renderer.pose())
        frame_id = hashlib.sha256(bytes(prefix)).hexdigest()
        path = frame_dir / f"{frame_id}.jpg"
        renderer.save(rgb, path)
        saved[prefix] = path
        next_actions = sorted(children.get(prefix, ()))
        if len(next_actions) > 1:
            position, rotation = renderer.pose()
        for index, action in enumerate(next_actions):
            if index:
                renderer.restore(position, rotation)
            visit((*prefix, action), renderer.step(action))

    visit((), renderer.current())
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "train-and-dev"), required=True)
    parser.add_argument("--dataset", choices=("r2r", "rxr"), required=True)
    parser.add_argument("--archive", action="append", help="Process only specific release TARs.")
    parser.add_argument("--audit", action="store_true", help="Inspect prefix/pose coverage on CPU.")
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--episodes", type=Path, action="append", default=[],
                        help="Original Habitat episode annotations (.json or .json.gz) for missing root poses.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed, verified archive outputs.")
    parser.add_argument("--sim-gpu", type=int, default=0)
    args = parser.parse_args()
    root = args.data_root.resolve()
    selected = set(args.archive or ())
    splits = ("train", "dev") if args.split == "train-and-dev" else ("train",)
    upstream_dir = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != SIMPLEVLN_REVISION:
        parser.error(f"expected SimpleVLN at {SIMPLEVLN_REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(upstream_dir), "status", "--porcelain", "--untracked-files=no", "--", "DataProcess"],
        text=True,
    ).strip():
        parser.error("SimpleVLN DataProcess contains local edits")
    sys.path.insert(0, str(upstream_dir / "DataProcess/tool"))
    import convert_vln

    if args.audit:
        renderer = None
        output = None
    else:
        if not (args.scenes_dir and args.output):
            parser.error("replay requires --scenes-dir and --output")
        from replay_habitat import HabitatRenderer

        renderer = HabitatRenderer(upstream_dir, args.scenes_dir.resolve(), args.sim_gpu)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    found = set()
    missing_roots = []
    starts = episode_starts(args.episodes)
    try:
        for split in splits:
            manifest = root / args.dataset / "manifests" / f"all_{split}.jsonl"
            for archive, rows in episode_groups(manifest):
                if selected and archive not in selected:
                    continue
                found.add(archive)
                counts.update(archives=1, states=len(rows))
                by_prefix = index_episode(rows)
                if () not in by_prefix:
                    first = rows[0]
                    episode_key = Path(first["scene"]).stem, str(first["episode"])
                    if episode_key in starts:
                        position, rotation = starts[episode_key]
                        by_prefix[()] = {
                            "scene": first["scene"], "node_id": f"{archive}:source-root",
                            "position": position, "rotation_xyzw": rotation,
                        }
                        counts["annotation_roots"] += 1
                    else:
                        missing_roots.append(archive)
                        if renderer:
                            raise ValueError(f"Missing step-zero pose for {archive}. Supply --episodes.")
                for prefix in by_prefix:
                    historical = set(history_positions(len(prefix), convert_vln)) - {len(prefix)}
                    counts["history_refs"] += len(historical)
                    counts["history_refs_without_released_rgb"] += sum(
                        prefix[:step] not in by_prefix
                        or "rgb_ref" not in by_prefix[prefix[:step]]
                        for step in historical
                    )
                if renderer:
                    receipt = prepare_archive(
                        renderer, root, output, archive, rows, by_prefix, convert_vln, args.resume
                    )
                    counts["rendered_frames"] += receipt["rendered_frames"]
                    counts["black_frames"] += receipt["black_frames"]
    finally:
        if renderer:
            renderer.close()
    if selected - found:
        raise ValueError(f"No manifest records for: {sorted(selected - found)}")
    report = dict(counts, dataset=args.dataset, split=args.split, missing_roots=missing_roots)
    print(json.dumps(report, ensure_ascii=False))


def prepare_archive(renderer, root, output, archive, rows, by_prefix, upstream, resume):
    import shutil
    import tempfile

    for relative in {archive, *(row.get("labels_archive", archive) for row in rows)}:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    key = hashlib.sha256(archive.encode()).hexdigest()
    episode_dir = output / key
    source_digest = hashlib.sha256()
    for row in [by_prefix[()], *rows]:
        source_digest.update(json.dumps(row, sort_keys=True).encode())
        source_digest.update(b"\n")
    receipt = {
        "archive": archive,
        "simplevln_revision": SIMPLEVLN_REVISION,
        "input_sha256": source_digest.hexdigest(),
        "source_root": str(root),
        "states": len(rows),
    }
    if episode_dir.exists():
        saved = episode_dir / "receipt.json"
        observations = episode_dir / "observations.jsonl"
        if resume and saved.is_file() and observations.is_file():
            with observations.open(encoding="utf-8") as stream:
                samples = [json.loads(line) for line in stream]
            completed = json.loads(saved.read_text())
            if (completed.get("input") == receipt and len(samples) == len(rows)
                    and all(Path(image).is_file() for sample in samples for image in sample["images"])):
                return completed
        raise FileExistsError(f"Output already exists without a matching verified receipt: {episode_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=output))
    try:
        images = render_episode(renderer, by_prefix, temporary / "rgb")
        with (temporary / "observations.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                prefix = tuple(row["prefix_actions"])
                positions = history_positions(len(prefix), upstream)
                image_paths = [
                    str(episode_dir / "rgb" / images[prefix[:step]].name) for step in positions
                ]
                label_archive = root / row.get("labels_archive", archive)
                target.write(json.dumps({
                    "system": upstream.SYSTEM_PROMPT,
                    "messages": [
                        {"from": "user", "value": upstream.build_vln_prompt(row["instruction"], len(positions))}
                    ],
                    "images": image_paths,
                    "nav2": {
                        "node_id": row["node_id"], "archive": str(label_archive),
                        "labels_ref": row["labels_ref"], "observation_steps": positions,
                    },
                }, ensure_ascii=False) + "\n")
        result = {"input": receipt, "rendered_frames": len(images), "black_frames": renderer.black_frames}
        (temporary / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        os.rename(temporary, episode_dir)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
