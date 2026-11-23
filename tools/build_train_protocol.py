"""Index completed observations for the four-GPU ms-swift NAV2 trainer."""

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_observations import episode_groups


def read_samples(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def finished_samples(observations, data, split):
    sources = [("r2r", "train"), ("rxr", "train")]
    if split == "train-and-dev":
        sources.extend([("r2r", "dev"), ("rxr", "dev")])
    for dataset, source_split in sources:
        manifest = data / dataset / "manifests" / f"all_{source_split}.jsonl"
        for archive, rows in episode_groups(manifest):
            key = hashlib.sha256(archive.encode()).hexdigest()
            directory = observations / dataset / key
            if not (directory / "receipt.json").is_file():
                raise FileNotFoundError(f"Unfinished observation archive: {archive}")
            samples = read_samples(directory / "observations.jsonl")
            for raw, sample in itertools.zip_longest(rows, samples):
                if raw is None or sample is None:
                    raise ValueError(f"Observation count mismatch in {archive}")
                expected = data / raw.get("labels_archive", archive)
                target = sample["nav2"]
                if (target["node_id"] != raw["node_id"] or target["labels_ref"] != raw["labels_ref"]
                        or Path(target["archive"]) != expected or target["observation_steps"][-1] != raw["step"]):
                    raise ValueError(f"NAV2 supervision mismatch in {archive}")
                yield sample


def pilot_samples(observations):
    for manifest in sorted(observations.glob("*/observations.jsonl")):
        yield from read_samples(manifest)


def validate_sample(sample, observations, checked_archives):
    images = sample["images"]
    navigation = sample["nav2"]
    positions = navigation["observation_steps"]
    if not 2 <= len(images) <= 9 or len(images) != len(positions):
        raise ValueError(f"Invalid SimpleVLN observation count for {navigation['node_id']}")
    if positions[0] != 0 or (positions[-1] == 0 and (positions != [0, 0] or images[0] != images[-1])):
        raise ValueError(f"Invalid starting observation for {navigation['node_id']}")
    if positions[-1] and any(a >= b for a, b in zip(positions, positions[1:])):
        raise ValueError(f"Unordered observations for {navigation['node_id']}")
    for name in images:
        if not (observations / name).is_file():
            raise FileNotFoundError(observations / name)
    archive = navigation["archive"]
    if archive not in checked_archives:
        checked_archives[archive] = Path(archive).stat().st_size
    ref = navigation["labels_ref"]
    if ref["offset"] < 0 or ref["size"] <= 0 or ref["offset"] + ref["size"] > checked_archives[archive]:
        raise ValueError(f"Invalid label TAR location for {navigation['node_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["train", "train-and-dev"])
    parser.add_argument("--smoke-updates", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    observation_root = args.observation_root.resolve()
    model = args.model.resolve()
    if not (model / "config.json").is_file():
        parser.error("--model must point to a local Qwen3.5-4B snapshot")
    if args.smoke_updates is not None:
        if args.smoke_updates < 2 or args.split or args.data_root:
            parser.error("the smoke test requires at least two updates and no full-data arguments")
        samples = pilot_samples(observation_root)
    else:
        if not (args.data_root and args.split):
            parser.error("full training requires --data-root and --split")
        samples = finished_samples(observation_root, args.data_root.resolve(), args.split)

    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "observations.jsonl"
    offsets = []
    checked_archives = {}
    frame_counts = Counter()
    digest = hashlib.sha256()
    with manifest.open("wb") as stream:
        for sample in samples:
            validate_sample(sample, observation_root, checked_archives)
            offsets.append(stream.tell())
            raw = (json.dumps(sample, ensure_ascii=False) + "\n").encode()
            stream.write(raw)
            digest.update(raw)
            frame_counts[len(sample["images"])] += 1
    if not offsets:
        raise ValueError("No rendered observation samples found")
    batch = 32
    seed = 20261027
    shuffled = np.random.default_rng(seed).permutation(len(offsets))
    if args.smoke_updates:
        shuffled = np.resize(shuffled, args.smoke_updates * batch)
    offset_file = output / "offsets.npy"
    sequence_file = output / "sequence.npy"
    np.save(offset_file, np.asarray(offsets, dtype=np.uint64))
    np.save(sequence_file, shuffled)
    protocol = dict(
        observation_root=str(observation_root), manifest=str(manifest), offsets=str(offset_file),
        sequence=str(sequence_file), processor=str(model), model_path=str(model), global_batch=batch,
        seed=seed, unique_rows=len(offsets), sample_exposures=len(shuffled),
        updates=(len(shuffled) + batch - 1) // batch, frame_counts=dict(sorted(frame_counts.items())),
        observation_stride=1, manifest_sha256=digest.hexdigest(),
        smoke=bool(args.smoke_updates),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol), flush=True)


if __name__ == "__main__":
    main()
