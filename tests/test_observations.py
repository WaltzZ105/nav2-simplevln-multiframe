"""Exercise prefix selection and replay on branched synthetic trajectories."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(os.environ["SIMPLEVLN_ROOT"]) / "DataProcess/tool"))

import convert_vln
from prepare_observations import history_positions, index_episode, prepare_archive, render_episode


def row(prefix, position=None):
    return {
        "dataset": "r2r", "scene": "test", "episode": 1, "node_id": str(prefix),
        "step": len(prefix), "prefix_actions": list(prefix),
        "position": position or [sum(prefix), 0, 0], "rotation_xyzw": [0, 0, 0, 1],
    }


class FakeRenderer:
    def start(self, root):
        self.black_frames = 0
        self.restore(root["position"], root["rotation_xyzw"])

    def restore(self, position, rotation):
        self.position, self.rotation = list(position), list(rotation)

    def pose(self):
        return self.position.copy(), self.rotation.copy()

    def current(self):
        return str(self.position[0])

    def step(self, action):
        self.position[0] += action
        return self.current()

    def save(self, frame, path):
        path.write_text(frame)


class ObservationsTest(unittest.TestCase):
    def test_upstream_history(self):
        self.assertEqual(history_positions(0, convert_vln), [0, 0])
        self.assertEqual(history_positions(3, convert_vln), [0, 1, 2, 3])
        steps = history_positions(30, convert_vln)
        self.assertEqual((len(steps), steps[0], steps[-2], steps[-1]), (9, 0, 29, 30))

    def test_reconstructs_intermediate_frames_on_each_action_prefix(self):
        rows = index_episode([row(()), row((1, 2, 1)), row((1, 3, 1))])
        with tempfile.TemporaryDirectory() as directory:
            frames = render_episode(FakeRenderer(), rows, Path(directory) / "rgb")
            self.assertEqual(frames[(1, 2)].read_text(), "3")
            self.assertEqual(frames[(1, 3)].read_text(), "4")
            self.assertEqual(frames[(1, 2, 1)].read_text(), "4")
            self.assertEqual(frames[(1, 3, 1)].read_text(), "5")

    def test_rejects_replay_drift(self):
        rows = index_episode([row(()), row((1, 2, 1), [3, 0, 0])])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Replay pose mismatch"):
                render_episode(FakeRenderer(), rows, Path(directory) / "rgb")

    def test_export_keeps_archive_label_reference_and_reuses_complete_output(self):
        archive = "r2r/shards/000/example.tar"
        rows = [
            dict(row(prefix), instruction="Walk straight.", labels_ref={"offset": index, "size": 8})
            for index, prefix in enumerate([(), (1, 2, 1)])
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tar = root / archive
            tar.parent.mkdir(parents=True)
            tar.touch()
            output = root / "output"
            output.mkdir()
            states = index_episode(rows)
            receipt = prepare_archive(FakeRenderer(), root, output, archive, rows, states, convert_vln, False)
            episode = output / hashlib.sha256(archive.encode()).hexdigest()
            samples = [json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()]
            self.assertEqual(samples[1]["nav2"]["labels_ref"], rows[1]["labels_ref"])
            self.assertEqual(samples[1]["nav2"]["archive"], str(tar))
            self.assertEqual(samples[1]["nav2"]["observation_steps"], [0, 1, 2, 3])
            self.assertEqual(samples[0]["images"][0], samples[0]["images"][1])
            self.assertEqual(receipt, prepare_archive(None, root, output, archive, rows, states, convert_vln, True))


if __name__ == "__main__":
    unittest.main()
